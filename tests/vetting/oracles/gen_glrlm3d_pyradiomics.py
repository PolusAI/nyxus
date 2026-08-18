"""OFFLINE PyRadiomics oracle for the 3D GLRLM features, on the compat phantom.

    python tests/vetting/oracles/gen_glrlm3d_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_glrlm_pyradiomics.h,
exiting non-zero on any mismatch or on any pin it cannot produce.

Recipe `glrlm3d.pyradiomics_bincount20`: the compat phantom
(tests/data/nifti/compat_int/compat_int_mri.nii + compat_seg/compat_seg_liver.nii, label 1) with
binCount=20, no resampling, weightingNorm=None, imageType=Original -- the settings the test header's
provenance block records. On the Nyxus side that is GREYDEPTH=100, IBSI=false, GLRLM_GREYDEPTH=-20
(negative activates radiomics binCount-based binning, so the magnitude is the bin count).

TWO REFERENCES, NOT ONE. PyRadiomics' public API reports one value per feature over its whole
direction set, which is the Nyxus *_AVE aggregation over the 13 3D angles. That scalar cannot vet
the 13 directional values of the unsuffixed base feature: per-direction errors that cancel leave the
mean untouched. So this generator emits BOTH tables --

  glrlm_3d_pyradiomics_ref_vals            the direction-set scalar, reference for the *_AVE features
  glrlm_3d_pyradiomics_ref_vals_by_angle   one value per direction, reference for the base features

The per-direction values come from PyRadiomics' own feature formulas: RadiomicsGLRLM computes each
feature per angle and only then averages (`numpy.nanmean(x, 1)` as the last line of every feature
method), so intercepting that average yields the per-angle vector with nothing reimplemented.

ANGLE ORDER: SLOT k HERE IS SLOT k IN PYRADIOMICS. That is not the obvious answer, and it is the
opposite of the 3D GLCM family, so it is worth stating why.

PyRadiomics returns its angle rows as (dz, dy, dx) -- verified by construction, not assumed: on a
volume that is constant along numpy axis 2 only, the single slot carrying full-length runs is the
one whose row reads (0, 0, 1).

Nyxus' `shifts13` in 3d_glrlm.cpp is written `{1, 1, 1}, {1, 1, 0}, ...`, the same 13 triples in the
same order, and it looks like the GLCM table -- but it is typed `AngleShift`, declared in
texture_feature.h as

    struct AngleShift { int dz, dy, dx; };

while 3D GLCM's identical-looking table is typed `ShiftToNeighbor { int dx, dy, dz; }`. The same
brace initialiser therefore lands in reversed fields: GLRLM's slot 4 `{1,0,0}` means dz=1 (the z
direction) where GLCM's slot 4 means dx=1 (the x direction). Because PyRadiomics also orders its
rows (dz, dy, dx), GLRLM's slots line up with PyRadiomics' one-to-one, and GLCM's need reversing.

The two families therefore label the SAME slot index with different directions. Nothing is missing
from either -- the 13 offsets are a complete set on both sides, which is why the averages agree to
1e-16 and this went unnoticed -- but a caller reading the per-angle vector of a GLCM feature and of
a GLRLM feature is reading two different directions at the same index. That is filed as a defect;
this generator pins what Nyxus actually emits, and says which direction that is.

All 16 public Nyxus 3D GLRLM features have a PyRadiomics counterpart, so this table is complete: the
family has no identity-vetted leftovers the way 3D GLCM does.

Provenance: tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs
Python <= 3.9); generator=tests/vetting/oracles/gen_glrlm3d_pyradiomics.py. Run offline; CI never
invokes it.
"""
import os
import re
import sys

import numpy
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, glrlm, imageoperations

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
INTEN = os.path.join(DATA, "compat_int", "compat_int_mri.nii")
MASK = os.path.join(DATA, "compat_seg", "compat_seg_liver.nii")
TEST_H = os.path.join(TESTS, "test_3d_glrlm_pyradiomics.h")

RELTOL = 1e-1          # matches the C++ assertions (agrees_gt frac_tolerance = 10)
LABEL = 1
BINCOUNT = 20

# Nyxus feature -> PyRadiomics GLRLM feature. Every name maps one-to-one here; unlike GLCM, no pair
# of them collides.
PYRAD = {
    "3GLRLM_GLN": "GrayLevelNonUniformity",
    "3GLRLM_GLNN": "GrayLevelNonUniformityNormalized",
    "3GLRLM_GLV": "GrayLevelVariance",
    "3GLRLM_HGLRE": "HighGrayLevelRunEmphasis",
    "3GLRLM_LGLRE": "LowGrayLevelRunEmphasis",
    "3GLRLM_LRE": "LongRunEmphasis",
    "3GLRLM_LRHGLE": "LongRunHighGrayLevelEmphasis",
    "3GLRLM_LRLGLE": "LongRunLowGrayLevelEmphasis",
    "3GLRLM_RE": "RunEntropy",
    "3GLRLM_RLN": "RunLengthNonUniformity",
    "3GLRLM_RLNN": "RunLengthNonUniformityNormalized",
    "3GLRLM_RP": "RunPercentage",
    "3GLRLM_RV": "RunVariance",
    "3GLRLM_SRE": "ShortRunEmphasis",
    "3GLRLM_SRHGLE": "ShortRunHighGrayLevelEmphasis",
    "3GLRLM_SRLGLE": "ShortRunLowGrayLevelEmphasis",
}

# Mathematical bounds the family's features carry by construction. Checked against the oracle's own
# output, so a misconfigured run is caught before anything is pinned -- and it is the same check that
# found 3GLRLM_RP leaving [0,1] on the Nyxus side, recorded in
# ../audit/glrlm_3d_pyradiomics_vetting_report.md.
BOUNDS = {
    "3GLRLM_SRE": (0.0, 1.0),
    "3GLRLM_LRE": (1.0, None),
    "3GLRLM_RP": (0.0, 1.0),
    "3GLRLM_GLNN": (0.0, 1.0),
    "3GLRLM_RLNN": (0.0, 1.0),
}


# The 13 offsets each tool walks, in ITS OWN slot order, both as (dz, dy, dx): Nyxus' `shifts13`
# (3d_glrlm.cpp) because AngleShift declares its fields in that order, PyRadiomics' `angles` array
# because that is what cMatrices returns.
NYXUS_SHIFTS = [(1, 1, 1), (1, 1, 0), (1, 1, -1), (1, 0, 1), (1, 0, 0), (1, 0, -1),
                (1, -1, 1), (1, -1, 0), (1, -1, -1), (0, 1, 1), (0, 1, 0), (0, 1, -1), (0, 0, 1)]
PYRAD_ANGLES = [(1, 1, 1), (1, 1, 0), (1, 1, -1), (1, 0, 1), (1, 0, 0), (1, 0, -1),
                (1, -1, 1), (1, -1, 0), (1, -1, -1), (0, 1, 1), (0, 1, 0), (0, 1, -1), (0, 0, 1)]


def nyxus_to_pyrad():
    """-> for each Nyxus GLRLM angle slot, the PyRadiomics slot holding the same direction.

    Both are (dz, dy, dx) -- see the module docstring for why GLRLM's AngleShift is that way and
    GLCM's ShiftToNeighbor is not -- so the mapping is the identity. It is still computed rather
    than written down, so that a change to either table is caught here instead of silently
    re-labelling 208 goldens.
    """
    perm = []
    for dz, dy, dx in NYXUS_SHIFTS:              # AngleShift field order: dz, dy, dx
        want = (dz, dy, dx)
        neg = tuple(-c for c in want)
        hit = [k for k, a in enumerate(PYRAD_ANGLES) if a == want or a == neg]
        if len(hit) != 1:
            raise RuntimeError(f"offset {(dz, dy, dx)} maps to {hit}, expected exactly one slot")
        perm.append(hit[0])
    if sorted(perm) != list(range(13)):
        raise RuntimeError(f"angle mapping is not a permutation: {perm}")
    return perm


def per_angle():
    """-> {nyxus feature: [13 PyRadiomics values, in NYXUS angle order]}."""
    img = sitk.ReadImage(INTEN)
    msk = sitk.ReadImage(MASK)
    bb, _ = imageoperations.checkMask(img, msk, label=LABEL)
    img, msk = imageoperations.cropToTumorMask(img, msk, bb)

    f = glrlm.RadiomicsGLRLM(img, msk, binCount=BINCOUNT, label=LABEL, weightingNorm=None,
                             interpolator=sitk.sitkBSpline, resampledPixelSpacing=None,
                             force2D=False)
    f._initCalculation()
    if f.P_glrlm.shape[-1] != 13:
        raise RuntimeError(f"expected 13 angles, got {f.P_glrlm.shape}")

    real_nanmean = numpy.nanmean

    def keep_angles(a, axis=None, **kw):
        a = numpy.asarray(a)
        if axis == 1 or axis == (1, 2, 3):     # the angle-averaging step inside a feature method
            return a.reshape(a.shape[0], -1)
        return real_nanmean(a, axis=axis, **kw)

    perm = nyxus_to_pyrad()
    out = {}
    numpy.nanmean = keep_angles
    try:
        for nyx, pyr in PYRAD.items():
            v = numpy.asarray(getattr(f, f"get{pyr}FeatureValue")()).ravel()
            if v.size != 13:
                raise RuntimeError(f"{pyr}: got {v.size} value(s) per feature, expected 13")
            out[nyx] = [float(v[k]) for k in perm]
    finally:
        numpy.nanmean = real_nanmean
    return out


def parse_pins_by_angle(txt, table):
    """-> {angle index: {feature: value}} out of a ref_vals_map_by_angle literal."""
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found")
    body = txt[m.end():]
    out = {}
    for blk in re.finditer(r"\{\s*(\d+)\s*,\s*\{(.*?)\}\s*\}", body, re.S):
        inner = re.sub(r"//[^\n]*", "", blk.group(2))
        out[int(blk.group(1))] = {
            n: float(v) for n, v in
            re.findall(r'\{\s*"(3GLRLM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', inner)}
        if len(out) == 13:
            break
    return out


def parse_pins(txt, table):
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found")
    body = txt[m.end():].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"(3GLRLM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def run():
    radiomics.logger.setLevel(40)                 # keep progress chatter out of the golden table
    settings = {
        "binCount": BINCOUNT,
        "label": LABEL,
        "interpolator": sitk.sitkBSpline,
        "resampledPixelSpacing": None,
        "weightingNorm": None,
        "force2D": False,
    }
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.disableAllFeatures()
    ex.enableFeatureClassByName("glrlm")
    res = ex.execute(INTEN, MASK)
    out = {}
    for nyx, pyr in PYRAD.items():
        key = f"original_glrlm_{pyr}"
        if key not in res:
            raise RuntimeError(f"PyRadiomics produced no {key} (for {nyx})")
        out[nyx] = float(res[key])
    return out


def check_bounds(got):
    bad = 0
    for name, (lo, hi) in BOUNDS.items():
        v = got[name]
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print(f"  OUT OF RANGE {name} = {v!r}, expected [{lo}, {hi}]")
            bad += 1
    print(f"# {len(BOUNDS) - bad}/{len(BOUNDS)} bounded features in range")
    return bad


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1

    got = run()

    print(f"# pyradiomics {radiomics.__version__}, SimpleITK {sitk.__version__}, "
          f"binCount={BINCOUNT}, label={LABEL}")
    print("# paste-ready goldens")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// original_glrlm_{PYRAD[name]}")

    print()
    nbad = check_bounds(got)

    ang = per_angle()
    print("")
    print("# paste-ready per-direction goldens, in NYXUS angle order (identity; see nyxus_to_pyrad)")
    for k in range(13):
        dz, dy, dx = NYXUS_SHIFTS[k]
        print(f"\t{{{k}, {{   // AngleShift dz={dz} dy={dy} dx={dx}")
        for name in sorted(ang):
            print(f'\t\t{{"{name}", {ang[name][k]!r}}},')
        print("\t}},")

    txt_h = open(TEST_H, encoding="utf-8", errors="replace").read()
    pins_ang = parse_pins_by_angle(txt_h, "glrlm_3d_pyradiomics_ref_vals_by_angle")
    npin = sum(len(v) for v in pins_ang.values())
    print(f"\n# verifying {npin} pinned per-direction goldens against this run")
    nbad = 0
    for k in sorted(pins_ang):
        for name, want in sorted(pins_ang[k].items()):
            have = ang[name][k]
            rel = abs(have - want) / max(abs(want), 1e-12)
            if rel > 1e-12:
                print(f"  FAIL angle {k} {name}: pyradiomics={have!r} pinned={want!r} rel={rel:.3g}")
                nbad += 1
    print("  all per-direction pins reproduce" if not nbad
          else f"  {nbad} per-direction mismatches")

    pins = parse_pins(txt_h, "glrlm_3d_pyradiomics_ref_vals")
    print(f"\n# verifying {len(pins)} pinned goldens against this run")
    nok = nfail = nmiss = 0
    for name in sorted(pins):
        want = pins[name]
        if name not in got:
            print(f"  MISSING {name}: pinned {want!r} but PyRadiomics reports no counterpart")
            nmiss += 1
            continue
        have = got[name]
        rel = abs(have - want) / max(abs(want), 1e-12)
        if rel <= RELTOL:
            print(f"  OK   {name}: pyradiomics={have!r} pinned={want!r} rel={rel:.3g}")
            nok += 1
        else:
            print(f"  FAIL {name}: pyradiomics={have!r} pinned={want!r} rel={rel:.3g}")
            nfail += 1

    missing_pin = sorted(set(got) - set(pins))
    for name in missing_pin:
        print(f"  UNPINNED {name}: PyRadiomics reports {got[name]!r} and nothing pins it")

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible, "
          f"{nbad} per-direction mismatch(es)")
    if nfail or nmiss or nbad:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
