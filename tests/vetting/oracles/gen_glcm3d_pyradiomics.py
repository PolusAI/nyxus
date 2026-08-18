"""OFFLINE PyRadiomics oracle for the 3D GLCM features, on the compat phantom.

    python tests/vetting/oracles/gen_glcm3d_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_glcm_pyradiomics.h,
exiting non-zero on any mismatch or on any pin it cannot produce.

Recipe `glcm3d.pyradiomics_bincount20`: the compat phantom
(tests/data/nifti/compat_int/compat_int_mri.nii + compat_seg/compat_seg_liver.nii, label 1) with
binCount=20, no resampling, weightingNorm=None, imageType=Original -- the settings the test header's
provenance block records. On the Nyxus side that is GREYDEPTH=100, IBSI=false, GLCM_GREYDEPTH=-20
(negative activates radiomics binCount-based binning), GLCM_OFFSET=1, GLCM_SPARSEINTENS=true.

TWO REFERENCES, NOT ONE. PyRadiomics' public API reports one value per feature over its whole
direction set, which is the Nyxus *_AVE aggregation over the 13 3D angles. That scalar cannot vet
the 13 directional values of the unsuffixed base feature: per-direction errors that cancel leave
the mean untouched. So this generator emits BOTH tables --

  glcm_3d_pyradiomics_ref_vals            the direction-set scalar, reference for the *_AVE features
  glcm_3d_pyradiomics_ref_vals_by_angle   one value per direction, reference for the base features

The per-direction values come from PyRadiomics' own feature formulas: RadiomicsGLCM computes each
feature per angle and only then averages, so intercepting that final average yields the per-angle
vector without reimplementing anything (see per_angle() below).

ANGLE ORDER. Both tools walk the same 13 offsets in the same order, but PyRadiomics rows are
(dz, dy, dx) while Nyxus' `shifts` table (3d_glcm.cpp) is (dx, dy, dz). Reversing a Nyxus triple
therefore lands on a PyRadiomics row, up to an overall sign -- and a sign flip is the same
unordered pixel pair, hence the same symmetric cooccurrence matrix. NYXUS_TO_PYRAD below is derived
from those two lists rather than guessed, and the assertion would fail loudly if the mapping were
wrong: the 13 values of a feature like ACOR are all distinct.

SYMMETRY. Nyxus symmetrises the cooccurrence matrix whenever radiomics grey binning is active
(3d_glcm.cpp: `if (symmetric_glcm || radiomics_grey_binning(greyInfo) || ibsi_grey_binning(...))`),
and GLCM_GREYDEPTH=-20 is exactly that path, so both sides are symmetric here. An older comment in
the test header claimed Nyxus was asymmetric and used a 10% band to cover the difference; there is
no such difference to cover, and the measured residuals below are the real ones.

Six Nyxus features have no PyRadiomics counterpart and are deliberately absent: DIS (PyRadiomics
deprecates it as equivalent to DifferenceAverage), ENERGY, ENTROPY, HOM1, SUMVARIANCE and VARIANCE.
They are vetted through identities against twins that do appear here -- see
test_3d_glcm_ave_equivalence_pyradiomics(). PyRadiomics' MCC has no Nyxus counterpart.

Provenance: tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs
Python <= 3.9); generator=tests/vetting/oracles/gen_glcm3d_pyradiomics.py. Run offline; CI never
invokes it.
"""
import os
import re
import sys

import numpy
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, glcm, imageoperations

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
INTEN = os.path.join(DATA, "compat_int", "compat_int_mri.nii")
MASK = os.path.join(DATA, "compat_seg", "compat_seg_liver.nii")
TEST_H = os.path.join(TESTS, "test_3d_glcm_pyradiomics.h")

RELTOL = 1e-9          # measured residual is <= 1.2e-15 everywhere except the entropy
                       # family, whose per-feature bands live in the test header
LABEL = 1
BINCOUNT = 20

# Nyxus feature -> PyRadiomics GLCM feature. Three of these do NOT line up by name:
# ASM is PyRadiomics' JointEnergy, JVAR is its SumSquares, JMAX its MaximumProbability.
PYRAD = {
    "3GLCM_ACOR": "Autocorrelation",
    "3GLCM_ASM": "JointEnergy",
    "3GLCM_CLUPROM": "ClusterProminence",
    "3GLCM_CLUSHADE": "ClusterShade",
    "3GLCM_CLUTEND": "ClusterTendency",
    "3GLCM_CONTRAST": "Contrast",
    "3GLCM_CORRELATION": "Correlation",
    "3GLCM_DIFAVE": "DifferenceAverage",
    "3GLCM_DIFENTRO": "DifferenceEntropy",
    "3GLCM_DIFVAR": "DifferenceVariance",
    "3GLCM_ID": "Id",
    "3GLCM_IDN": "Idn",
    "3GLCM_IDM": "Idm",
    "3GLCM_IDMN": "Idmn",
    "3GLCM_INFOMEAS1": "Imc1",
    "3GLCM_INFOMEAS2": "Imc2",
    "3GLCM_IV": "InverseVariance",
    "3GLCM_JAVE": "JointAverage",
    "3GLCM_JE": "JointEntropy",
    "3GLCM_JMAX": "MaximumProbability",
    "3GLCM_JVAR": "SumSquares",
    "3GLCM_SUMAVERAGE": "SumAverage",
    "3GLCM_SUMENTROPY": "SumEntropy",
}


# The 13 offsets each tool walks, in ITS OWN slot order. Nyxus: `shifts` in 3d_glcm.cpp, as
# (dx, dy, dz). PyRadiomics: the `angles` array cMatrices.calculate_glcm returns, as (dz, dy, dx).
NYXUS_SHIFTS = [(1, 1, 1), (1, 1, 0), (1, 1, -1), (1, 0, 1), (1, 0, 0), (1, 0, -1),
                (1, -1, 1), (1, -1, 0), (1, -1, -1), (0, 1, 1), (0, 1, 0), (0, 1, -1), (0, 0, 1)]
PYRAD_ANGLES = [(1, 1, 1), (1, 1, 0), (1, 1, -1), (1, 0, 1), (1, 0, 0), (1, 0, -1),
                (1, -1, 1), (1, -1, 0), (1, -1, -1), (0, 1, 1), (0, 1, 0), (0, 1, -1), (0, 0, 1)]


def nyxus_to_pyrad():
    """-> for each Nyxus angle slot, the PyRadiomics slot holding the same direction.

    A Nyxus (dx, dy, dz) reversed is a PyRadiomics (dz, dy, dx); the negated offset is the same
    unordered pixel pair and so the same symmetric matrix, which is why the sign is matched too.
    """
    perm = []
    for dx, dy, dz in NYXUS_SHIFTS:
        want = (dz, dy, dx)
        neg = tuple(-c for c in want)
        hit = [k for k, a in enumerate(PYRAD_ANGLES) if a == want or a == neg]
        if len(hit) != 1:
            raise RuntimeError(f"offset {(dx, dy, dz)} maps to {hit}, expected exactly one slot")
        perm.append(hit[0])
    if sorted(perm) != list(range(13)):
        raise RuntimeError(f"angle mapping is not a permutation: {perm}")
    return perm


def per_angle():
    """-> {nyxus feature: [13 PyRadiomics values, in NYXUS angle order]}.

    RadiomicsGLCM computes every feature per angle and averages as its last step, so replacing that
    average with the identity hands back PyRadiomics' own per-angle numbers. Two features reduce
    over a tuple of axes (Correlation) or read a coefficient directly (JointAverage), which is why
    the patch handles both spellings rather than only `axis=1`.
    """
    img = sitk.ReadImage(INTEN)
    msk = sitk.ReadImage(MASK)
    bb, _ = imageoperations.checkMask(img, msk, label=LABEL)
    img, msk = imageoperations.cropToTumorMask(img, msk, bb)

    f = glcm.RadiomicsGLCM(img, msk, binCount=BINCOUNT, label=LABEL, weightingNorm=None,
                           interpolator=sitk.sitkBSpline, resampledPixelSpacing=None, force2D=False)
    f._initCalculation()
    if f.P_glcm.shape[-1] != 13:
        raise RuntimeError(f"expected 13 angles, got {f.P_glcm.shape}")

    real_nanmean = numpy.nanmean

    def keep_angles(a, axis=None, **kw):
        a = numpy.asarray(a)
        if axis == 1 or axis == (1, 2, 3):        # the angle-averaging step inside a feature method
            return a.reshape(a.shape[0], -1)
        return real_nanmean(a, axis=axis, **kw)

    perm = nyxus_to_pyrad()
    out = {}
    numpy.nanmean = keep_angles
    try:
        for nyx, pyr in PYRAD.items():
            if pyr == "JointAverage":
                # its method averages with ndarray.mean((1,2,3)) over the ux coefficient rather than
                # numpy.nanmean, so the patch above cannot see it; ux IS the per-angle joint average
                v = numpy.asarray(f.coefficients["ux"]).ravel()
            else:
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
            re.findall(r'\{\s*"(3GLCM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', inner)}
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
            re.findall(r'\{\s*"(3GLCM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


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
    ex.enableFeatureClassByName("glcm")
    res = ex.execute(INTEN, MASK)
    out = {}
    for nyx, pyr in PYRAD.items():
        key = f"original_glcm_{pyr}"
        if key not in res:
            raise RuntimeError(f"PyRadiomics produced no {key} (for {nyx})")
        out[nyx] = float(res[key])
    return out


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
        print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// original_glcm_{PYRAD[name]}")

    ang = per_angle()
    print("")
    print("# paste-ready per-direction goldens, in NYXUS angle order (see nyxus_to_pyrad)")
    for k in range(13):
        dx, dy, dz = NYXUS_SHIFTS[k]
        print(f"\t{{{k}, {{   // Nyxus shift ({dx},{dy},{dz})")
        for name in sorted(ang):
            print(f'\t\t{{"{name}", {ang[name][k]!r}}},')
        print("\t}},")

    txt_h = open(TEST_H, encoding="utf-8", errors="replace").read()
    pins_ang = parse_pins_by_angle(txt_h, "glcm_3d_pyradiomics_ref_vals_by_angle")
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

    pins = parse_pins(txt_h, "glcm_3d_pyradiomics_ref_vals")
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

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible, "
          f"{nbad} per-direction mismatch(es)")
    if nfail or nmiss or nbad:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
