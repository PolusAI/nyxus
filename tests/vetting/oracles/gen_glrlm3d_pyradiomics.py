"""OFFLINE PyRadiomics oracle for the 3D GLRLM features, on the compat phantom.

    python tests/vetting/oracles/gen_glrlm3d_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_glrlm_pyradiomics.h,
exiting non-zero on any mismatch or on any pin it cannot produce.

Recipe `glrlm3d.pyradiomics_bincount20`: the compat phantom
(tests/data/nifti/compat_int/compat_int_mri.nii + compat_seg/compat_seg_liver.nii, label 1) with
binCount=20, no resampling, weightingNorm=None, imageType=Original -- the settings the test header's
provenance block records. On the Nyxus side that is GREYDEPTH=100, IBSI=false, GLRLM_GREYDEPTH=-20
(negative activates radiomics binCount-based binning, so the magnitude is the bin count).

THE CONVENTION THAT MATTERS: PyRadiomics reports ONE value per feature over its whole direction set.
That is the Nyxus *_AVE aggregation over the 13 3D angles, not a per-angle value -- so each golden
below is the reference for both the per-angle base feature (through calc_ave) and the stored *_AVE
feature.

All 16 public Nyxus 3D GLRLM features have a PyRadiomics counterpart, so this table is complete: the
family has no identity-vetted leftovers the way 3D GLCM does.

Provenance: tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs
Python <= 3.9); generator=tests/vetting/oracles/gen_glrlm3d_pyradiomics.py. Run offline; CI never
invokes it.
"""
import os
import re
import sys

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

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

    pins = parse_pins(open(TEST_H, encoding="utf-8", errors="replace").read(),
                      "glrlm_3d_pyradiomics_ref_vals")
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

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible, {len(missing_pin)} unpinned")
    if nfail or nmiss or nbad:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
