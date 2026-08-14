"""OFFLINE PyRadiomics oracle for the 3D GLCM features, on the compat phantom.

    python tests/vetting/oracles/gen_glcm3d_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_glcm_pyradiomics.h,
exiting non-zero on any mismatch or on any pin it cannot produce.

Recipe `glcm3d.pyradiomics_bincount20`: the compat phantom
(tests/data/nifti/compat_int/compat_int_mri.nii + compat_seg/compat_seg_liver.nii, label 1) with
binCount=20, no resampling, weightingNorm=None, imageType=Original -- the settings the test header's
provenance block records. On the Nyxus side that is GREYDEPTH=100, IBSI=false, GLCM_GREYDEPTH=-20
(negative activates radiomics binCount-based binning), GLCM_OFFSET=1, GLCM_SPARSEINTENS=true.

THE CONVENTION THAT MATTERS: PyRadiomics reports ONE value per feature over its whole direction set.
That is the Nyxus *_AVE aggregation over the 13 3D angles, not a per-angle value -- so each golden
below is the reference for both the per-angle base feature (through calc_ave) and the stored *_AVE
feature. Nyxus' GLCM is asymmetric with offset 1; PyRadiomics' is symmetric by default, which is why
the agreement band is 10% rather than exact.

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

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
INTEN = os.path.join(DATA, "compat_int", "compat_int_mri.nii")
MASK = os.path.join(DATA, "compat_seg", "compat_seg_liver.nii")
TEST_H = os.path.join(TESTS, "test_3d_glcm_pyradiomics.h")

RELTOL = 1e-1          # matches the C++ assertions (agrees_gt frac_tolerance = 10)
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

    pins = parse_pins(open(TEST_H, encoding="utf-8", errors="replace").read(),
                      "glcm_3d_pyradiomics_ref_vals")
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

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible")
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
