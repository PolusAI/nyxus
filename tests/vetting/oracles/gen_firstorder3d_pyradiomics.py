"""PyRadiomics reference generator for the 3D first-order features.

Runs the oracle the registry claims for these rows and re-verifies every pin in the header it
feeds, exiting non-zero on a mismatch, on a pin it cannot produce, or on an oracle value the header
pins nothing for.

  python gen_firstorder3d_pyradiomics.py [--header <path>] [--emit]

Recipe `firstorder.pyradiomics.bincount20`: binCount 20, label 1, no resampling, no weighting.
PyRadiomics computes first-order features on the original intensities; only Entropy and Uniformity
read the discretized histogram.
"""

import argparse
import os
import sys

import SimpleITK as sitk
from radiomics import firstorder

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, "..", ".."))
INTEN = os.path.join(TESTS, "data", "nifti", "compat_int", "compat_int_mri.nii")
MASK = os.path.join(TESTS, "data", "nifti", "compat_seg", "compat_seg_liver.nii")
LABEL = 1
DEFAULT_HEADER = os.path.join(TESTS, "test_3d_firstorder_pyradiomics.h")
MAP_NAME = "firstorder_3d_pyradiomics_ref_vals"

# Nyxus feature name -> PyRadiomics first-order feature name.
NAME_MAP = {
    "3P10": "10Percentile",
    "3P90": "90Percentile",
    "3ENERGY": "Energy",
    "3ENTROPY": "Entropy",
    "3INTERQUARTILE_RANGE": "InterquartileRange",
    "3KURTOSIS": "Kurtosis",
    "3MAX": "Maximum",
    "3MEAN_ABSOLUTE_DEVIATION": "MeanAbsoluteDeviation",
    "3MEAN": "Mean",
    "3MEDIAN": "Median",
    "3MIN": "Minimum",
    "3RANGE": "Range",
    "3ROBUST_MEAN_ABSOLUTE_DEVIATION": "RobustMeanAbsoluteDeviation",
    "3ROOT_MEAN_SQUARED": "RootMeanSquared",
    "3SKEWNESS": "Skewness",
    "3UNIFORMITY": "Uniformity",
    "3VARIANCE": "Variance",
}


def parse_header_pins(path):
    """Read the golden map out of the header by counting braces, not by a non-greedy regex."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    start = text.index(MAP_NAME)
    start = text.index("{", start)
    depth, end = 0, None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        sys.exit("unterminated golden map in %s" % path)

    pins = {}
    for chunk in text[start + 1:end].split("{")[1:]:
        entry = chunk.split("}")[0]
        name, _, value = entry.partition(",")
        value = value.split("//")[0]
        pins[name.strip().strip('"')] = float(value.strip().rstrip(","))
    return pins


def run_pyradiomics():
    img = sitk.ReadImage(INTEN)
    msk = sitk.ReadImage(MASK)
    settings = {
        "binCount": 20,
        "label": LABEL,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": None,
        "weightingNorm": None,
    }
    fo = firstorder.RadiomicsFirstOrder(img, msk, **settings)
    fo.enableAllFeatures()
    return {k: float(v) for k, v in fo.execute().items()}


def relerr(a, b):
    d = max(abs(a), abs(b))
    return 0.0 if d == 0.0 else abs(a - b) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default=DEFAULT_HEADER)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    print("fixture: %s / %s, label %d" % (os.path.basename(INTEN), os.path.basename(MASK), LABEL))
    result = run_pyradiomics()
    pins = parse_header_pins(args.header)
    print("header : %s  (%d pins)\n" % (os.path.basename(args.header), len(pins)))

    print("%-34s %-30s %22s %22s %11s  %s"
          % ("nyxus feature", "pyradiomics feature", "header pin", "fresh run", "rel", "verdict"))
    print("-" * 132)

    failures, missing = [], []
    for name in sorted(pins):
        pr_name = NAME_MAP.get(name)
        if pr_name is None or pr_name not in result:
            missing.append(name)
            print("%-34s %-30s %22.12g %22s %11s  NOT PRODUCED BY THE ORACLE"
                  % (name, pr_name or "-", pins[name], "-", "-"))
            continue
        got = result[pr_name]
        r = relerr(pins[name], got)
        ok = r <= 1e-12
        if not ok:
            failures.append((name, pins[name], got, r))
        print("%-34s %-30s %22.12g %22.12g %11.3e  %s"
              % (name, pr_name, pins[name], got, r, "matches oracle" if ok else "ROTTED PIN"))

    reverse = [k for k in NAME_MAP if k not in pins]
    print()
    if reverse:
        print("REVERSE CHECK: the oracle produces %d mapped feature(s) the header pins nothing for: %s"
              % (len(reverse), ", ".join(sorted(reverse))))

    print("\nRANGE AND IDENTITY CHECKS")
    checks = []
    if "3UNIFORMITY" in pins:
        checks.append(("3UNIFORMITY in [0,1]", 0.0 <= pins["3UNIFORMITY"] <= 1.0, pins["3UNIFORMITY"]))
    if "3ENTROPY" in pins:
        checks.append(("3ENTROPY >= 0", pins["3ENTROPY"] >= 0.0, pins["3ENTROPY"]))
    if {"3MIN", "3MAX", "3RANGE"} <= set(pins):
        checks.append(("3RANGE == 3MAX - 3MIN",
                       relerr(pins["3RANGE"], pins["3MAX"] - pins["3MIN"]) < 1e-9, pins["3RANGE"]))
    if {"3MIN", "3P10", "3MEDIAN", "3P90", "3MAX"} <= set(pins):
        checks.append(("3MIN <= 3P10 <= 3MEDIAN <= 3P90 <= 3MAX",
                       pins["3MIN"] <= pins["3P10"] <= pins["3MEDIAN"] <= pins["3P90"] <= pins["3MAX"],
                       pins["3MEDIAN"]))
    if {"3VARIANCE", "3ROOT_MEAN_SQUARED", "3MEAN"} <= set(pins):
        # RMS^2 == mean^2 + population variance
        checks.append(("3ROOT_MEAN_SQUARED^2 == 3MEAN^2 + 3VARIANCE",
                       relerr(pins["3ROOT_MEAN_SQUARED"] ** 2,
                              pins["3MEAN"] ** 2 + pins["3VARIANCE"]) < 1e-6,
                       pins["3ROOT_MEAN_SQUARED"]))
    if {"3ROBUST_MEAN_ABSOLUTE_DEVIATION", "3MEAN_ABSOLUTE_DEVIATION"} <= set(pins):
        checks.append(("3ROBUST_MEAN_ABSOLUTE_DEVIATION <= 3MEAN_ABSOLUTE_DEVIATION",
                       pins["3ROBUST_MEAN_ABSOLUTE_DEVIATION"] <= pins["3MEAN_ABSOLUTE_DEVIATION"],
                       pins["3ROBUST_MEAN_ABSOLUTE_DEVIATION"]))

    bad = 0
    for label, ok, value in checks:
        print("  [%s] %-58s  value=%.10g" % ("PASS" if ok else "FAIL", label, value))
        if not ok:
            bad += 1

    if args.emit:
        print()
        for name in sorted(NAME_MAP):
            pr_name = NAME_MAP[name]
            if pr_name in result:
                print('    {"%s", %.17g}, // Case-1_original_firstorder_%s'
                      % (name, result[pr_name], pr_name))

    print("\nSUMMARY: %d pins, %d mismatched, %d not produced, %d unpinned, %d failed checks"
          % (len(pins), len(failures), len(missing), len(reverse), bad))
    if failures or missing or bad:
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
