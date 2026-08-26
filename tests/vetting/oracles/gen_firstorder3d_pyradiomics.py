"""Reproduce the PyRadiomics pins for the 3D first-order oracle tests.

Runs the oracle named by the registry and verifies every pin in
tests/test_3d_firstorder_pyradiomics.h. The command exits non-zero if a pin has changed, cannot be
produced, or is missing for a mapped PyRadiomics first-order feature.

    python gen_firstorder3d_pyradiomics.py [--header <path>] [--emit]

Recipe ``firstorder3d.pyradiomics_bincount20``: binCount 20, label 1, no resampling, and no
weighting. PyRadiomics computes first-order features from the original intensities; only Entropy
and Uniformity use the discretized histogram.
"""

import argparse
import os
import sys
from importlib.metadata import version

import SimpleITK as sitk
from radiomics import firstorder


HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, "..", ".."))
INTENSITY = os.path.join(TESTS, "data", "nifti", "compat_int", "compat_int_mri.nii")
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
    """Read the golden map by balancing braces instead of relying on a non-greedy regex."""
    with open(path, "r", encoding="utf-8") as stream:
        text = stream.read()

    start = text.index(MAP_NAME)
    start = text.index("{", start)
    depth = 0
    end = None
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
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
    image = sitk.ReadImage(INTENSITY)
    mask = sitk.ReadImage(MASK)
    settings = {
        "binCount": 20,
        "label": LABEL,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": None,
        "weightingNorm": None,
    }
    oracle = firstorder.RadiomicsFirstOrder(image, mask, **settings)
    oracle.enableAllFeatures()
    return {name: float(value) for name, value in oracle.execute().items()}


def relative_error(left, right):
    denominator = max(abs(left), abs(right))
    return 0.0 if denominator == 0.0 else abs(left - right) / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", default=DEFAULT_HEADER)
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()

    print("oracle : PyRadiomics %s / SimpleITK %s / NumPy %s / Python %d.%d.%d" % (
        version("pyradiomics"),
        version("SimpleITK"),
        version("numpy"),
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro))
    print("fixture: %s / %s, label %d" % (
        os.path.basename(INTENSITY), os.path.basename(MASK), LABEL))
    result = run_pyradiomics()
    pins = parse_header_pins(args.header)
    print("header : %s  (%d pins)\n" % (os.path.basename(args.header), len(pins)))

    print("%-34s %-30s %22s %22s %11s  %s" % (
        "nyxus feature", "pyradiomics feature", "header pin", "fresh run", "rel", "verdict"))
    print("-" * 132)

    failures = []
    missing = []
    for name in sorted(pins):
        oracle_name = NAME_MAP.get(name)
        if oracle_name is None or oracle_name not in result:
            missing.append(name)
            print("%-34s %-30s %22.12g %22s %11s  NOT PRODUCED BY THE ORACLE" % (
                name, oracle_name or "-", pins[name], "-", "-"))
            continue

        value = result[oracle_name]
        error = relative_error(pins[name], value)
        matches = error <= 1.0e-12
        if not matches:
            failures.append((name, pins[name], value, error))
        print("%-34s %-30s %22.12g %22.12g %11.3e  %s" % (
            name,
            oracle_name,
            pins[name],
            value,
            error,
            "matches oracle" if matches else "ROTTED PIN"))

    unpinned = [name for name in NAME_MAP if name not in pins]
    if unpinned:
        print("\nREVERSE CHECK: %d mapped feature(s) have no header pin: %s" % (
            len(unpinned), ", ".join(sorted(unpinned))))

    print("\nRANGE AND IDENTITY CHECKS")
    checks = []
    if "3UNIFORMITY" in pins:
        checks.append((
            "3UNIFORMITY in [0,1]",
            0.0 <= pins["3UNIFORMITY"] <= 1.0,
            pins["3UNIFORMITY"]))
    if "3ENTROPY" in pins:
        checks.append(("3ENTROPY >= 0", pins["3ENTROPY"] >= 0.0, pins["3ENTROPY"]))
    if {"3MIN", "3MAX", "3RANGE"} <= set(pins):
        checks.append((
            "3RANGE == 3MAX - 3MIN",
            relative_error(pins["3RANGE"], pins["3MAX"] - pins["3MIN"]) < 1.0e-9,
            pins["3RANGE"]))
    if {"3MIN", "3P10", "3MEDIAN", "3P90", "3MAX"} <= set(pins):
        checks.append((
            "3MIN <= 3P10 <= 3MEDIAN <= 3P90 <= 3MAX",
            pins["3MIN"] <= pins["3P10"] <= pins["3MEDIAN"] <= pins["3P90"] <= pins["3MAX"],
            pins["3MEDIAN"]))
    if {"3VARIANCE", "3ROOT_MEAN_SQUARED", "3MEAN"} <= set(pins):
        checks.append((
            "3ROOT_MEAN_SQUARED^2 == 3MEAN^2 + 3VARIANCE",
            relative_error(
                pins["3ROOT_MEAN_SQUARED"] ** 2,
                pins["3MEAN"] ** 2 + pins["3VARIANCE"]) < 1.0e-6,
            pins["3ROOT_MEAN_SQUARED"]))
    if {"3ROBUST_MEAN_ABSOLUTE_DEVIATION", "3MEAN_ABSOLUTE_DEVIATION"} <= set(pins):
        checks.append((
            "3ROBUST_MEAN_ABSOLUTE_DEVIATION <= 3MEAN_ABSOLUTE_DEVIATION",
            pins["3ROBUST_MEAN_ABSOLUTE_DEVIATION"] <= pins["3MEAN_ABSOLUTE_DEVIATION"],
            pins["3ROBUST_MEAN_ABSOLUTE_DEVIATION"]))

    bad_checks = 0
    for label, passed, value in checks:
        print("  [%s] %-58s  value=%.10g" % (
            "PASS" if passed else "FAIL", label, value))
        if not passed:
            bad_checks += 1

    if args.emit:
        print()
        for name in sorted(NAME_MAP):
            oracle_name = NAME_MAP[name]
            if oracle_name in result:
                print('    {"%s", %.17g}, // Case-1_original_firstorder_%s' % (
                    name, result[oracle_name], oracle_name))

    print("\nSUMMARY: %d pins, %d mismatched, %d not produced, %d unpinned, %d failed checks" % (
        len(pins), len(failures), len(missing), len(unpinned), bad_checks))
    if failures or missing or unpinned or bad_checks:
        sys.exit(1)


if __name__ == "__main__":
    main()
