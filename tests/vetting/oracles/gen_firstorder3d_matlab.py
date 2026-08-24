"""MATLAB-semantics (GNU Octave) reference generator for the 3D first-order features.

Runs the oracle and re-verifies every pin in the header it feeds, exiting non-zero on any
mismatch, on a pin it cannot produce, or on an oracle value the header pins nothing for.

  python gen_firstorder3d_matlab.py --octave <octave-cli> [--header <path>] [--emit]

THE LOADER DOMAIN
-----------------
Nyxus does not featurize the stored NIfTI voxels. `NiftiLoader::unhounsfield` (src/nyx/raw_nifti.h)
scans the WHOLE volume and, when its minimum is negative, shifts every voxel by -min before the
cast to the unsigned voxel buffer truncates it. `ut_inten.nii` has a whole-volume minimum of -1024,
so the label-57 ROI Nyxus actually measures is trunc(stored + 1024), spanning [1024, 3024].

The oracle therefore runs on trunc(stored - whole_volume_min), reproduced here from the stored
voxels via SimpleITK. Comparing against the stored voxels instead would put every location
statistic 1024 off and none of the shift-invariant ones, which is a loader question rather than a
first-order one (tests/vetting/not_covered.md).
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, "..", ".."))
INTEN = os.path.join(TESTS, "data", "nifti", "phantoms", "ut_inten.nii")
MASK = os.path.join(TESTS, "data", "nifti", "phantoms", "ut_mask57.nii")
LABEL = 57
DEFAULT_HEADER = os.path.join(TESTS, "test_3d_firstorder_matlab.h")
MAP_NAME = "firstorder_3d_matlab_ref_vals"

# Features whose Nyxus value comes from the 100-bin interpolated histogram in
# TrivialHistogram::calc_percentiles rather than from an exact order statistic, plus the ones
# derived from those percentiles. The oracle emits the exact statistic for each; the divergence is
# a definitional difference and is reported, never silently absorbed into a tolerance.
HISTOGRAM_DERIVED = {
    "3P01", "3P10", "3P25", "3P75", "3P90", "3P99",
    "3INTERQUARTILE_RANGE", "3QCOD", "3ROBUST_MEAN",
    "3ROBUST_MEAN_ABSOLUTE_DEVIATION",
}

# DEFAULT_NUM_HISTO_BINS (src/nyx/constants.h). A default-constructed Fsettings leaves
# STNGS_MISSING true, so 3d_intensity.cpp bins the histogram statistics at this many levels.
HISTO_BINS = 24

# Features the oracle cannot produce from the voxel vector alone, with the reason.
NOT_PRODUCIBLE = {
    "3COVERED_IMAGE_INTENSITY_RANGE":
        "needs SlideProps whole-slide min/max, not an ROI statistic",
}


def loader_domain_voxels():
    """Reproduce the voxel vector Nyxus featurizes for label 57."""
    a = sitk.GetArrayFromImage(sitk.ReadImage(INTEN))
    m = sitk.GetArrayFromImage(sitk.ReadImage(MASK))
    if a.shape != m.shape:
        sys.exit("intensity/mask shape mismatch")
    vol_min = float(a.min())
    roi = a[m == LABEL].astype(np.float64)
    shifted = roi - vol_min if vol_min < 0.0 else roi
    return np.trunc(shifted), vol_min


def parse_header_pins(path):
    """Read the golden map out of the header by counting braces.

    A non-greedy regex swallows the last entry's closing brace; counting does not.
    """
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
    body = text[start + 1:end]

    pins = {}
    for chunk in body.split("{")[1:]:
        entry = chunk.split("}")[0]
        name, _, value = entry.partition(",")
        pins[name.strip().strip('"')] = float(value.strip().rstrip(","))
    return pins


def run_octave(octave, voxels):
    with tempfile.TemporaryDirectory() as td:
        csv = os.path.join(td, "voxels.csv")
        np.savetxt(csv, voxels, fmt="%.17g")
        script = os.path.join(HERE, "gen_firstorder3d_matlab.m")
        proc = subprocess.run(
            [octave, "--quiet", "--no-gui", script, csv, str(HISTO_BINS)],
            capture_output=True, text=True)
    if proc.returncode != 0 or "OCTAVE_FO3D_DONE" not in proc.stdout:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        sys.exit("octave run failed")
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line and not line.startswith("warning"):
            k, _, v = line.partition("=")
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def relerr(a, b):
    d = max(abs(a), abs(b))
    return 0.0 if d == 0.0 else abs(a - b) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octave", required=True, help="path to octave-cli")
    ap.add_argument("--header", default=DEFAULT_HEADER)
    ap.add_argument("--emit", action="store_true", help="print a ready-to-paste golden map")
    args = ap.parse_args()

    voxels, vol_min = loader_domain_voxels()
    print("fixture      : ut_inten.nii / ut_mask57.nii, label %d" % LABEL)
    print("whole-vol min: %.17g  ->  loader shift %+.17g" % (vol_min, -vol_min))
    print("ROI          : n=%d  min=%.17g  max=%.17g" % (voxels.size, voxels.min(), voxels.max()))

    oracle = run_octave(args.octave, voxels)
    if int(oracle.get("N", -1)) != voxels.size:
        sys.exit("octave saw %r voxels, expected %d" % (oracle.get("N"), voxels.size))
    pins = parse_header_pins(args.header)
    print("header       : %s  (%d pins)\n" % (os.path.basename(args.header), len(pins)))

    print("%-34s %20s %20s %11s  %s" % ("feature", "header pin", "octave", "rel", "verdict"))
    print("-" * 104)

    failures, unproducible, reverse = [], [], []
    for name in sorted(pins):
        pin = pins[name]
        if name in NOT_PRODUCIBLE:
            unproducible.append(name)
            print("%-34s %20.10g %20s %11s  NOT PRODUCIBLE (%s)"
                  % (name, pin, "-", "-", NOT_PRODUCIBLE[name]))
            continue
        if name not in oracle:
            failures.append(name)
            print("%-34s %20.10g %20s %11s  MISSING FROM ORACLE" % (name, pin, "-", "-"))
            continue
        got = oracle[name]
        r = relerr(pin, got)
        # The header pins ARE this oracle's output, so anything above round-trip noise is a rotted
        # pin, not a disagreement. The band the C++ test asserts at is a separate question - it has
        # to cover Nyxus-vs-oracle, which is measured in the audit report, not here.
        ok = r <= 1e-12
        if not ok:
            failures.append(name)
        note = "  [binned estimator vs exact order statistic]" if name in HISTOGRAM_DERIVED else ""
        print("%-34s %20.10g %20.10g %11.3e  %s%s"
              % (name, pin, got, r, "matches oracle" if ok else "ROTTED PIN", note))

    for name in sorted(oracle):
        if name.startswith("3") and name not in pins:
            reverse.append(name)

    print()
    if reverse:
        print("REVERSE CHECK: oracle produces %d feature(s) the header pins nothing for: %s"
              % (len(reverse), ", ".join(reverse)))
    if unproducible:
        print("UNPRODUCIBLE : %d pin(s) the oracle cannot back: %s"
              % (len(unproducible), ", ".join(unproducible)))

    # Range / identity checks the definitions force, run over the pins mechanically.
    print("\nRANGE AND IDENTITY CHECKS")
    checks = []
    if "3UNIFORMITY" in pins:
        checks.append(("3UNIFORMITY in [0,1] (sum of squared bin probabilities)",
                       0.0 <= pins["3UNIFORMITY"] <= 1.0, pins["3UNIFORMITY"]))
    if "3ENTROPY" in pins:
        checks.append(("3ENTROPY >= 0", pins["3ENTROPY"] >= 0.0, pins["3ENTROPY"]))
    if "3UNIFORMITY_PIU" in pins:
        checks.append(("3UNIFORMITY_PIU in [0,100]",
                       0.0 <= pins["3UNIFORMITY_PIU"] <= 100.0, pins["3UNIFORMITY_PIU"]))
    if {"3MIN", "3MAX", "3RANGE"} <= set(pins):
        checks.append(("3RANGE == 3MAX - 3MIN",
                       relerr(pins["3RANGE"], pins["3MAX"] - pins["3MIN"]) < 1e-2,
                       pins["3RANGE"]))
    if {"3VARIANCE", "3STANDARD_DEVIATION"} <= set(pins):
        checks.append(("3STANDARD_DEVIATION == sqrt(3VARIANCE)",
                       relerr(pins["3STANDARD_DEVIATION"], pins["3VARIANCE"] ** 0.5) < 1e-2,
                       pins["3STANDARD_DEVIATION"]))
    if {"3VARIANCE_BIASED", "3STANDARD_DEVIATION_BIASED"} <= set(pins):
        checks.append(("3STANDARD_DEVIATION_BIASED == sqrt(3VARIANCE_BIASED)",
                       relerr(pins["3STANDARD_DEVIATION_BIASED"],
                              pins["3VARIANCE_BIASED"] ** 0.5) < 1e-2,
                       pins["3STANDARD_DEVIATION_BIASED"]))
    if {"3EXCESS_KURTOSIS", "3KURTOSIS"} <= set(pins):
        checks.append(("3EXCESS_KURTOSIS == 3KURTOSIS - 3",
                       relerr(pins["3EXCESS_KURTOSIS"], pins["3KURTOSIS"] - 3.0) < 1e-2,
                       pins["3EXCESS_KURTOSIS"]))
    if {"3MEAN", "3INTEGRATED_INTENSITY"} <= set(pins):
        checks.append(("3MEAN == 3INTEGRATED_INTENSITY / n",
                       relerr(pins["3MEAN"], pins["3INTEGRATED_INTENSITY"] / voxels.size) < 1e-2,
                       pins["3MEAN"]))
    if {"3P25", "3P75", "3INTERQUARTILE_RANGE"} <= set(pins):
        checks.append(("3INTERQUARTILE_RANGE == 3P75 - 3P25",
                       relerr(pins["3INTERQUARTILE_RANGE"], pins["3P75"] - pins["3P25"]) < 1e-2,
                       pins["3INTERQUARTILE_RANGE"]))
    if {"3P25", "3P75", "3QCOD"} <= set(pins):
        checks.append(("3QCOD == (3P75-3P25)/(3P75+3P25)",
                       relerr(pins["3QCOD"],
                              (pins["3P75"] - pins["3P25"]) / (pins["3P75"] + pins["3P25"])) < 1e-2,
                       pins["3QCOD"]))
    if {"3MIN", "3P01", "3P99", "3MAX"} <= set(pins):
        checks.append(("3MIN <= 3P01 <= 3P99 <= 3MAX",
                       pins["3MIN"] <= pins["3P01"] <= pins["3P99"] <= pins["3MAX"],
                       pins["3P01"]))
    if {"3ROBUST_MEAN", "3P10", "3P90"} <= set(pins):
        checks.append(("3P10 <= 3ROBUST_MEAN <= 3P90",
                       pins["3P10"] <= pins["3ROBUST_MEAN"] <= pins["3P90"],
                       pins["3ROBUST_MEAN"]))

    bad = 0
    for label, ok, value in checks:
        print("  [%s] %-58s  value=%.10g" % ("PASS" if ok else "FAIL", label, value))
        if not ok:
            bad += 1

    if args.emit:
        print("\nstatic ref_vals_map<double> %s\n{" % MAP_NAME)
        for name in sorted(oracle):
            if name.startswith("3") and name not in NOT_PRODUCIBLE:
                print('\t{ "%s",\t%.17g },' % (name, oracle[name]))
        print("};")

    print("\nSUMMARY: %d pins, %d mismatched, %d unproducible, %d unpinned oracle values, "
          "%d failed range/identity checks" % (len(pins), len(failures), len(unproducible),
                                               len(reverse), bad))
    if failures or bad or reverse or unproducible:
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
