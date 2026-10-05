"""PyRadiomics reference generator for 3D shape on a non-cubic voxel grid.

Runs the oracle the registry claims for these rows and re-verifies every pin in the header it
feeds, exiting non-zero on a mismatch, on a pin it cannot produce, or on an oracle value the header
pins nothing for.

  python gen_morphology3d_pyradiomics.py [--header <path>] [--emit]

Recipe `morphology3d.pyradiomics_anisotropic`: the segmented phantom (ut_inten.nii / ut_mask57.nii,
label 57) with the voxel spacing (1.3, 0.8, 2.5) in x, y, z set on both SimpleITK images, origin 0
and identity direction, through shape.RadiomicsShape. PyRadiomics builds its marching-cubes mesh and
its covariance on the voxels as acquired, scaled by that spacing; Nyxus is given the same spacing as
--aniso*.

The header pins PyRadiomics' own values. The axis lengths are 4*sqrt of the eigenvalues of the
covariance of the physical voxel coordinates: PyRadiomics normalizes that covariance by the voxel
count N and Nyxus by N-1, so the test compares Nyxus with these pins times sqrt(N/(N-1)), and the
ratios 3ELONGATION and 3FLATNESS cancel the factor.
"""

import argparse
import math
import os
import sys

import SimpleITK as sitk
from radiomics import shape

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, "..", ".."))
INTEN = os.path.join(TESTS, "data", "nifti", "phantoms", "ut_inten.nii")
MASK = os.path.join(TESTS, "data", "nifti", "phantoms", "ut_mask57.nii")
LABEL = 57
SPACING = (1.3, 0.8, 2.5)
DEFAULT_HEADER = os.path.join(TESTS, "test_3d_morphology_pyradiomics.h")
MAP_NAME = "morphology_3d_pyradiomics_ref_vals"

# Nyxus feature name -> PyRadiomics shape feature name.
NAME_MAP = {
    "3MESH_VOLUME": "MeshVolume",
    "3MAJOR_AXIS_LEN": "MajorAxisLength",
    "3MINOR_AXIS_LEN": "MinorAxisLength",
    "3LEAST_AXIS_LEN": "LeastAxisLength",
    "3ELONGATION": "Elongation",
    "3FLATNESS": "Flatness",
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


def run_pyradiomics(spacing):
    img = sitk.ReadImage(INTEN)
    msk = sitk.ReadImage(MASK)
    for im in (img, msk):
        im.SetSpacing(spacing)
        im.SetOrigin((0.0, 0.0, 0.0))
        im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    sh = shape.RadiomicsShape(img, msk, label=LABEL)
    sh.enableAllFeatures()
    return {k: float(v) for k, v in sh.execute().items()}


def relerr(a, b):
    d = max(abs(a), abs(b))
    return 0.0 if d == 0.0 else abs(a - b) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default=DEFAULT_HEADER)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    print("fixture: %s / %s, label %d, spacing %s"
          % (os.path.basename(INTEN), os.path.basename(MASK), LABEL, SPACING))
    result = run_pyradiomics(SPACING)
    unit = run_pyradiomics((1.0, 1.0, 1.0))
    pins = parse_header_pins(args.header)
    print("header : %s  (%d pins)\n" % (os.path.basename(args.header), len(pins)))

    print("%-20s %-18s %22s %22s %11s  %s"
          % ("nyxus feature", "pyradiomics", "header pin", "fresh run", "rel", "verdict"))
    print("-" * 110)

    failures, missing = [], []
    for name in sorted(pins):
        pr_name = NAME_MAP.get(name)
        if pr_name is None or pr_name not in result:
            missing.append(name)
            print("%-20s %-18s %22.12g %22s %11s  NOT PRODUCED BY THE ORACLE"
                  % (name, pr_name or "-", pins[name], "-", "-"))
            continue
        got = result[pr_name]
        r = relerr(pins[name], got)
        ok = r <= 1e-12
        if not ok:
            failures.append((name, pins[name], got, r))
        print("%-20s %-18s %22.12g %22.12g %11.3e  %s"
              % (name, pr_name, pins[name], got, r, "matches oracle" if ok else "ROTTED PIN"))

    reverse = [k for k in NAME_MAP if k not in pins]
    print()
    if reverse:
        print("REVERSE CHECK: the oracle produces %d mapped feature(s) the header pins nothing for: %s"
              % (len(reverse), ", ".join(sorted(reverse))))

    print("\nRANGE AND IDENTITY CHECKS")
    checks = []
    if {"3MAJOR_AXIS_LEN", "3MINOR_AXIS_LEN", "3LEAST_AXIS_LEN"} <= set(pins):
        checks.append(("3MAJOR_AXIS_LEN >= 3MINOR_AXIS_LEN >= 3LEAST_AXIS_LEN > 0",
                       pins["3MAJOR_AXIS_LEN"] >= pins["3MINOR_AXIS_LEN"] >= pins["3LEAST_AXIS_LEN"] > 0,
                       pins["3LEAST_AXIS_LEN"]))
    if {"3ELONGATION", "3MAJOR_AXIS_LEN", "3MINOR_AXIS_LEN"} <= set(pins):
        checks.append(("3ELONGATION == 3MINOR_AXIS_LEN / 3MAJOR_AXIS_LEN",
                       relerr(pins["3ELONGATION"], pins["3MINOR_AXIS_LEN"] / pins["3MAJOR_AXIS_LEN"]) < 1e-12,
                       pins["3ELONGATION"]))
    if {"3FLATNESS", "3MAJOR_AXIS_LEN", "3LEAST_AXIS_LEN"} <= set(pins):
        checks.append(("3FLATNESS == 3LEAST_AXIS_LEN / 3MAJOR_AXIS_LEN",
                       relerr(pins["3FLATNESS"], pins["3LEAST_AXIS_LEN"] / pins["3MAJOR_AXIS_LEN"]) < 1e-12,
                       pins["3FLATNESS"]))
    if "3MESH_VOLUME" in pins:
        # the mesh of a grid of sx*sy*sz voxels is the linear image of the unit-spacing one
        product = SPACING[0] * SPACING[1] * SPACING[2]
        checks.append(("3MESH_VOLUME == unit-spacing MeshVolume * sx*sy*sz",
                       relerr(pins["3MESH_VOLUME"], unit["MeshVolume"] * product) < 1e-12,
                       pins["3MESH_VOLUME"]))
    checks.append(("VoxelVolume == N * sx*sy*sz (N = 274432, the count the axis factor uses)",
                   relerr(result["VoxelVolume"], 274432 * SPACING[0] * SPACING[1] * SPACING[2]) < 1e-12,
                   result["VoxelVolume"]))

    bad = 0
    for label, ok, value in checks:
        print("  [%s] %-72s  value=%.10g" % ("PASS" if ok else "FAIL", label, value))
        if not ok:
            bad += 1

    if args.emit:
        print()
        for name in sorted(NAME_MAP):
            pr_name = NAME_MAP[name]
            if pr_name in result:
                print('    {"%s", %.17g}, // original_shape_%s at spacing (1.3, 0.8, 2.5)'
                      % (name, result[pr_name], pr_name))

    print("\nSUMMARY: %d pins, %d mismatched, %d not produced, %d unpinned, %d failed checks"
          % (len(pins), len(failures), len(missing), len(reverse), bad))
    if failures or missing or reverse or bad:
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
