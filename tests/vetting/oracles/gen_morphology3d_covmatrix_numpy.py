#!/usr/bin/env python3
"""Regenerate and re-verify the covariance / eigenvalue goldens in
tests/test_3d_morphology_mechanics.h.

These are kernel mechanics, not feature values: the covariance matrix of a fixed ten-point cloud and
its eigenvalues, which is the arithmetic the 3D PCA shape features are built on. They were MATLAB
`cov`/`eig` output quoted to five significant figures from a session that cannot be re-run from this
tree; numpy computes the same two quantities and is runnable here.

`Nyxus::calc_covariance` (src/nyx/helpers/helpers.cpp) normalises by n-1, i.e. the sample covariance,
which is what MATLAB `cov` and numpy `ddof=1` both compute -- so the replacement is the same
quantity, not merely a similar one.

Usage:
    python gen_morphology3d_covmatrix_numpy.py            # print the goldens
    python gen_morphology3d_covmatrix_numpy.py --check    # re-verify what the header pins

--check parses every value out of the header and compares it against a fresh numpy run, exiting
non-zero on any mismatch or on a pin it cannot produce.
"""

import argparse
import os
import re
import sys

import numpy as np

HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "test_3d_morphology_mechanics.h")

# The cloud under test, X/Y/Z. Must match morphology_3d_covmatrix_cloud in the header; the intensity
# column is dropped because calc_cov_matrix is a moment of the coordinates and never reads it.
CLOUD = np.array([
    [9, 96, 4], [26, 55, 89], [80, 52, 91], [3, 23, 80], [93, 49, 10],
    [73, 62, 26], [49, 68, 34], [58, 40, 68], [24, 37, 14], [46, 99, 72],
], dtype=float)

REL = 1e-12   # the two sides are the same arithmetic; this only absorbs decimal round-tripping


def compute():
    K = np.cov(CLOUD.T, ddof=1)                       # sample covariance, the n-1 normalisation
    L = np.sort(np.linalg.eigvalsh(K))[::-1]          # descending, the order calc_eigvals returns
    return K, L


def parse_cloud(text):
    """The cloud as the header spells it, so a drifted fixture is caught rather than assumed."""
    body = re.search(r"morphology_3d_covmatrix_cloud\s*=\s*\{(.*?)\n\};", text, re.S)
    if not body:
        return None
    rows = re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\d+\s*\}", body.group(1))
    return np.array([[float(v) for v in r] for r in rows]) if rows else None


def parse_list(text, name):
    body = re.search(name + r"\s*\{(.*?)\n\};", text, re.S)
    if not body:
        return None
    return [float(v) for v in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", body.group(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    K, L = compute()

    if not args.check:
        print("numpy", np.__version__)
        print("covariance (row-major, ddof=1):")
        for row in K:
            print("   " + ", ".join("%.17g" % v for v in row))
        print("eigenvalues (descending):")
        for v in L:
            print("   %.17g" % v)
        return 0

    text = open(HEADER, encoding="utf-8").read()
    nfail = 0

    pinned_cloud = parse_cloud(text)
    if pinned_cloud is None:
        print("FAIL: could not read morphology_3d_covmatrix_cloud from the header")
        return 1
    if pinned_cloud.shape != CLOUD.shape or not np.array_equal(pinned_cloud, CLOUD):
        print("FAIL: the header's point cloud differs from this generator's")
        return 1

    for name, want in (("morphology_3d_mechanics_covmatrix_ref_vals", K.reshape(-1)),
                       ("morphology_3d_mechanics_eigenvalues_ref_vals", L)):
        pins = parse_list(text, name)
        if pins is None:
            print("FAIL: %s not found in the header" % name)
            nfail += 1
            continue
        if len(pins) != len(want):
            print("FAIL: %s pins %d values, expected %d" % (name, len(pins), len(want)))
            nfail += 1
            continue
        for i, (pin, ref) in enumerate(zip(pins, want)):
            rel = abs(pin - ref) / abs(ref) if ref else abs(pin - ref)
            if rel > REL:
                print("FAIL: %s[%d] pinned %.17g, numpy %.17g (rel %.3g)" % (name, i, pin, ref, rel))
                nfail += 1

    total = K.size + L.size
    if nfail:
        print("SOME CHECKS FAILED -- do not promote (%d of %d)" % (nfail, total))
        return 1
    print("checked %d pins against numpy %s: clean" % (total, np.__version__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
