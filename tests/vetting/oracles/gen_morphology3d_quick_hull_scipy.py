#!/usr/bin/env python3
"""Regenerate and re-verify the quick_hull volume pins in tests/test_3d_morphology_mechanics.h.

These are kernel mechanics, not feature values: the volume of the convex hull of a lattice
ellipsoid's voxel centres, which is the quantity quick_hull builds 3VOLUME_CONVEXHULL from. The
reference is scipy.spatial.ConvexHull, i.e. qhull, an implementation independent of quick_hull.

A hull of lattice points has a volume that is a multiple of 1/6, so each pin is written as that
multiple over six. --check also confirms that qhull's volume sits on a multiple of 1/6, which is
what makes the pin exact rather than qhull's floating-point estimate of it.

Usage:
    python gen_morphology3d_quick_hull_scipy.py            # print the pins
    python gen_morphology3d_quick_hull_scipy.py --check    # re-verify what the header pins

--check parses the ellipsoids and the pinned volumes out of the header and compares them against a
fresh qhull run, exiting non-zero on any mismatch or on a pin it cannot produce.
"""

import argparse
import os
import re
import sys

import numpy as np
import scipy
from scipy.spatial import ConvexHull

HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "test_3d_morphology_mechanics.h")

# Semi-axes of each ellipsoid. Must match morphology_3d_quick_hull_ellipsoids in the header.
ELLIPSOIDS = [(5, 9, 6), (4, 9, 4), (6, 9, 4)]

REL = 1e-12   # qhull's volume against the exact multiple of 1/6


def cloud(a, b, c):
    """Every voxel with (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1, in integers as the header computes it."""
    abc2 = a * a * b * b * c * c
    return np.array([(x, y, z)
                     for x in range(-a, a + 1)
                     for y in range(-b, b + 1)
                     for z in range(-c, c + 1)
                     if x * x * b * b * c * c + y * y * a * a * c * c + z * z * a * a * b * b <= abc2],
                    dtype=float)


def compute():
    """-> [(a, b, c, voxels, qhull volume, nearest multiple of 1/6 as sixths)]"""
    out = []
    for a, b, c in ELLIPSOIDS:
        P = cloud(a, b, c)
        v = ConvexHull(P).volume
        out.append((a, b, c, len(P), v, int(round(v * 6))))
    return out


def parse_ellipsoids(text):
    body = re.search(r"morphology_3d_quick_hull_ellipsoids\[\]\[3\]\s*=\s*\{(.*?)\};", text, re.S)
    if not body:
        return None
    return [tuple(int(v) for v in r)
            for r in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body.group(1))]


def parse_sixths(text):
    body = re.search(r"morphology_3d_mechanics_quick_hull_volume_ref_vals\s*\{(.*?)\n\};", text, re.S)
    if not body:
        return None
    return [float(v) for v in re.findall(r"(\d+(?:\.\d+)?)\s*/\s*6(?:\.0)?", body.group(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rows = compute()

    if not args.check:
        print("scipy", scipy.__version__, "numpy", np.__version__)
        for a, b, c, n, v, s in rows:
            print("   ellipsoid %d,%d,%d  %4d voxels  qhull %.17g  = %d / 6" % (a, b, c, n, v, s))
        return 0

    text = open(HEADER, encoding="utf-8").read()

    pinned = parse_ellipsoids(text)
    if pinned is None:
        print("FAIL: could not read morphology_3d_quick_hull_ellipsoids from the header")
        return 1
    if pinned != ELLIPSOIDS:
        print("FAIL: the header's ellipsoids %s differ from this generator's %s" % (pinned, ELLIPSOIDS))
        return 1

    sixths = parse_sixths(text)
    if sixths is None or len(sixths) != len(rows):
        print("FAIL: expected %d volume pins written as N / 6, found %s" % (len(rows), sixths))
        return 1

    nfail = 0
    for (a, b, c, n, v, s), pin in zip(rows, sixths):
        exact = s / 6.0
        if abs(v - exact) > REL * exact:
            print("FAIL: ellipsoid %d,%d,%d: qhull %.17g is not a multiple of 1/6" % (a, b, c, v))
            nfail += 1
        if pin != s:
            print("FAIL: ellipsoid %d,%d,%d pinned %g / 6, qhull %d / 6" % (a, b, c, pin, s))
            nfail += 1

    if nfail:
        print("SOME CHECKS FAILED (%d of %d)" % (nfail, 2 * len(rows)))
        return 1
    print("checked %d pins against qhull (scipy %s): clean" % (len(rows), scipy.__version__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
