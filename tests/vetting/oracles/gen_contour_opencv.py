"""OFFLINE OpenCV oracle for the 2D ROI contour and PERIMETER.

    python tests/vetting/oracles/gen_contour_opencv.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in contour_2d_opencv_ref_vals
(tests/test_2d_contour_opencv.h), exiting non-zero on any mismatch or on a pin it cannot produce.

WHAT IS COMPARED. ContourFeature::buildRegularContour traces every border of the ROI -- the outer
border of each 8-connected fragment and the border of each hole -- as a closed walk in 8-connected
steps, and PERIMETER is the sum of the walks' Euclidean lengths. That is the border following of
Suzuki and Abe (1985), which OpenCV implements as

    cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

RETR_LIST returns every border, holes included, and CHAIN_APPROX_NONE keeps every pixel of each
walk. The reference PERIMETER is the sum over those walks of the closed length, accumulated here in
double from OpenCV's integer points: cv2.arcLength sums in float32 and is off by ~1e-8 relative.
The mask is padded by one background pixel so a border on the frame edge is traced like any other.

Fixtures are read from the headers that define them, never copied: bench_blob97_filled from
tests/test_2d_contour_common.h and shape2d_morphology_mask from tests/test_data.h.

Provenance: tool=OpenCV (cv2) 4.13.0, numpy 2.4.6; env=nyxus_mirp (conda). OpenCV 5.0.0 reproduces
every pin too;
generator=tests/vetting/oracles/gen_contour_opencv.py. Run offline; CI never invokes it.
"""
import math
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
COMMON_H = os.path.join(TESTS, "test_2d_contour_common.h")
TEST_H = os.path.join(TESTS, "test_2d_contour_opencv.h")

RELTOL = 1e-9          # same algorithm on both sides: integer walks, only the summation order differs


def parse_rows(txt, name):
    """The text-row mask `name` from test_2d_contour_common.h, '1' = ROI."""
    body = txt.split(name + " = {", 1)[1].split("};", 1)[0]
    rows = re.findall(r'"([^"]*)"', body)
    return np.array([[c == "1" for c in r] for r in rows], dtype=bool)


def parse_fixture(txt, name):
    """The {x, y, value} pixel array `name` from test_data.h, as a 2-D numpy array."""
    body = txt.split(name + "[] = {", 1)[1].split("};", 1)[0]
    tok = [(int(x), int(y), int(v)) for x, y, v in
           re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
    w = max(t[0] for t in tok) + 1
    h = max(t[1] for t in tok) + 1
    a = np.zeros((h, w), dtype=np.int64)
    for x, y, v in tok:
        a[y, x] = v
    return a


def parse_pins(txt, table):
    """Every {"NAME", value} entry of one named ref_vals_map in a test header."""
    body = txt.split(table + "{", 1)[1].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"([A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def opencv_borders(mask):
    """Number of borders, distinct border pixels and summed closed-walk length of a boolean mask."""
    padded = np.pad(mask.astype(np.uint8), 1)
    contours, _ = cv2.findContours(padded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pixels = set()
    length = 0.0
    for c in contours:
        pts = [(int(p[0][0]), int(p[0][1])) for p in c]
        pixels.update(pts)
        if len(pts) > 1:
            for i in range(len(pts)):
                (ax, ay), (bx, by) = pts[i - 1], pts[i]
                length += math.hypot(ax - bx, ay - by)
    return len(contours), len(pixels), length


def main():
    print(f"# opencv {cv2.__version__}, numpy {np.__version__}")
    shapes = {
        "BLOB97": parse_rows(open(COMMON_H, encoding="utf-8").read(), "bench_blob97_filled_rows"),
        "SHAPE2D": parse_fixture(open(DATA_H, encoding="utf-8", errors="replace").read(),
                                 "shape2d_morphology_mask") > 0,
    }
    got = {}
    print("\n# paste-ready goldens (17 significant digits)")
    for tag, m in shapes.items():
        n, npx, length = opencv_borders(m)
        got[f"NUM_CONTOURS_{tag}"] = float(n)
        got[f"CONTOUR_PIXELS_{tag}"] = float(npx)
        got[f"PERIMETER_{tag}"] = length
        print(f"\t// {tag}: {int(m.sum())} ROI pixels")
        print(f'\t{{"NUM_CONTOURS_{tag}", {n}}},')
        print(f'\t{{"CONTOUR_PIXELS_{tag}", {npx}}},')
        print(f'\t{{"PERIMETER_{tag}", {length!r}}},')

    pins = parse_pins(open(TEST_H, encoding="utf-8").read(), "contour_2d_opencv_ref_vals")
    print(f"\n# verifying {len(pins)} pinned goldens against this run")
    if not pins:
        print("NO PINS FOUND in contour_2d_opencv_ref_vals -- nothing is covered")
        return 1
    nok = nfail = nmiss = 0
    for name in sorted(pins):
        want = pins[name]
        if name not in got:
            print(f"  MISSING {name}: pinned {want!r} but this generator produces no such value")
            nmiss += 1
            continue
        have = got[name]
        rel = abs(have - want) / max(abs(want), 1e-12)
        ok = rel <= RELTOL
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: opencv={have!r} pinned={want!r} rel={rel:.3g}")
        nok, nfail = (nok + 1, nfail) if ok else (nok, nfail + 1)

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible")
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
