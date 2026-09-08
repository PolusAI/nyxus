"""OFFLINE scikit-image oracle for the 2D ROI_RADIUS_* features.

    python tests/vetting/oracles/gen_morphology_radius_skimage.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in the disk table of
tests/test_2d_morphology_skimage.h, exiting non-zero on any mismatch, so a pin added later cannot
silently stop being covered.

WHAT THE FEATURES MEASURE. RoiRadiusFeature (src/nyx/features/roi_radius.cpp) takes every ROI pixel,
measures its distance to the nearest pixel of the ROI's own contour, and reports the mean, maximum
and median of those distances. The reference here computes the same quantity out of two library
calls, with no part of Nyxus re-implemented:

  boundary = skimage.segmentation.find_boundaries(mask, connectivity=1, mode='inner')
  radii    = min distance from each mask pixel to a boundary pixel

find_boundaries(connectivity=1, mode='inner') is the whole of the convention: the foreground pixels
4-adjacent to background. It is CellProfiler's own edge definition, which is why
gen_morphology_cellprofiler.py and gen_morphology_wholeslide_cellprofiler.py already reproduce the
EDGE_* statistics through it. On the 8x8 shape2d mask it does not merely agree in COUNT with the
Nyxus chain-code contour: measured pixel by pixel, the Nyxus contour IS this boundary shifted by
(+1,+1) -- all 18 coordinates, exactly. On the r=20 disk it returns 112, the edge-pixel count
benchmarks.md records for `bench_disk64_diagonal_boundary`, the same 1257-pixel shape. The distance step carries no
convention at all, and is computed here twice -- an exhaustive numpy minimum and
scipy.ndimage.distance_transform_edt over the complement of the boundary -- with the two required to
agree, so neither implementation is taken on trust.

WHY DISKS AND NOT THE 8x8 shape2d RASTER. Same reason PERIMETER is vetted on the circles benchmark:
on a 26-pixel object with a hole the two boundary conventions have nothing to converge to. A disk
also has an EXACT closed form for the maximum -- sqrt((R-1)^2+1), the centre's distance to the
boundary pixel at offset (1, R-1) -- which test_2d_morphology_analytic.h asserts as a second oracle.
MAX/(R-1) = sqrt(1+1/(R-1)^2) then converges to 1 as 1/(2(R-1)^2): 0.62%, 0.14%, 0.03% at
R = 10, 20, 40, i.e. LINEAR growth in R.

WHAT IS VETTED AND WHAT IS NOT. MAX and MEDIAN agree with this reference to double precision.
ROI_RADIUS_MEAN does not, and the residual is not this family's: ContourFeature::buildRegularContour
returns every contour pixel one pixel right and one pixel down of where it is (the padding it traces
in is never subtracted), so Nyxus measures its pixels against a shifted boundary. Shifting the
reference boundary by (+1, +1) reproduces all three Nyxus values exactly, on the disks and on the
shape2d raster alike -- this script prints that comparison as the standing evidence. MAX and MEDIAN
survive the shift on a disk because the maximizing pixel simply moves with it; MEAN does not, and
stays a regression row until the contour offset is fixed.

Provenance: tool=scikit-image 0.26.0, scipy 1.17.1, numpy 2.4.6; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_morphology_radius_skimage.py. Run offline; CI never invokes it.
"""
import os
import re

import numpy as np
import scipy
import skimage
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
TEST_H = os.path.join(TESTS, "test_2d_morphology_skimage.h")

RADII = (10, 20, 40)
RELTOL = 1e-3          # SPEC 7 same-definition-oracle tier, matching the C++ assertions


def disk(R, pad=3):
    """The fixture calculate_disk_radius_values(R) builds: (x-c)^2 + (y-c)^2 <= R^2, c = R + pad."""
    side = int(2 * R) + 2 * pad
    c = R + pad
    y, x = np.mgrid[0:side + 1, 0:side + 1]
    return ((x - c) ** 2 + (y - c) ** 2) <= R * R


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


def radius_stats(mask, shift=0):
    """Mean/max/median distance from each mask pixel to the inner boundary, optionally shifted."""
    b = find_boundaries(mask, connectivity=1, mode="inner")
    by, bx = np.nonzero(b)
    by, bx = by + shift, bx + shift
    py, px = np.nonzero(mask)
    d = np.sqrt(np.min((py[:, None] - by) ** 2 + (px[:, None] - bx) ** 2, axis=1).astype(float))

    if shift == 0:
        # Second, independent implementation of the same distance: an EDT over everything that is
        # not boundary. Only the boundary definition above may carry a convention; this catches a
        # mistake in the exhaustive minimum rather than trusting it.
        edt = distance_transform_edt(~b)[mask]
        assert np.allclose(d, edt, rtol=0, atol=1e-12), "numpy minimum and scipy EDT disagree"

    return {"MEAN": float(d.mean()), "MAX": float(d.max()), "MEDIAN": float(np.median(d))}, int(b.sum())


def main():
    print(f"# scikit-image {skimage.__version__}, scipy {scipy.__version__}, numpy {np.__version__}")

    got = {}
    print("\n# paste-ready goldens (17 significant digits)")
    for R in RADII:
        m = disk(R)
        ref, nb = radius_stats(m)
        got[f"ROI_RADIUS_MAX_R{R}"] = ref["MAX"]
        got[f"ROI_RADIUS_MEDIAN_R{R}"] = ref["MEDIAN"]
        print(f'\t{{"ROI_RADIUS_MAX_R{R}", {ref["MAX"]!r}}},')
        print(f'\t{{"ROI_RADIUS_MEDIAN_R{R}", {ref["MEDIAN"]!r}}},')
        print(f"\t# R={R}: {int(m.sum())} pixels, {nb} boundary pixels, "
              f"MAX/(R-1) = {ref['MAX'] / (R - 1):.6f}")

    print("\n# the contour offset, measured: reference vs reference shifted by (+1,+1)")
    data = open(DATA_H, encoding="utf-8", errors="replace").read()
    shapes = [(f"disk_r{R}", disk(R)) for R in RADII]
    shapes.append(("shape2d", parse_fixture(data, "shape2d_morphology_mask") > 0))
    for name, m in shapes:
        a, _ = radius_stats(m, shift=0)
        b, _ = radius_stats(m, shift=1)
        print(f"  {name:9s} MEAN {a['MEAN']!r} -> {b['MEAN']!r}   "
              f"MAX {a['MAX']!r} -> {b['MAX']!r}   "
              f"MEDIAN {a['MEDIAN']!r} -> {b['MEDIAN']!r}")
    print("  (the shifted column is what Nyxus reports today; MAX and MEDIAN are unmoved by the")
    print("   shift on a disk, MEAN is not, which is why only the first two are vetted)")

    test = open(TEST_H, encoding="utf-8", errors="replace").read()
    pins = parse_pins(test, "morphology_2d_skimage_radius_disks_ref_vals")

    print(f"\n# verifying {len(pins)} pinned goldens against this run")
    if not pins:
        # An empty table is not "nothing to check" -- it is the table having been renamed or emptied
        # out from under this generator, which would otherwise report ALL CHECKS PASSED.
        print("NO PINS FOUND in morphology_2d_skimage_radius_disks_ref_vals -- nothing is covered")
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
        verdict = "OK  " if rel <= RELTOL else "FAIL"
        print(f"  {verdict} {name}: skimage={have!r} pinned={want!r} rel={rel:.3g}")
        nok, nfail = (nok + 1, nfail) if rel <= RELTOL else (nok, nfail + 1)

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible")
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
