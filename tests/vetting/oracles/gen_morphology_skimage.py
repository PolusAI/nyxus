"""OFFLINE scikit-image oracle for the 2D morphology features vetted against skimage.

    python tests/vetting/oracles/gen_morphology_skimage.py        (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in
tests/test_2d_morphology_skimage.h -- both the shape2d table and the circles table. Any pin this
generator cannot produce is reported by name and the script exits non-zero on any mismatch, so the
check cannot silently stop covering a golden added later.

The fixtures are read out of tests/test_data.h rather than copied here, so the generator and the C++
tests share one copy of the pixels (same discipline as ibsi_phantom.py).

Conventions that make each Nyxus feature comparable to skimage:

CONVEX_HULL_AREA / SOLIDITY -- Nyxus computes a Pick's-theorem pixel-count hull area through pixel
  CENTRES, which is convex_hull_image(offset_coordinates=False).sum() == 27. skimage's regionprops
  DEFAULT (offset_coordinates=True) first expands every pixel to its +/-0.5 corners and rasterises
  to 28; that +1 px is a corner-expansion convention, not an error.

ORIENTATION -- regionprops(...).orientation is the major-axis angle from the ROW (axis-0)
  direction, CCW, radians. Nyxus measures the same ellipse from the X (column) axis in degrees, so
    NYXUS_ORIENTATION == 90 - degrees(skimage.orientation).
  The angle is invariant to the pixel finite-size (+1/12) second-moment correction (it shifts mu20
  and mu02 equally, leaving mu20-mu02 and mu11 unchanged), so it matches to numerical precision even
  though the AXIS LENGTHS differ ~1.4%. Those are vetted against MATLAB instead, which applies the
  same +1/12 correction Nyxus does -- see gen_morphology_matlab.m.

EROSIONS_2_VANISH -- number of binary erosions until the object disappears. Nyxus uses a 3x3
  (8-connected) structuring element == skimage.morphology.footprint_rectangle((3, 3)). The
  4-connected disk(1) gives a different count (2 vs 1), so the value also pins the connectivity
  convention. (EROSIONS_2_VANISH_COMPLEMENT is a degenerate 0 on this fixture -- the complement is
  the bbox background ring -- and is not vetted.)

DIAMETER_EQUAL_AREA -- regionprops(...).equivalent_diameter_area = sqrt(4*Area/pi), the same closed
  form Nyxus uses, over an exact pixel count.

PERIMETER -- skimage.measure.perimeter (the 4-neighbourhood boundary walk regionprops.perimeter
  uses) on the circles benchmark, where it and the Nyxus chain-code contour walk agree to ~4e-15.
  They do NOT agree on the small shape2d mask (12.657 vs 26.935), so PERIMETER is vetted on the
  circles benchmark only.

Provenance: tool=scikit-image 0.26.0; numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_morphology_skimage.py. Run offline; CI never invokes it.
"""
import os
import re
import sys

import numpy as np
import skimage
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import convex_hull_image, erosion, footprint_rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
TEST_H = os.path.join(TESTS, "test_2d_morphology_skimage.h")

RELTOL = 1e-3          # SPEC 7 same-definition-oracle tier, matching the C++ assertions


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


def main():
    data = open(DATA_H, encoding="utf-8", errors="replace").read()
    M = (parse_fixture(data, "shape2d_morphology_mask") > 0).astype(np.uint8)
    C = (parse_fixture(data, "roiDataForPerimeterTest") > 0).astype(np.uint8)

    rp = regionprops(label(M, connectivity=2))[0]

    hull = convex_hull_image(M.astype(bool), offset_coordinates=False)
    hull_area = float(hull.sum())

    # footprint_rectangle((3, 3)) is odd-sized, so erosion() applies no half-pixel shift
    m, erosions = M.astype(bool), 0
    while m.any():
        m = erosion(m, footprint_rectangle((3, 3)))
        erosions += 1

    got = {
        "CONVEX_HULL_AREA": hull_area,
        "SOLIDITY": float(M.sum()) / hull_area,
        "EROSIONS_2_VANISH": float(erosions),
        "ORIENTATION": float(90.0 - np.degrees(rp.orientation)),
        "DIAMETER_EQUAL_AREA": float(rp.equivalent_diameter_area),
        "PERIMETER": float(perimeter(C)),
    }

    print(f"# scikit-image {skimage.__version__}, numpy {np.__version__}")
    print("# paste-ready goldens (17 significant digits)")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},')

    # informational: the moment-normalization gap that keeps the axis lengths out of the skimage set
    print(f"  (info) axis_major sk={rp.axis_major_length:.6f} nyx=6.968816 ; "
          f"eccentricity sk={rp.eccentricity:.6f} nyx=0.616174 -> ~1.4% gap, vetted vs MATLAB")

    test = open(TEST_H, encoding="utf-8", errors="replace").read()
    pins = {}
    for table in ("morphology_2d_skimage_shape2d_ref_vals",
                  "morphology_2d_skimage_circles_ref_vals"):
        pins.update(parse_pins(test, table))

    print(f"\n# verifying {len(pins)} pinned goldens against this run")
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
