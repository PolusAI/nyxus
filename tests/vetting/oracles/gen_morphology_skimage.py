"""OFFLINE scikit-image oracle for the 2D morphology features ORIENTATION and
EROSIONS_2_VANISH (SPEC 4, oracle=skimage), on the 8x8 shape2d fixture
(shape2d_morphology_mask, test_data.h). Validates the goldens pinned in
test_morphology_skimage.h.

ORIENTATION -- skimage.measure.regionprops(...).orientation is the angle of the
  major axis measured from the ROW (axis-0) direction, CCW, radians. Nyxus measures
  the same ellipse's major axis from the X (column) axis in degrees, so
    NYXUS_ORIENTATION == 90 - degrees(skimage.orientation).
  The ellipse ORIENTATION is invariant to the pixel finite-size (+1/12) second-moment
  correction (it shifts mu20 and mu02 equally, leaving mu20-mu02 and mu11 unchanged),
  so the angle matches to numerical precision even though the AXIS LENGTHS differ
  ~1.4% (that gap is why MAJOR/MINOR_AXIS_LENGTH and ECCENTRICITY are NOT vetted here).

EROSIONS_2_VANISH -- number of binary erosions until the object disappears. Nyxus
  uses a 3x3 (8-connected) structuring element == skimage.morphology.square(3). The
  4-connected disk(1) gives a different count (2 vs 1), so the test also discriminates
  the connectivity convention. (EROSIONS_2_VANISH_COMPLEMENT is a degenerate 0 on this
  fixture -- the complement is the bbox background ring -- and is not vetted.)

Provenance: tool=scikit-image 0.26; numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_morphology_skimage.py. Run offline; CI never invokes it.
"""
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, footprint_rectangle

# shape2d_morphology_mask (test_data.h): rows y=0..7, cols x=0..7; mask[y, x]
ROWS = [
    "00110000", "01111000", "11111100", "11101100",
    "01111000", "00111000", "00011000", "00000000",
]
MASK = np.array([[int(c) for c in r] for r in ROWS], dtype=np.uint8)

# Nyxus goldens (test_morphology_common.h)
NYX = {"ORIENTATION": 70.4173944984207, "EROSIONS_2_VANISH": 1.0}
TOL_ANG = 1e-3   # degrees
TOL_CNT = 0


def main():
    all_ok = True
    rp = regionprops(label(MASK, connectivity=2))[0]

    # ORIENTATION: skimage angle from row axis -> Nyxus angle from x axis
    ori = 90.0 - np.degrees(rp.orientation)
    ok = abs(ori - NYX["ORIENTATION"]) <= TOL_ANG
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'} ORIENTATION: skimage(90-deg)={ori:.10f} nyxus={NYX['ORIENTATION']}")

    # EROSIONS_2_VANISH: 8-connected (square(3)) erosions to vanish
    m = MASK.astype(bool); n = 0
    while m.any():
        m = binary_erosion(m, footprint_rectangle((3, 3)))
        n += 1
    ok = abs(n - NYX["EROSIONS_2_VANISH"]) <= TOL_CNT
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'} EROSIONS_2_VANISH: skimage(square3)={n} nyxus={NYX['EROSIONS_2_VANISH']:.0f}")

    # informational: the moment-normalization gap that keeps these OUT of the vetted set
    print(f"  (info) axis_major sk={rp.axis_major_length:.4f} nyx=6.9688 ; "
          f"eccentricity sk={rp.eccentricity:.4f} nyx=0.6162 -> ~1.4% gap, not vetted")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
