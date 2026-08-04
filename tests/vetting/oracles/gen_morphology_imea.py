"""OFFLINE imea oracle for the 2D morphology feature DIAMETER_EQUAL_PERIMETER
(SPEC 4, oracle=imea), on the 8x8 shape2d fixture (shape2d_morphology_mask,
test_data.h). Validates the golden pinned in test_morphology_imea.h.

DIAMETER_EQUAL_PERIMETER -- the DIN ISO 9276-6 perimeter-equal diameter: the diameter
  of the circle whose circumference equals the object's perimeter, i.e. perimeter/pi.
  Nyxus computes it in contour.cpp; imea implements the same documented transform in
  imea.measure_2d.macro.perimeter_equal_diameter(perimeter). Feeding imea the Nyxus
  PERIMETER reproduces the Nyxus DIAMETER_EQUAL_PERIMETER to double precision.

SCOPE -- this vets the TRANSFORM, not the perimeter it consumes. The two packages use
  different perimeter conventions: Nyxus walks its chain-code contour (26.9349412836191
  on this fixture) while imea takes cv2.arcLength over the OpenCV contour (12.6568542495),
  so imea's END-TO-END diameter_equal_perimeter is 4.0288018356 -- it does NOT agree with
  Nyxus, and the disagreement is entirely inherited from PERIMETER. PERIMETER therefore
  stays a regression row of its own; what the assertion pins is that Nyxus' derived
  diameter is the ISO quantity a third-party package computes from the same input.
  (Same reason the other imea morphology rows are vetted on clean analytic fixtures.)

Provenance: tool=imea 0.3.5; numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_morphology_imea.py. Run offline; CI never invokes it.
"""
import numpy as np
from imea.measure_2d import macro

# shape2d_morphology_mask (test_data.h): rows y=0..7, cols x=0..7; mask[y, x]
ROWS = [
    "00110000", "01111000", "11111100", "11101100",
    "01111000", "00111000", "00011000", "00000000",
]
MASK = np.array([[int(c) for c in r] for r in ROWS], dtype=bool)

# Nyxus goldens (test_morphology_common.h)
NYX_PERIMETER = 26.9349412836191
NYX_DEP = 8.57365809435587
TOL_ABS = 1e-9   # SPEC 7 "exact" tier: both sides evaluate the same closed form in double


def main():
    dep = macro.perimeter_equal_diameter(NYX_PERIMETER)
    ok = abs(dep - NYX_DEP) <= TOL_ABS
    print(f"  {'OK ' if ok else 'FAIL'} DIAMETER_EQUAL_PERIMETER: "
          f"imea(perimeter_equal_diameter, P={NYX_PERIMETER})={dep:.15g} "
          f"nyxus={NYX_DEP:.15g} |diff|={abs(dep - NYX_DEP):.3g}")

    # Scope check: imea's own end-to-end value on the same mask, to keep the perimeter-convention
    # gap on the record rather than hidden behind the agreeing transform.
    import imea
    e2e = imea.shape_measurements_2d(MASK, spatial_resolution_xy=1.0, dalpha=9).iloc[0]
    print(f"  (info) imea end-to-end: perimeter={e2e['perimeter']:.10f} -> "
          f"diameter_equal_perimeter={e2e['diameter_equal_perimeter']:.10f}; "
          f"nyxus perimeter={NYX_PERIMETER} (chain-code contour vs cv2.arcLength) -> "
          f"the transform agrees, the perimeter convention does not")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
