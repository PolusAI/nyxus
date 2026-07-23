"""OFFLINE imea oracle for the 2D morphology features DIAMETER_EQUAL_PERIMETER,
GEODETIC_LENGTH and THICKNESS (SPEC 4, oracle=imea), on the 8x8 shape2d fixture
(shape2d_morphology_mask, test_data.h). Validates the goldens pinned in
test_morphology_imea.h.

DIAMETER_EQUAL_PERIMETER -- the DIN ISO 9276-6 perimeter-equal diameter: the diameter
  of the circle whose circumference equals the object's perimeter, i.e. perimeter/pi.
  Nyxus computes it in contour.cpp; imea implements the same documented transform in
  imea.measure_2d.macro.perimeter_equal_diameter(perimeter). Feeding imea the Nyxus
  PERIMETER reproduces the Nyxus DIAMETER_EQUAL_PERIMETER to double precision.

GEODETIC_LENGTH / THICKNESS -- the DIN ISO 9276-6 rectangle model: the rectangle with the
  same area and perimeter as the object, whose side lengths are the roots of
  x^2 - (P/2)x + A = 0, i.e. P/4 +- sqrt(P^2/16 - A). Nyxus computes them in
  geo_len_thickness.cpp; imea implements the same documented model in
  imea.measure_2d.macro.geodeticlength_and_thickness(area, perimeter). Feeding imea the
  Nyxus AREA_PIXELS_COUNT and PERIMETER reproduces both Nyxus values bit for bit.

SCOPE -- these vet the TRANSFORM, not the area and perimeter it consumes. The two packages
  use different perimeter conventions: Nyxus walks its chain-code contour (26.9349412836191
  on this fixture) while imea takes cv2.arcLength over the OpenCV contour (12.6568542495),
  so imea's END-TO-END values do NOT agree with Nyxus -- diameter_equal_perimeter is
  4.0288018356, and with the shorter perimeter (P/4)^2 - A goes negative, imea clamps the
  root, and geodeticlength and thickness both collapse to 3.1642135624. The disagreement is
  entirely inherited from PERIMETER, which therefore stays a regression row of its own; what
  the assertions pin is that Nyxus' derived quantities are the ISO quantities a third-party
  package computes from the same inputs.
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
NYX_AREA = 26.0
NYX_PERIMETER = 26.9349412836191
NYX_DEP = 8.57365809435587
NYX_GEODETIC_LENGTH = 11.13182483477333
NYX_THICKNESS = 2.3356458070362205
TOL_ABS = 1e-9   # SPEC 7 "exact" tier: both sides evaluate the same closed form in double


def main():
    ok = True

    dep = macro.perimeter_equal_diameter(NYX_PERIMETER)
    ok_dep = abs(dep - NYX_DEP) <= TOL_ABS
    ok = ok and ok_dep
    print(f"  {'OK ' if ok_dep else 'FAIL'} DIAMETER_EQUAL_PERIMETER: "
          f"imea(perimeter_equal_diameter, P={NYX_PERIMETER})={dep:.15g} "
          f"nyxus={NYX_DEP:.15g} |diff|={abs(dep - NYX_DEP):.3g}")

    geodetic_length, thickness = macro.geodeticlength_and_thickness(NYX_AREA, NYX_PERIMETER)
    for name, got, want in (("GEODETIC_LENGTH", geodetic_length, NYX_GEODETIC_LENGTH),
                            ("THICKNESS", thickness, NYX_THICKNESS)):
        ok_i = abs(got - want) <= TOL_ABS
        ok = ok and ok_i
        print(f"  {'OK ' if ok_i else 'FAIL'} {name}: "
              f"imea(geodeticlength_and_thickness, A={NYX_AREA}, P={NYX_PERIMETER})={got:.17g} "
              f"nyxus={want:.17g} |diff|={abs(got - want):.3g}")

    # Scope check: imea's own end-to-end values on the same mask, to keep the perimeter-convention
    # gap on the record rather than hidden behind the agreeing transforms.
    import imea
    e2e = imea.shape_measurements_2d(MASK, spatial_resolution_xy=1.0, dalpha=9).iloc[0]
    print(f"  (info) imea end-to-end: perimeter={e2e['perimeter']:.10f} -> "
          f"diameter_equal_perimeter={e2e['diameter_equal_perimeter']:.10f}, "
          f"geodeticlength={e2e['geodeticlength']:.10f}, thickness={e2e['thickness']:.10f}; "
          f"nyxus perimeter={NYX_PERIMETER} (chain-code contour vs cv2.arcLength) -> "
          f"the transforms agree, the perimeter convention does not")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
