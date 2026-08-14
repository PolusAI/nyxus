"""OFFLINE imea oracle for the 2D morphology features vetted against imea.

    python tests/vetting/oracles/gen_morphology_imea.py        (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_2d_morphology_imea.h --
both the shape2d ISO-transform table and the ellipse caliper table. Any pin this generator cannot
produce is reported by name and the script exits non-zero on any mismatch.

Two benchmarks, because the two claims have different scopes:

SHAPE2D (the 8x8 mask from test_data.h) -- the two DIN ISO 9276-6 transforms only.
  DIAMETER_EQUAL_PERIMETER is perimeter/pi; GEODETIC_LENGTH / THICKNESS are the rectangle model
  P/4 +- sqrt(P^2/16 - A). imea implements both in imea.measure_2d.macro, and fed the NYXUS area and
  perimeter it reproduces the Nyxus values to double precision.
  SCOPE: this vets the TRANSFORM, not the area and perimeter it consumes. imea derives its own
  perimeter from cv2.arcLength (12.657 here) rather than a chain-code contour walk (26.935), so
  imea's END-TO-END values on this mask do not agree with Nyxus; the whole gap is inherited from
  PERIMETER, which is vetted separately against scikit-image on the circles benchmark.
  The 19 caliper/chord statistics that used to be pinned in this file's imea table were never imea
  values at all -- imea's own numbers on this 8x8 raster differ by 12-79%, printed below as the
  `raster gap` block. They are now regression snapshots in test_2d_morphology_regression.h.

ELLIPSE (a=20, b=10, from calculate_ellipse_caliper_values) -- the caliper distributions.
  Run at dalpha=10 because that is the step Nyxus' own calipers sweep (rot_angle_increment = 10
  degrees, caliper.h), so both sample the same angles. Worst residual 4.99%.
  NOT vetted here: the three _MODE statistics. imea's own mode moves across 19..24 as dalpha goes
  5 -> 30, i.e. further than the Nyxus-imea gap, so no tolerance separates agreement from angular
  sampling noise. They are regression rows; the `mode instability` block below is the evidence.

Provenance: tool=imea 0.3.5; numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_morphology_imea.py. Run offline; CI never invokes it.
"""
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)     # imea 0.3.5 vs scikit-image 0.26

import imea
from imea.measure_2d import macro

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
TEST_H = os.path.join(TESTS, "test_2d_morphology_imea.h")

DALPHA = 10                # matches Nyxus' rot_angle_increment (caliper.h)
RELTOL_ISO = 1e-9          # SPEC 7 exact tier: both sides evaluate the same closed form
RELTOL_CALIPER = 0.06      # hull-vs-raster convention gap, matching the C++ assertion

# Nyxus inputs to the ISO transforms, from the shape2d fixture (test_2d_morphology_regression.h)
NYX_AREA = 26.0
NYX_PERIMETER = 26.9349412836191

CALIPER_KEYS = {
    "STAT_MARTIN_DIAM_MIN": "martin_min", "STAT_MARTIN_DIAM_MAX": "martin_max",
    "STAT_MARTIN_DIAM_MEAN": "martin_mean", "STAT_MARTIN_DIAM_MEDIAN": "martin_median",
    "STAT_MARTIN_DIAM_STDDEV": "martin_std",
    "STAT_NASSENSTEIN_DIAM_MIN": "nassenstein_min", "STAT_NASSENSTEIN_DIAM_MAX": "nassenstein_max",
    "STAT_NASSENSTEIN_DIAM_MEAN": "nassenstein_mean",
    "STAT_NASSENSTEIN_DIAM_MEDIAN": "nassenstein_median",
    "STAT_NASSENSTEIN_DIAM_STDDEV": "nassenstein_std",
    "STAT_FERET_DIAM_MIN": "feret_min", "STAT_FERET_DIAM_MAX": "feret_max",
    "STAT_FERET_DIAM_MEAN": "feret_mean", "STAT_FERET_DIAM_MEDIAN": "feret_median",
    "STAT_FERET_DIAM_STDDEV": "feret_std",
    "ALLCHORDS_MIN": "allchords_min",
    "DIAMETER_MIN_ENCLOSING_CIRCLE": "diameter_min_enclosing_circle",
}


def parse_fixture(txt, name):
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
    body = txt.split(table + "{", 1)[1].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"([A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def ellipse_mask():
    """The fixture calculate_ellipse_caliper_values() builds in test_2d_remaining_common.h."""
    a, b, cx, cy = 20.0, 10.0, 26.0, 16.0
    yy, xx = np.mgrid[0:33, 0:53]
    return (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2) <= 1.0


def main():
    data = open(DATA_H, encoding="utf-8", errors="replace").read()
    shape2d = (parse_fixture(data, "shape2d_morphology_mask") > 0)
    ell = ellipse_mask()

    got = {}
    got["DIAMETER_EQUAL_PERIMETER"] = float(macro.perimeter_equal_diameter(NYX_PERIMETER))
    gl, th = macro.geodeticlength_and_thickness(NYX_AREA, NYX_PERIMETER)
    got["GEODETIC_LENGTH"], got["THICKNESS"] = float(gl), float(th)

    e = imea.shape_measurements_2d(ell, spatial_resolution_xy=1.0, dalpha=DALPHA).iloc[0]
    for feat, key in CALIPER_KEYS.items():
        got[feat] = float(e[key])

    print(f"# imea {getattr(imea, '__version__', '0.3.5')}, numpy {np.__version__}, "
          f"dalpha={DALPHA}")
    print("# paste-ready goldens")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},')

    # Evidence block 1: why the 19 shape2d caliper statistics are regression, not imea rows.
    s = imea.shape_measurements_2d(shape2d, spatial_resolution_xy=1.0, dalpha=DALPHA).iloc[0]
    print("\n# raster gap -- imea's own caliper values on the 8x8 shape2d mask, against the "
          "Nyxus snapshots now pinned in test_2d_morphology_regression.h")
    for feat, pin in (("STAT_FERET_DIAM_MIN", 4.47301), ("STAT_FERET_DIAM_MAX", 6.3222),
                      ("STAT_MARTIN_DIAM_MIN", 4.25885), ("STAT_MARTIN_DIAM_MAX", 6.12801),
                      ("STAT_NASSENSTEIN_DIAM_MIN", 1.67316),
                      ("STAT_NASSENSTEIN_DIAM_MAX", 6.24165)):
        got_i = float(s[CALIPER_KEYS[feat]])
        print(f"    {feat:32s} nyxus={pin:8.5f} imea={got_i:8.5f} "
              f"rel={100*abs(got_i-pin)/pin:5.1f}%")

    # Evidence block 2: why the three _MODE statistics cannot be vetted at any honest tolerance.
    print("\n# mode instability -- imea's own mode on the ellipse, across its angular step")
    print(f"    {'dalpha':>6s} {'feret_mode':>11s} {'martin_mode':>12s} {'nassenstein_mode':>17s}")
    for d in (5, 9, 10, 15, 18, 30):
        ed = imea.shape_measurements_2d(ell, spatial_resolution_xy=1.0, dalpha=d).iloc[0]
        print(f"    {d:6d} {float(ed['feret_mode']):11.1f} {float(ed['martin_mode']):12.1f} "
              f"{float(ed['nassenstein_mode']):17.1f}")

    test = open(TEST_H, encoding="utf-8", errors="replace").read()
    pins, tol = {}, {}
    for table, rel in (("morphology_2d_imea_shape2d_ref_vals", RELTOL_ISO),
                       ("morphology_2d_imea_ellipse_ref_vals", RELTOL_CALIPER)):
        for name, value in parse_pins(test, table).items():
            pins[name], tol[name] = value, rel

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
        if rel <= tol[name]:
            print(f"  OK   {name}: imea={have!r} pinned={want!r} rel={rel:.3g}")
            nok += 1
        else:
            print(f"  FAIL {name}: imea={have!r} pinned={want!r} rel={rel:.3g} "
                  f"(tol {tol[name]})")
            nfail += 1

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible")
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
