"""OFFLINE PyRadiomics oracle for the 2D GLSZM features on four small zone-connectivity fixtures.

    python tests/vetting/oracles/gen_glszm_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_2d_glszm_pyradiomics.h,
exiting non-zero on any mismatch, on a fixture the header and this script disagree about, or on a
feature in scope that the header does not pin.

RECIPE `glszm.pyradiomics_connectivity_2d`: each fixture is one 2D ROI read at binWidth=1 with
force2D=True, no resampling. Every fixture's smallest ROI intensity is 1, so PyRadiomics' bin of an
intensity x is floor(x) - floor(1) + 1 = x and its grey levels are the intensities themselves. That is
what Nyxus computes in IBSI mode, where the intensities are not binned. A pixel whose mask is 0 is
outside the ROI on both sides.

WHAT THE FIXTURES ARE FOR. A GLSZM zone is an 8-connected set of same-level pixels. Three fixtures
hold zones that can be reached from their first pixel in raster order only by stepping west, north
or north-east, so a walk that steps only east, south-east, south and south-west splits
them; `forward_zone_count()` below counts that split, and the script fails unless each of the three
fixtures has fewer 8-connected zones than forward-walk zones. The fourth fixture is the negative
case: same-level pixels two steps apart, and two same-level columns joined only through a pixel
outside the mask, which must stay separate zones under any correct connectivity.

  glszm_2d_docs_2x3     the 2 x 3 image of the Nyxus GLSZM page; the bottom row needs a west step
  glszm_2d_pyrad_5x5    the 5 x 5 example of the PyRadiomics GLSZM documentation (11 zones)
  glszm_2d_hook_6x6     a zone that turns north-east and climbs north
  glszm_2d_apart_3x7    the negative case: zones that touch only through a gap or a masked pixel

Each fixture's own 8-connected zone count is checked against `zone_count_8()` (scipy-free flood
fill), so PyRadiomics, the flood fill and the header agree on the zones before any feature is read.

Provenance: tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs
Python <= 3.9); generator=tests/vetting/oracles/gen_glszm_pyradiomics.py. Run offline; CI never
invokes it.
"""
import math
import os
import re
import sys

import numpy
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
TEST_H = os.path.join(TESTS, "test_2d_glszm_pyradiomics.h")

RELTOL = 1e-12        # the pins are PyRadiomics' own output at repr() precision
SCOPE_LOG = ("GLSZM_ZE",)  # carries Nyxus' fast_log10 approximation; pinned, asserted in its own band

PYRAD = {
    "GLSZM_SAE": "SmallAreaEmphasis",
    "GLSZM_LAE": "LargeAreaEmphasis",
    "GLSZM_GLN": "GrayLevelNonUniformity",
    "GLSZM_GLNN": "GrayLevelNonUniformityNormalized",
    "GLSZM_SZN": "SizeZoneNonUniformity",
    "GLSZM_SZNN": "SizeZoneNonUniformityNormalized",
    "GLSZM_ZP": "ZonePercentage",
    "GLSZM_GLV": "GrayLevelVariance",
    "GLSZM_ZV": "ZoneVariance",
    "GLSZM_ZE": "ZoneEntropy",
    "GLSZM_LGLZE": "LowGrayLevelZoneEmphasis",
    "GLSZM_HGLZE": "HighGrayLevelZoneEmphasis",
    "GLSZM_SALGLE": "SmallAreaLowGrayLevelEmphasis",
    "GLSZM_SAHGLE": "SmallAreaHighGrayLevelEmphasis",
    "GLSZM_LALGLE": "LargeAreaLowGrayLevelEmphasis",
    "GLSZM_LAHGLE": "LargeAreaHighGrayLevelEmphasis",
}

# name -> (intensity rows, mask rows or None for an all-ones mask). Must match the header byte for byte.
FIXTURES = {
    "glszm_2d_docs_2x3": (
        [[2, 2, 1],
         [1, 1, 1]], None),
    "glszm_2d_pyrad_5x5": (
        [[5, 2, 5, 4, 4],
         [3, 3, 3, 1, 3],
         [2, 1, 1, 1, 3],
         [4, 2, 2, 2, 3],
         [3, 5, 3, 3, 2]], None),
    "glszm_2d_hook_6x6": (
        [[1, 2, 2, 2, 2, 1],
         [1, 2, 1, 1, 2, 1],
         [1, 2, 1, 3, 2, 1],
         [1, 2, 3, 3, 2, 1],
         [1, 2, 2, 2, 2, 1],
         [1, 1, 1, 1, 1, 3]], None),
    "glszm_2d_apart_3x7": (
        [[1, 3, 1, 2, 4, 4, 4],
         [3, 3, 3, 2, 4, 4, 4],
         [1, 3, 1, 2, 4, 4, 4]],
        [[1, 1, 1, 1, 1, 0, 1],
         [1, 1, 1, 1, 1, 0, 1],
         [1, 1, 1, 1, 1, 0, 1]]),
}
SPLIT_BY_FORWARD_WALK = ("glszm_2d_docs_2x3", "glszm_2d_pyrad_5x5", "glszm_2d_hook_6x6")

FORWARD = ((0, 1), (1, 1), (1, 0), (1, -1))
EIGHT = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def _zones(img, mask, steps):
    """Zone sizes per grey level, growing each zone from its first raster pixel through `steps`."""
    h, w = img.shape
    seen = numpy.zeros_like(img, dtype=bool)
    zones = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            g = img[r, c]
            seen[r, c] = True
            stack, size = [(r, c)], 1
            while stack:
                y, x = stack.pop()
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx] and img[ny, nx] == g:
                        seen[ny, nx] = True
                        size += 1
                        stack.append((ny, nx))
            zones.append((int(g), size))
    return zones


def zone_count_8(img, mask):
    return len(_zones(img, mask, EIGHT))


def forward_zone_count(img, mask):
    return len(_zones(img, mask, FORWARD))


def pyradiomics_values(img, mask):
    radiomics.setVerbosity(40)
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=1, force2D=True, force2Ddimension=0, label=1, resampledPixelSpacing=None)
    ext.disableAllFeatures()
    ext.enableFeatureClassByName("glszm")
    # PyRadiomics refuses a mask with no 0 in it, so each fixture gets a one-pixel border outside the
    # mask; it crops to the ROI's bounding box before building the matrix, so the border adds nothing.
    img, mask = numpy.pad(img, 1), numpy.pad(mask, 1)
    im = sitk.GetImageFromArray(img[numpy.newaxis].astype(numpy.float64))
    ma = sitk.GetImageFromArray(mask[numpy.newaxis].astype(numpy.uint8))
    res = ext.execute(im, ma)
    return {code: float(res["original_glszm_" + name]) for code, name in PYRAD.items()}


def header_pins():
    """{fixture: {feature: value}} parsed from the header's glszm_2d_pyradiomics_<fixture>_ref_vals tables."""
    text = open(TEST_H, encoding="utf-8").read()
    pins = {}
    for m in re.finditer(r"static const ref_vals_map<double> glszm_2d_pyradiomics_(\w+?)_ref_vals\s*\{(.*?)\};", text, re.S):
        pins["glszm_2d_" + m.group(1)] = {k: float(v) for k, v in re.findall(r'\{"(GLSZM_\w+)",\s*([-0-9.eE+]+)\}', m.group(2))}
    return pins


def main():
    failures = []
    pins = header_pins() if os.path.exists(TEST_H) else {}
    for name, (rows, mrows) in FIXTURES.items():
        img = numpy.array(rows, dtype=numpy.int64)
        mask = numpy.ones_like(img) if mrows is None else numpy.array(mrows, dtype=numpy.int64)
        assert img[mask > 0].min() == 1, f"{name}: smallest ROI intensity must be 1 for binWidth=1 to keep the levels"
        n8, nf = zone_count_8(img, mask), forward_zone_count(img, mask)
        if name in SPLIT_BY_FORWARD_WALK and not nf > n8:
            failures.append(f"{name}: the forward walk finds {nf} zones, no more than the {n8} 8-connected ones")
        if name not in SPLIT_BY_FORWARD_WALK and nf != n8:
            failures.append(f"{name}: the negative fixture must not discriminate the walk ({nf} vs {n8})")
        vals = pyradiomics_values(img, mask)
        npix = int((mask > 0).sum())
        if not math.isclose(vals["GLSZM_ZP"], n8 / npix, rel_tol=1e-12):
            failures.append(f"{name}: PyRadiomics ZP {vals['GLSZM_ZP']} is not {n8} zones / {npix} pixels")
        print(f"// {name}: {n8} zones (8-connected), {nf} under an E/SE/S/SW walk, {npix} pixels")
        table = name.replace("glszm_2d_", "glszm_2d_pyradiomics_") + "_ref_vals"
        print(f"static const ref_vals_map<double> {table}")
        print("{")
        print(",\n".join(f'    {{"{k}", {v!r}}}' for k, v in vals.items()))
        print("};\n")
        got = pins.get(name)
        if got is None:
            failures.append(f"{name}: no table in {os.path.basename(TEST_H)}")
            continue
        for k, v in vals.items():
            if k not in got:
                failures.append(f"{name}: {k} not pinned")
            elif not math.isclose(got[k], v, rel_tol=RELTOL):
                failures.append(f"{name}: {k} pinned {got[k]!r}, PyRadiomics gives {v!r}")
    if failures:
        print("\n".join("FAIL " + f for f in failures), file=sys.stderr)
        sys.exit(1)
    print("OK: every pin reproduces")


if __name__ == "__main__":
    main()
