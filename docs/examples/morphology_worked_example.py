# Example for docs/source/Math/f_morphology.rst: 2D morphology features of the page's worked
# example, a notched ROI (label 1) with a small rectangle touching it (label 2). The page shows the
# lines between the "example" markers. The assertions below pin every value the page quotes, so the
# page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

roi = """
..XXX...XXX.
.XXXX...XXXX
XXXXX...XXXX
XXXXX...XXXX
XXXXX...XXXX
XXXXXXXXXXXX
XXXXXXXXXXXX
XXXXXXXXXXXX
XXXXXXXXXXXX
XXXXXXXXXXX.
.XXXXXXXXX..
..XXXXXXX...
""".split()

mask = np.zeros((14, 17), dtype=np.uint32)
mask[1:13, 1:13] = [[c == "X" for c in row] for row in roi]   # label 1: x 1..12, y 1..12
mask[4:9, 13:16] = 2                                          # label 2: x 13..15, y 4..8
y, x = np.mgrid[0:14, 0:17]
img = (1 + (3 * x + 5 * y) % 9).astype(np.uint32)             # grey levels 1 to 9

nyx = Nyxus(["*ALL_MORPHOLOGY*"])
features = nyx.featurize(img, mask)
print(features[["ROI_label", "AREA_PIXELS_COUNT", "PERIMETER", "CONVEX_HULL_AREA",
                "ORIENTATION", "STAT_FERET_DIAM_MAX", "NUM_NEIGHBORS", "PERCENT_TOUCHING"]])
# example-end

# Every value in the page's feature indexes, as recorded in docs/feature_extras.toml (ROI 1)
# Some of these values pin known defects; feature_extras.toml names the issue above each of them.
FAMILIES = ["basic_morphology", "contour", "convex_hull", "ellipse", "circles", "roi_radius", "extrema",
            "euler", "geodetic", "feret", "martin", "nassenstein", "chords", "neighbors"]
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))
one = features[features["ROI_label"] == 1].reset_index(drop=True)
two = features[features["ROI_label"] == 2].reset_index(drop=True)
for family in FAMILIES:
    assert extras[family], family
    for code, facts in extras[family].items():
        got = one[code][0]
        assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
            f"{code} = {got}, page says {facts['example']}"


def check(table, quoted, label):
    for name, expected in quoted.items():
        got = table[name][0]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-9), \
            f"{label} {name} = {got}, page says {expected}"


def run(image, labels, **kwargs):
    t = Nyxus(["*ALL_MORPHOLOGY*"], **kwargs).featurize(image, labels)
    return t.sort_values("ROI_label").reset_index(drop=True)


# The worked example as the page walks through it
check(one, {"POLYGONALITY_AVE": -1, "HEXAGONALITY_AVE": -1, "EDGE_INTEGRATED_INTENSITY": 228}, "ROI 1")
assert round(one["EDGE_INTEGRATED_INTENSITY"][0] / one["EDGE_MEAN_INTENSITY"][0]) == 44   # contour pixels
check(two, {"AREA_PIXELS_COUNT": 15, "PERIMETER": 12, "CIRCULARITY": 1.144, "ASPECT_RATIO": 0.6,
            "ORIENTATION": 90, "CONVEX_HULL_AREA": 15, "PERCENT_TOUCHING": 41.67,
            "CLOSEST_NEIGHBOR1_ANG": 174.4, "DIAMETER_MIN_ENCLOSING_CIRCLE": 4.4723,
            "DIAMETER_CIRCUMSCRIBING_CIRCLE": 4.472, "DIAMETER_INSCRIBING_CIRCLE": 2,
            "ROI_RADIUS_MEAN": 0.2, "ROI_RADIUS_MAX": 1, "ROI_RADIUS_MEDIAN": 0,
            "MAXCHORDS_MIN": 2, "MAXCHORDS_MEAN": 3.722}, "ROI 2")

# Settings that change nothing in memory: IBSI mode, the grey-level depth, the anisotropy keywords
cols = list(features.columns)
codes = cols[cols.index("AREA_PIXELS_COUNT"):]
for kwargs in ({"ibsi": True}, {"coarse_gray_depth": 8}, {"anisotropy_x": 2.0}):
    other = run(img, mask, **kwargs)
    for code in codes:
        assert np.allclose(other[code], features.sort_values("ROI_label")[code]), (kwargs, code)

# AREA_UM2 is 0 whatever the resolution
# pins known defect https://github.com/PolusAI/nyxus/issues/496
assert run(img, mask, pixels_per_micron=2.0)["AREA_UM2"].tolist() == [0, 0]

# ORIENTATION: a bar running down and to the right in the image is at about +45 degrees
bar = np.zeros((20, 20), dtype=np.uint32)
for k in range(2, 17):
    bar[k, k - 1:k + 2] = 1
check(run(bar * 5, bar), {"ORIENTATION": 44.49}, "diagonal bar")

# A ROI with all-zero intensities: weighted centroid (0, 0), mass displacement = |centroid|
zero = img.copy()
zero[mask == 2] = 0
z2 = run(zero, mask).iloc[[1]].reset_index(drop=True)
check(z2, {"WEIGHTED_CENTROID_X": 0, "WEIGHTED_CENTROID_Y": 0, "MASS_DISPLACEMENT": 15.23,
           "EDGE_MEAN_INTENSITY": 0, "MAXCHORDS_MAX": 0}, "zero-intensity ROI")

# One pixel of intensity 0 inside ROI 1 splits chords
hole = img.copy()
hole[9, 6] = 0
check(run(hole, mask), {"AREA_PIXELS_COUNT": 116, "MAXCHORDS_MEAN": 8.8, "ALLCHORDS_MEAN": 5.197},
      "zero pixel")

# PERIMETER of a ROI with a hole, and of a label in two pieces: the sum of their closed walks. The
# circles of the holed square are measured to its outer border only.
sq = np.zeros((13, 13), dtype=np.uint32)
sq[2:11, 2:11] = 1
sq[5:8, 5:8] = 0
check(run(sq * 4, sq), {"PERIMETER": 45.66, "EULER_NUMBER": 0, "DIAMETER_CIRCUMSCRIBING_CIRCLE": 11.31,
                        "DIAMETER_INSCRIBING_CIRCLE": 8}, "holed square")
pieces = np.zeros((10, 16), dtype=np.uint32)
pieces[2:7, 2:6] = 1
pieces[2:7, 10:14] = 1
check(run(pieces * 4, pieces), {"PERIMETER": 28, "EULER_NUMBER": 2}, "two pieces")
one_piece = np.zeros((10, 10), dtype=np.uint32)
one_piece[2:7, 2:6] = 1
check(run(one_piece * 4, one_piece), {"PERIMETER": 14}, "one piece")

# A 1 x 13 bar: walked on both sides, so PERIMETER is 24; hull of two vertices, Nassenstein 0,
# Martin only rounding residue below 1e-6
line = np.zeros((7, 20), dtype=np.uint32)
line[3, 3:16] = 1
flat = run(line * 100, line)
check(flat, {"PERIMETER": 24}, "1 x 13 bar")

# A single pixel is its own contour; two pixels are walked there and back
dot = run(np.pad(np.full((1, 1), 7, np.uint32), 2), np.pad(np.ones((1, 1), np.uint32), 2))
check(dot, {"PERIMETER": 0, "EDGE_MEAN_INTENSITY": 7, "EDGE_INTEGRATED_INTENSITY": 7, "CIRCULARITY": 0,
            "DIAMETER_MIN_ENCLOSING_CIRCLE": 2e-4, "DIAMETER_CIRCUMSCRIBING_CIRCLE": 0,
            "DIAMETER_INSCRIBING_CIRCLE": 0, "ROI_RADIUS_MAX": 0, "CONVEX_HULL_AREA": 1}, "one pixel")
pair = run(np.pad(np.full((1, 2), 7, np.uint32), 2), np.pad(np.ones((1, 2), np.uint32), 2))
check(pair, {"PERIMETER": 2, "CIRCULARITY": 2.507}, "two pixels")
for stat in ("MIN", "MAX", "MEAN", "MEDIAN", "STDDEV", "MODE"):
    assert flat[f"STAT_NASSENSTEIN_DIAM_{stat}"][0] == 0, stat
    assert 0 <= flat[f"STAT_MARTIN_DIAM_{stat}"][0] < 1e-6, stat

# The neighbour gate is the gap between contours: ROI 2 moved right, 3 pixels from ROI 1
gap = mask.copy()
gap[gap == 2] = 0
gap[4:9, 15:17] = 2
for distance, expected in ((2, 0), (3, 1)):
    g = run(img, gap, neighbor_distance=distance)
    assert g["NUM_NEIGHBORS"].tolist() == [expected, expected], (distance, g["NUM_NEIGHBORS"].tolist())
check(run(img, gap, neighbor_distance=3), {"CLOSEST_NEIGHBOR1_DIST": 9.126, "PERCENT_TOUCHING": 0}, "gap 3")

# Two neighbours on either side of the x axis: linear mean, n-1 deviation of 2 angles is 0
scene = np.zeros((30, 30), dtype=np.uint32)
scene[12:17, 10:15] = 1
scene[9:12, 18:23] = 2
scene[17:20, 18:23] = 3
# pins known defect https://github.com/PolusAI/nyxus/issues/499
check(run(scene * 2, scene), {"NUM_NEIGHBORS": 2, "CLOSEST_NEIGHBOR1_ANG": 333.4, "CLOSEST_NEIGHBOR2_ANG": 26.57,
                              "ANG_BW_NEIGHBORS_MEAN": 180, "ANG_BW_NEIGHBORS_STDDEV": 0,
                              "ANG_BW_NEIGHBORS_MODE": 27}, "two neighbours")

# A disk with four neighbours: polygonality and hexagonality
yy, xx = np.mgrid[0:40, 0:40]
disk = np.zeros((40, 40), dtype=np.uint32)
disk[(xx - 20) ** 2 + (yy - 20) ** 2 <= 36] = 1
disk[14:27, 28:32] = 2
disk[14:27, 8:12] = 3
disk[8:12, 14:27] = 4
disk[28:32, 16:24] = 5
dimg = (1 + (2 * xx + 7 * yy) % 11).astype(np.uint32)
check(run(dimg, disk), {"NUM_NEIGHBORS": 4, "POLYGONALITY_AVE": 8.484, "HEXAGONALITY_AVE": 9.223,
                        "HEXAGONALITY_STDDEV": 0.08352, "ANG_BW_NEIGHBORS_MEAN": 135.8}, "disk")

# Codes that need companions: requested without the family that computes their inputs
# pins known defect https://github.com/PolusAI/nyxus/issues/495


def alone(codes, image=img, labels=mask):
    t = Nyxus(codes).featurize(image, labels)
    return t[t["ROI_label"] == 1].reset_index(drop=True)


check(alone(["MAJOR_AXIS_LENGTH", "ORIENTATION", "ROUNDNESS"]),
      {"MAJOR_AXIS_LENGTH": 39.36, "ORIENTATION": 45.93, "ROUNDNESS": 0}, "ellipse codes alone")
assert alone(["ELONGATION"])["ELONGATION"][0] == 0
circle_codes = ["DIAMETER_MIN_ENCLOSING_CIRCLE", "DIAMETER_CIRCUMSCRIBING_CIRCLE", "DIAMETER_INSCRIBING_CIRCLE"]
assert (alone(circle_codes)[circle_codes].to_numpy() == 0).all()
check(alone(circle_codes + ["PERIMETER"]),
      {"DIAMETER_MIN_ENCLOSING_CIRCLE": 14.17, "DIAMETER_CIRCUMSCRIBING_CIRCLE": 30}, "circles with PERIMETER")
hex_codes = ["HEXAGONALITY_AVE", "HEXAGONALITY_STDDEV"]
assert (alone(hex_codes, dimg, disk)[hex_codes].to_numpy() == 0).all()

# Whole-slide mode on a 64 x 48 image: the box is one pixel larger than the image
# pins known defect https://github.com/PolusAI/nyxus/issues/493
import tifffile

tifffile.imwrite("whole.tif", np.full((48, 64), 7, np.uint32))
whole = Nyxus(["*ALL_MORPHOLOGY*"]).featurize_files([os.path.abspath("whole.tif")],
                                                    [os.path.abspath("whole.tif")], True)
check(whole, {"PERIMETER": 224, "EXTREMA_P3_X": 64, "EXTREMA_P3_Y": -1, "EXTREMA_P4_X": 64,
              "EXTREMA_P4_Y": -1, "EXTREMA_P5_X": -1, "EXTREMA_P5_Y": 48, "EXTREMA_P6_X": -1,
              "EXTREMA_P6_Y": 48}, "whole slide 64 x 48")
