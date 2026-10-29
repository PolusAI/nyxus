# Example for docs/source/Math/f_pixel_intensity.rst: intensity features of the page's 5x5 worked
# example. The page shows the lines between the "example" markers. The assertions below pin the
# worked-example values the page quotes, so the page cannot drift from the code; the page's
# Validation section names the values that were measured once and are not pinned.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

img = np.array([[5, 2, 5, 4, 4],
                [3, 3, 3, 1, 3],
                [2, 1, 1, 1, 3],
                [4, 2, 2, 2, 3],
                [3, 5, 3, 3, 2]], dtype=np.uint32)
mask = np.ones_like(img)

# HISTOGRAM is not part of *ALL_INTENSITY*; name it next to the group to get its columns
nyx = Nyxus(["*ALL_INTENSITY*", "HISTOGRAM"])
features = nyx.featurize(img, mask)
print(features[["MEAN", "STANDARD_DEVIATION", "P25", "P75", "ENTROPY", "HISTOGRAM_BIN_0"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["intensity"]
column = {"HISTOGRAM": "HISTOGRAM_BIN_0"}
for code, facts in extras.items():
    got = features[column.get(code, code)][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code} = {got}, page says {facts['example']}"


def check(table, quoted, label):
    for name, expected in quoted.items():
        got = table[name][0]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
            f"{label} {name} = {got}, page says {expected}"


def bins(table):
    return [table[c][0] for c in table.columns if c.startswith("HISTOGRAM_BIN_")]


# The histogram the page shows: 64 bins, counts 4, 6, 9, 3, 3 in bins 0, 16, 32, 48 and 63
hist = bins(features)
assert len(hist) == 64 and sum(hist) == 25, hist
assert {k: v for k, v in enumerate(hist) if v} == {0: 4, 16: 6, 32: 9, 48: 3, 63: 3}, hist

# HISTOGRAM requested on its own: 64 columns, all 0
alone = bins(Nyxus(["HISTOGRAM"]).featurize(img, mask))
# pins known defect https://github.com/PolusAI/nyxus/issues/489
assert len(alone) == 64 and not any(alone), alone

# Four bins instead of 64: entropy, uniformity and the histogram change; nothing else does
four = Nyxus(["*ALL_INTENSITY*", "HISTOGRAM"], coarse_gray_depth=4).featurize(img, mask)
check(four, {"ENTROPY": 1.942, "UNIFORMITY": 0.2704, "P25": 2.0, "MEAN": 2.8}, "4 bins")
assert bins(four) == [4, 6, 9, 6], bins(four)

# IBSI mode changes nothing
ibsi = Nyxus(["*ALL_INTENSITY*"], ibsi=True).featurize(img, mask)
for code in extras:
    if code != "HISTOGRAM":
        assert ibsi[code][0] == features[code][0], code

# The percentiles are numpy's default linear rule; MATLAB's default prctile is numpy's 'hazen' rule
xs = img.astype(float).ravel()
pct = [features[f"P{q:02d}"][0] for q in (1, 10, 25, 75, 90, 99)]
assert all(math.isclose(a, float(np.percentile(xs, q)), rel_tol=1e-12)
           for a, q in zip(pct, (1, 10, 25, 75, 90, 99))), pct
assert [round(p, 4) for p in pct] == [1, 1, 2, 3, 4.6, 5], pct
hazen = [float(np.percentile(xs, q, method="hazen")) for q in (75, 90)]
assert hazen == [3.25, 5.0], hazen

# A rank between two distinct intensities: 5 ones and 15 threes, then 6 ones and 14 threes
v = np.array([[1] * 5 + [3] * 15], dtype=np.uint32)
check(Nyxus(["*ALL_INTENSITY*"]).featurize(v, np.ones_like(v)), {"P25": 2.5}, "5 ones")
assert np.percentile(v, 25, method="hazen") == 2.0
v = np.array([[1] * 6 + [3] * 14], dtype=np.uint32)
check(Nyxus(["*ALL_INTENSITY*"]).featurize(v, np.ones_like(v)), {"P25": 1.0}, "6 ones")
assert np.percentile(v, 25, method="hazen") == 1.0

# Small ROIs: skewness needs 4 pixels and kurtosis 5; the 5th and 6th moments have no minimum
three = np.array([[1, 2, 4]], dtype=np.uint32)
check(Nyxus(["*ALL_INTENSITY*"]).featurize(three, np.ones_like(three)),
      {"SKEWNESS": 0.0, "KURTOSIS": 0.0, "EXCESS_KURTOSIS": 0.0, "HYPERSKEWNESS": 0.3464,
       "HYPERFLATNESS": 0.7099}, "3 pixels")

# A constant ROI
const = np.full((4, 4), 7, dtype=np.uint32)
flat = Nyxus(["*ALL_INTENSITY*", "HISTOGRAM"]).featurize(const, np.ones_like(const))
check(flat, {"MEAN": 7, "MODE": 7, "P01": 7, "P99": 7, "ROBUST_MEAN": 7, "UNIFORMITY": 1.0,
             "UNIFORMITY_PIU": 100.0, "STANDARD_DEVIATION": 0.0, "COV": 0.0, "SKEWNESS": 0.0,
             "KURTOSIS": 0.0, "HYPERSKEWNESS": 0.0, "QCOD": 0.0, "HISTOGRAM_BIN_0": 16}, "constant")
assert abs(flat["ENTROPY"][0]) < 1e-15

# A pixel of intensity 0 inside the mask is an ROI pixel
zero = img.copy()
zero[0, 0] = 0
check(Nyxus(["*ALL_INTENSITY*"]).featurize(zero, mask), {"MIN": 0.0, "MEAN": 2.6}, "zero in mask")

# In-memory arrays: negative values are shifted up by the minimum, fractions truncated
check(Nyxus(["*ALL_INTENSITY*"]).featurize(img.astype(np.int32) - 10, mask),
      {"MIN": 0.0, "MAX": 4.0, "MEAN": 1.8}, "negative array")
check(Nyxus(["*ALL_INTENSITY*"]).featurize(img + 0.7, mask), {"MIN": 1.0, "MEAN": 2.8}, "float array")

# The IBSI standardizations the page compares with: sigma_b in place of sigma
m = features.iloc[0]
sb, sd = m["STANDARD_DEVIATION_BIASED"], m["STANDARD_DEVIATION"]
cov_ibsi, hf_ibsi, hs_ibsi = sb / m["MEAN"], m["HYPERFLATNESS"] * (sd / sb) ** 6, m["HYPERSKEWNESS"] * (sd / sb) ** 5
check(features, {"COV": 0.4374, "HYPERFLATNESS": 5.769, "HYPERSKEWNESS": 1.227}, "worked example")
assert [round(v, 4) for v in (cov_ibsi, hf_ibsi, hs_ibsi)] == [0.4286, 6.5201, 1.3588], (cov_ibsi, hf_ibsi, hs_ibsi)
gaps = [round(100 * (b - a) / b) for a, b in ((m["COV"], cov_ibsi), (m["HYPERSKEWNESS"], hs_ibsi),
                                               (m["HYPERFLATNESS"], hf_ibsi))]
assert gaps == [-2, 10, 12], gaps

# Grey levels above 65535: each pixel's square wraps modulo 2^32 before it is summed
big = Nyxus(["*ALL_INTENSITY*"]).featurize(img * 20000, mask)
# pins known defect https://github.com/PolusAI/nyxus/issues/490
check(big, {"MEAN": 56000.0, "ENERGY": 5.4145e10, "ROOT_MEAN_SQUARED": 46538.28}, "x 20000")
x = img.astype(float).ravel() * 20000
assert math.isclose(x @ x, 9.28e10) and math.isclose(math.sqrt(x @ x / 25), 60926.18, rel_tol=1e-6)

# UNIFORMITY_PIU of a ROI from -99 to -95 read from a file, in the image's own units
import tifffile
ct = np.array([[-99, -98, -97, -96, -95]], dtype=np.int16)
tifffile.imwrite("ct_int.tif", ct)
tifffile.imwrite("ct_seg.tif", np.ones_like(ct, dtype=np.uint16))
piu = Nyxus(["*ALL_INTENSITY*"]).featurize_files([os.path.abspath("ct_int.tif")],
                                                   [os.path.abspath("ct_seg.tif")], False)
check(piu, {"MIN": -99.0, "MAX": -95.0, "UNIFORMITY_PIU": 102.06}, "CT")
