# Example for docs/source/Math/f_intensity_histogram.rst: IBSI intensity histogram features of the
# page's 5x5 worked example. The page shows the lines between the "example" markers. The assertions
# below pin the worked-example values the page quotes, so the page cannot drift from the code; the
# page's Validation section names the values that were measured once and are not pinned.
import json
import math
import os
import sys
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

# IH_* features exist only in IBSI mode; 5 bins keep the levels 1 to 5 as they are
nyx = Nyxus(["*ALL_IH*"], ibsi=True, coarse_gray_depth=5)
features = nyx.featurize(img, mask)
print(features[["IH_MEAN_IDX", "IH_MEAN_VAL", "IH_P90_IDX", "IH_MAX_GRADIENT"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["intensity_histogram"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"


def ih(values, n_bins, **kw):
    a = np.array([values], dtype=np.uint32)
    return Nyxus(["*ALL_IH*"], ibsi=True, coarse_gray_depth=n_bins, **kw).featurize(a, np.ones_like(a))


def check(table, quoted, label):
    for name, expected in quoted.items():
        got = table[f"IH_{name}"][0]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
            f"{label} IH_{name} = {got}, page says {expected}"


# Values the page quotes for the default 64 bins
default = Nyxus(["*ALL_IH*"], ibsi=True).featurize(img, mask)
check(default, {"NUM_BINS": 64, "BIN_SIZE": 0.0625, "MEAN_VAL": 2.824, "MEAN_IDX": 29.68, "MEDIAN_VAL": 3.031,
                "MEDIAN_IDX": 33, "P90_VAL": 4.948, "P90_IDX": 64, "VARIANCE_IDX": 360.3, "ENTROPY_IDX": 2.182,
                "MAX_GRADIENT": 4.5, "MAX_GRADIENT_IDX": 32}, "64 bins")

# Without IBSI mode the family is dropped from the request, even through *ALL*
assert not [c for c in Nyxus(["*ALL_IH*"]).featurize(img, mask).columns if c.startswith("IH_")]
assert not [c for c in Nyxus(["*ALL*"]).featurize(img, mask).columns if c.startswith("IH_")]

# A constant ROI, and fewer than 2 bins: every feature is the no-value substitute, 0 by default
const = np.full((4, 4), 7, dtype=np.uint32)
for table, label in ((Nyxus(["*ALL_IH*"], ibsi=True).featurize(const, np.ones_like(const)), "constant"),
                     (Nyxus(["*ALL_IH*"], ibsi=True, coarse_gray_depth=1).featurize(img, mask), "1 bin")):
    assert all(table[c][0] == 0 for c in table.columns if c.startswith("IH_")), label

# Median of an even number of pixels: the upper of the two middle bins
check(ih([1, 1, 2, 2], 2), {"MEDIAN_IDX": 2, "MEDIAN_VAL": 1.75}, "even median")

# Mode ties go to the lowest bin
check(ih([1, 1, 3, 4, 4, 5, 5], 5), {"MODE_IDX": 1, "MODE_VAL": 1.4}, "mode tie")

# No bin's gradient above 0: the maximum gradient keeps its seed
# pins known defect https://github.com/PolusAI/nyxus/issues/491
check(ih([1, 2], 2), {"MAX_GRADIENT": sys.float_info.min, "MAX_GRADIENT_IDX": 0, "MIN_GRADIENT": 0,
                      "MIN_GRADIENT_IDX": 1}, "flat histogram")
check(ih([1, 1, 2], 2), {"MAX_GRADIENT": sys.float_info.min, "MAX_GRADIENT_IDX": 0}, "falling histogram")

# A percentile on a bin's upper edge names the next bin, here an empty one
check(ih([1, 3, 3, 3, 3, 3, 3, 3, 3, 5], 4), {"P10_VAL": 2, "P10_IDX": 2, "ROBUST_MEAN_IDX": 3.111}, "edge percentile")

# A value on an inner bin edge can fall one bin low
# pins known defect https://github.com/PolusAI/nyxus/issues/492
check(ih([1, 10, 19], 14), {"MEDIAN_IDX": 7}, "bin edge")

# Pixels of intensity 0 inside the mask take part
zero = img.copy()
zero[0, 0] = 0
check(Nyxus(["*ALL_IH*"], ibsi=True, coarse_gray_depth=5).featurize(zero, mask),
      {"MINIMUM_VAL": 0, "BIN_SIZE": 1.0}, "zero pixel")

# The worked example minus 10, with the example's 5 bins: in memory it is shifted up by its
# minimum; read from a file it keeps its own units. The _IDX features do not change.
import tifffile
shifted = nyx.featurize(img.astype(np.int32) - 10, mask)
check(shifted, {"MINIMUM_VAL": 0, "MEAN_VAL": 1.84}, "in-memory minus 10")
tifffile.imwrite("neg_int.tif", img.astype(np.int16) - 10)
tifffile.imwrite("neg_seg.tif", mask.astype(np.uint16))
from_file = nyx.featurize_files([os.path.abspath("neg_int.tif")], [os.path.abspath("neg_seg.tif")], False)
check(from_file, {"MINIMUM_VAL": -9, "MEAN_VAL": -7.16}, "file minus 10")
for c in (c for c in features.columns if c.startswith("IH_") and c.endswith("_IDX")):
    assert math.isclose(shifted[c][0], features[c][0]) and math.isclose(from_file[c][0], features[c][0]), c
