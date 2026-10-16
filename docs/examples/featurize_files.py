# Example for docs/source/cmdline_and_examples.rst: explicit intensity/mask pairs.
# The page shows the lines between the "example" markers. Everything else creates input
# data and checks the result, so tests/python/test_docs_examples_mechanics.py can run
# the script as written.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(0)
os.makedirs("images/intensities", exist_ok=True)
os.makedirs("images/labels", exist_ok=True)
for k in (1, 2):
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[4:20, 4:20] = 1
    mask[30:50, 8:24] = 2
    mask[12:40, 34:60] = 3
    tifffile.imwrite(f"images/intensities/i{k}.ome.tif", rng.integers(100, 4000, (64, 64), dtype=np.uint16))
    tifffile.imwrite(f"images/labels/m{k}.ome.tif", mask)

# example-start
from nyxus import Nyxus

nyx = Nyxus(["*ALL_INTENSITY*"])
features = nyx.featurize_files(
    ["images/intensities/i1.ome.tif", "images/intensities/i2.ome.tif"],
    ["images/labels/m1.ome.tif", "images/labels/m2.ome.tif"],
    single_roi=False)

print(features[["intensity_image", "mask_image", "ROI_label", "MEAN"]])
# example-end

assert list(features["ROI_label"]) == [1, 2, 3, 1, 2, 3]
assert list(features["intensity_image"]) == ["i1.ome.tif"] * 3 + ["i2.ome.tif"] * 3
