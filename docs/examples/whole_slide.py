# Example for docs/source/large_images.rst: whole-image (whole-slide) features.
# The page shows the lines between the "example" markers. Everything else creates input
# data and checks the result, so tests/python/test_docs_examples_mechanics.py can run
# the script as written.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(2)
os.makedirs("slides", exist_ok=True)
for k in (1, 2):
    tifffile.imwrite(f"slides/slide{k}.ome.tif", rng.integers(0, 4000, (48, 64), dtype=np.uint16))

# example-start
from nyxus import Nyxus

nyx = Nyxus(["*WHOLESLIDE*"])
features = nyx.featurize_files(
    ["slides/slide1.ome.tif", "slides/slide2.ome.tif"],
    None,              # no label images: each whole image is one ROI
    single_roi=True)

print(features[["intensity_image", "mask_image", "ROI_label", "MEAN", "PERIMETER"]])
# example-end

assert list(features["ROI_label"]) == [1, 1]
assert list(features["mask_image"]) == ["", ""]
# The rows can come back in either order, so each is matched to its file by name
for k in (1, 2):
    pixels = tifffile.imread(f"slides/slide{k}.ome.tif")
    row = features[features["intensity_image"] == f"slides/slide{k}.ome.tif"]
    assert len(row) == 1 and np.isclose(row["MEAN"].iloc[0], pixels.mean())
# The contour is the image's bounding box, so the perimeter is 2 * (width + height)
assert list(features["PERIMETER"]) == [2 * (64 + 48)] * 2
