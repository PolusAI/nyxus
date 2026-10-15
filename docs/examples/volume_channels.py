# Example for docs/source/volumetric.rst: a 5D OME-TIFF with channels and timepoints.
# The page shows the lines between the "example" markers. Everything else creates input
# data and checks the result, so tests/python/test_2d_docs_examples_mechanics.py can run
# the script as written.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(1)
os.makedirs("volumes", exist_ok=True)
# 2 timepoints, 3 channels, 8 Z planes of 32 x 32 pixels
image = rng.integers(100, 1000, (2, 3, 8, 32, 32), dtype=np.uint16)
image += (1000 * np.arange(3, dtype=np.uint16))[None, :, None, None, None]
labels = np.zeros((8, 32, 32), dtype=np.uint16)
labels[1:6, 4:14, 4:14] = 1

# example-start
import tifffile
from nyxus import Nyxus3D

# One file holds every timepoint (T) and channel (C); the label volume has one frame
tifffile.imwrite("volumes/series.ome.tif", image, ome=True, metadata={"axes": "TCZYX"})
tifffile.imwrite("volumes/series_labels.ome.tif", labels, ome=True, metadata={"axes": "ZYX"})

nyx = Nyxus3D(["3MEAN"])
features = nyx.featurize_files(
    ["volumes/series.ome.tif"], ["volumes/series_labels.ome.tif"], single_roi=False)
print(features[["ROI_label", "t_index", "c_index", "3MEAN"]])

# Keep only channel 1
channel_1 = features[features["c_index"] == 1]
print(channel_1[["t_index", "3MEAN"]])
# example-end

assert len(features) == 6
for _, row in features.iterrows():
    t, c = int(row["t_index"]), int(row["c_index"])
    assert np.isclose(row["3MEAN"], image[t, c][labels == 1].mean())
assert sorted(channel_1["t_index"].astype(int)) == [0, 1]
