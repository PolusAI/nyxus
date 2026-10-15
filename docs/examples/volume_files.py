# Example for docs/source/volumetric.rst: 3D features of a labelled volume.
# The page shows the lines between the "example" markers. Everything else creates input
# data and checks the result, so tests/python/test_2d_docs_examples_mechanics.py can run
# the script as written.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(0)
os.makedirs("volumes", exist_ok=True)
image = rng.integers(100, 4000, (16, 48, 48), dtype=np.uint16)
labels = np.zeros((16, 48, 48), dtype=np.uint16)
labels[2:10, 4:20, 4:20] = 1
labels[4:14, 24:44, 20:40] = 2

# example-start
import tifffile
from nyxus import Nyxus3D

# Write a (Z, Y, X) intensity volume and its label volume as OME-TIFF Z stacks
tifffile.imwrite("volumes/cells.ome.tif", image, ome=True, metadata={"axes": "ZYX"})
tifffile.imwrite("volumes/cells_labels.ome.tif", labels, ome=True, metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_ALL_INTENSITY*"])
features = nyx.featurize_files(
    ["volumes/cells.ome.tif"],
    ["volumes/cells_labels.ome.tif"],
    single_roi=False)

print(features[["intensity_image", "ROI_label", "3MEAN", "3MIN", "3MAX"]])
# example-end

assert list(features["ROI_label"]) == [1, 2]
assert list(features["intensity_image"]) == ["cells.ome.tif"] * 2
for k, lab in enumerate((1, 2)):
    assert np.isclose(features["3MEAN"][k], image[labels == lab].mean())
    assert features["3MIN"][k] == image[labels == lab].min()
    assert features["3MAX"][k] == image[labels == lab].max()
