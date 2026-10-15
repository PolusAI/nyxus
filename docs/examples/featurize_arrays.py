# Example for docs/source/cmdline_and_examples.rst: features of in-memory numpy arrays.
# The page shows the lines between the "example" markers; the rest checks the result.

# example-start
import numpy as np
from nyxus import Nyxus

# Two 4x5 intensity images and their label images, stacked along the first axis
intens = np.array([
    [[1, 4, 4, 1, 1],
     [1, 4, 6, 1, 1],
     [4, 1, 6, 4, 1],
     [4, 4, 6, 4, 1]],
    [[1, 4, 4, 1, 1],
     [1, 1, 6, 1, 1],
     [1, 1, 3, 1, 1],
     [4, 4, 6, 1, 1]],
])
seg = np.array([
    [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]],
])

nyx = Nyxus(["*ALL_INTENSITY*"])
features = nyx.featurize(intens, seg)
print(features[["intensity_image", "mask_image", "ROI_label", "MEAN", "MEDIAN"]])

# The same call with names for the images
features = nyx.featurize(intens, seg,
                         intensity_names=["cells_a", "cells_b"],
                         label_names=["cells_a_mask", "cells_b_mask"])
print(features[["intensity_image", "mask_image", "ROI_label", "MEAN", "MEDIAN"]])
# example-end

assert list(features["intensity_image"]) == ["cells_a", "cells_b"]
assert np.allclose(features["MEAN"], [2.95, 2.263158], atol=1e-6)
