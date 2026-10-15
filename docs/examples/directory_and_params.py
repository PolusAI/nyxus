# Example for docs/source/cmdline_and_examples.rst: a directory of image pairs, and reading
# and changing parameters. The page shows the lines between the "example" markers.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(0)
os.makedirs("data/intensities", exist_ok=True)
os.makedirs("data/labels", exist_ok=True)
for k in (1, 2):
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[4:20, 4:20] = 1
    mask[30:50, 8:24] = 2
    tifffile.imwrite(f"data/intensities/img{k}.ome.tif", rng.integers(100, 4000, (64, 64), dtype=np.uint16))
    tifffile.imwrite(f"data/labels/img{k}.ome.tif", mask)

# example-start
from nyxus import Nyxus

nyx = Nyxus(["*ALL_INTENSITY*", "*ALL_GLCM*"])

# Pairs files with the same name in the two directories
features = nyx.featurize_directory("data/intensities", "data/labels", file_pattern=r".*\.ome\.tif")
print(features.shape)

print(nyx.get_params("coarse_gray_depth", "n_feature_calc_threads"))
nyx.set_params(coarse_gray_depth=256)
print(nyx.get_params("coarse_gray_depth", "n_feature_calc_threads"))
# example-end

assert features.shape[0] == 4
p = nyx.get_params()
assert p["coarse_gray_depth"] == 256 and p["n_feature_calc_threads"] == 4
assert {"gabor_freqs", "gabor_thetas", "dynamic_range", "ram_limit"} <= set(p)
