# Example for docs/source/large_images.rst: forcing ROIs out of core with ram_limit.
# The page shows the lines between the "example" markers. Everything else creates input
# data and checks the result, so tests/python/test_docs_examples_mechanics.py can run
# the script as written.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(3)
os.makedirs("large/int", exist_ok=True)
os.makedirs("large/seg", exist_ok=True)
mask = np.zeros((96, 96), dtype=np.uint16)
mask[4:30, 6:40] = 1
mask[40:90, 10:35] = 2
mask[20:80, 50:92] = 3
tifffile.imwrite("large/int/tissue.ome.tif", rng.integers(100, 4000, (96, 96), dtype=np.uint16))
tifffile.imwrite("large/seg/tissue.ome.tif", mask)

# example-start
from nyxus import Nyxus

features = ["*ALL_INTENSITY*", "*ALL_GLCM*"]

# Default budget: half of the RAM available when the instance is created
in_ram = Nyxus(features).featurize_directory("large/int", "large/seg", ".*")

# A budget of 0 MB makes every ROI oversized, so each one is processed out of core
out_of_core = Nyxus(features, ram_limit=0).featurize_directory("large/int", "large/seg", ".*")

print(in_ram[["ROI_label", "MEAN", "GLCM_CONTRAST_AVE"]])
print(out_of_core[["ROI_label", "MEAN", "GLCM_CONTRAST_AVE"]])
# example-end

a = in_ram.sort_values("ROI_label").reset_index(drop=True)
b = out_of_core.sort_values("ROI_label").reset_index(drop=True)
assert list(a["ROI_label"]) == [1, 2, 3]
for column in a.columns[9:]:
    assert np.allclose(a[column], b[column], rtol=1e-9, equal_nan=True), column
