# Example for docs/source/troubleshooting.rst: the errors of a misspelled feature name and of a
# file pattern that selects nothing. The page shows the lines between the "example" markers.
# Everything else creates input data and checks the result, so
# tests/python/test_docs_examples_mechanics.py can run the script as written.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(0)
os.makedirs("data/int", exist_ok=True)
os.makedirs("data/seg", exist_ok=True)
mask = np.zeros((64, 64), dtype=np.uint16)
mask[4:20, 4:20] = 1
mask[30:50, 8:24] = 2
tifffile.imwrite("data/int/img1.tif", rng.integers(100, 4000, (64, 64), dtype=np.uint16))
tifffile.imwrite("data/seg/img1.tif", mask)

# example-start
from nyxus import Nyxus

# Python feature names are upper case
try:
    Nyxus(["mean"])
except ValueError as err:
    print(err)

nyx = Nyxus(["MEAN"])

# The file pattern is a regular expression, not a wildcard
try:
    nyx.featurize_directory("data/int", "data/seg", file_pattern="*.tif")
except ValueError as err:
    print(err)

# The pattern must match the whole file name
try:
    nyx.featurize_directory("data/int", "data/seg", file_pattern=r".*\.tiff")
except RuntimeError as err:
    print(err)

features = nyx.featurize_directory("data/int", "data/seg", file_pattern=r".*\.tif")
print(features[["intensity_image", "ROI_label", "MEAN"]])
# example-end

messages = []
for call in (lambda: Nyxus(["mean"]),
             lambda: nyx.featurize_directory("data/int", "data/seg", file_pattern="*.tif"),
             lambda: nyx.featurize_directory("data/int", "data/seg", file_pattern=r".*\.tiff")):
    try:
        call()
        messages.append(None)
    except (ValueError, RuntimeError) as err:
        messages.append(str(err))
assert messages[0].startswith("Error: 'mean' is not a valid 2D feature name")
assert messages[1] == "Invalid filepattern *.tif"
assert messages[2] == ("Error reading the dataset: no intensity and/or label files to process, "
                       "probably due to file pattern .*\\.tiff")
assert list(features["ROI_label"]) == [1, 2]
