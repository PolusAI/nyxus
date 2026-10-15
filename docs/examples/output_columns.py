# Example for docs/source/output.rst: the layout of the result table.
# The page shows the lines between the "example" markers.
import os

import numpy as np
import tifffile

# Two image pairs: each label image holds two ROIs
os.makedirs("data/int", exist_ok=True)
os.makedirs("data/seg", exist_ok=True)
rng = np.random.default_rng(0)
for name in ("plate1_well_A1.ome.tif", "plate1_well_B2.ome.tif"):
    intens = rng.integers(100, 4000, (64, 64), dtype=np.uint16)
    seg = np.zeros((64, 64), dtype=np.uint16)
    seg[4:28, 4:28] = 1
    seg[36:60, 30:62] = 2
    tifffile.imwrite("data/int/" + name, intens)
    tifffile.imwrite("data/seg/" + name, seg)

# example-start
from nyxus import Nyxus

nyx = Nyxus(["MEAN", "AREA_PIXELS_COUNT"])

# One row per ROI
features = nyx.featurize_directory("data/int", "data/seg")
print(features.columns.tolist())
print(features[["intensity_image", "ROI_label", "MEAN", "AREA_PIXELS_COUNT"]])

# One row per image: the mean over each image's ROIs
per_image = features.groupby("intensity_image")[["MEAN", "AREA_PIXELS_COUNT"]].mean()
print(per_image)
# example-end

assert features.columns.tolist()[:9] == ["intensity_image", "mask_image", "phys_unit", "ROI_label",
                                         "t_index", "c_index", "phys_x", "phys_y", "phys_z"]
assert features.columns.tolist()[9:] == ["MEAN", "AREA_PIXELS_COUNT"]
assert len(features) == 4 and sorted(features["ROI_label"].unique()) == [1, 2]
assert features["ROI_label"].dtype == np.uint32
assert (features["t_index"] == 0).all() and (features["c_index"] == 0).all()
assert (features["phys_x"] == 1).all()
assert len(per_image) == 2
