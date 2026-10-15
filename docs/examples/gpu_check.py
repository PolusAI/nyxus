# Example for docs/source/gpu.rst: using a CUDA GPU when one is available.
# Runs on any build: without a usable GPU it computes the same features on the CPU.
# The page shows the lines between the "example" markers.
import os

import numpy as np
import tifffile

os.makedirs("data/int", exist_ok=True)
os.makedirs("data/seg", exist_ok=True)
rng = np.random.default_rng(0)
intens = rng.integers(100, 4000, (128, 128), dtype=np.uint16)
seg = np.zeros((128, 128), dtype=np.uint16)
seg[10:60, 10:60] = 1
seg[70:120, 20:110] = 2
tifffile.imwrite("data/int/cells.ome.tif", intens)
tifffile.imwrite("data/seg/cells.ome.tif", seg)

# example-start
import nyxus
from nyxus import Nyxus

features = ["*GEOMOMS*", "EROSIONS_2_VANISH"]
nyx = Nyxus(features)

if nyxus.gpu_is_available(id(nyx)):
    for device in nyxus.get_gpu_properties(id(nyx)):
        print(device["Device number"], device["Device name"], device["Memory"])
    nyx = Nyxus(features, use_gpu_device=0)
else:
    print("No CUDA GPU available: the features are computed on the CPU")

result = nyx.featurize_directory("data/int", "data/seg")
print(result.shape)
# example-end

assert result.shape[0] == 2
assert "SPAT_MOMENT_00" in result.columns and "IMOM_RM_00" in result.columns
assert "EROSIONS_2_VANISH" in result.columns
assert not any(c.startswith("GABOR") for c in result.columns)
if not nyxus.gpu_is_available(id(nyx)):
    assert nyxus.get_gpu_properties(id(nyx)) == []
