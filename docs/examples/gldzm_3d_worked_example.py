# Example for docs/source/Math/f_gldzm_3d.rst: 3D GLDZM features of the page's 3 x 4 x 4 worked
# example. The page shows the lines between the "example" markers. The assertions below pin every
# value the page quotes, in IBSI mode and with the default settings, so the page cannot drift from
# the code. The volumes are written as OME-TIFF files into the current directory.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# grey levels, indexed (z, y, x)
vol = np.array([[[5, 1, 1, 1],
                 [1, 1, 1, 1],
                 [2, 2, 2, 2],
                 [2, 2, 2, 2]],
                [[1, 1, 1, 1],
                 [1, 3, 4, 1],
                 [2, 3, 4, 2],
                 [2, 2, 2, 2]],
                [[1, 1, 2, 3],
                 [1, 1, 2, 2],
                 [2, 2, 2, 4],
                 [2, 2, 2, 5]]], dtype=np.uint16)
mask = np.ones_like(vol)

tifffile.imwrite("int.ome.tif", vol, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
tifffile.imwrite("seg.ome.tif", mask, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_GLDZM*"], ibsi=True)
features = nyx.featurize_files(["./int.ome.tif"], ["./seg.ome.tif"], False)
print(features[["3GLDZM_SDE", "3GLDZM_LDE", "3GLDZM_ZP", "3GLDZM_ZDE"]])
# example-end


def run(volume, labels, **settings):
    tifffile.imwrite("int_x.ome.tif", volume, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    tifffile.imwrite("seg_x.ome.tif", labels, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    return Nyxus3D(["*3D_GLDZM*"], **settings).featurize_files(["./int_x.ome.tif"], ["./seg_x.ome.tif"], False)


def check(table, quoted, what):
    for name, expected in quoted.items():
        got = table[f"3GLDZM_{name}"][0]
        assert math.isclose(got, expected, rel_tol=5e-4), f"{what} 3GLDZM_{name} = {got}, page says {expected}"


# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["gldzm_3d"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"

# Values the page quotes for the default settings (64-level binning)
check(run(vol, mask), {"GLM": 42.43, "GLV": 311.67, "HGLZE": 2111.9, "LGLZE": 0.001367, "ZP": 0.1458},
      "default")

# Values the page quotes for a mask that leaves one voxel of the bounding box out of the ROI;
# the voxel's intensity does not matter, so it is set to 0 as a segmentation would leave it
hole = mask.copy()
hole[0, 1, 1] = 0
holed = vol.copy()
holed[0, 1, 1] = 0
quoted = {"SDE": 1.0, "ZDM": 1.0, "ZP": 0.1489}
check(run(holed, hole, ibsi=True), quoted, "masked IBSI")
check(run(holed, hole), quoted, "masked default")

# Values the page quotes for an ROI voxel of intensity 0 in IBSI mode
zero = vol.copy()
zero[1, 1, 1] = 0
check(run(zero, mask, ibsi=True), {"GLM": 3.875, "ZP": 0.1667}, "zero-voxel IBSI")
