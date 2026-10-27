# Example for docs/source/Math/f_glrlm_3d.rst: 3D GLRLM features of the page's 3x3x3 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, so the page cannot drift from the code. The volume is written as OME-TIFF files into
# the current directory.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# a 3 x 3 x 3 volume, indexed (z, y, x), with four grey levels
img = np.array([[[1, 2, 4], [3, 1, 2], [3, 3, 1]],
                [[2, 2, 3], [4, 1, 3], [2, 1, 1]],
                [[3, 4, 2], [2, 2, 3], [1, 2, 1]]], dtype=np.uint16)
mask = np.ones_like(img)

tifffile.imwrite("int.ome.tif", img, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
tifffile.imwrite("seg.ome.tif", mask, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_GLRLM*"], ibsi=True)
features = nyx.featurize_files([os.path.abspath("int.ome.tif")], [os.path.abspath("seg.ome.tif")], False)
print(features[["3GLRLM_SRE", "3GLRLM_RP", "3GLRLM_SRE_AVE", "3GLRLM_RP_AVE"]])
# example-end


def run(img, mask, metaparam=None, **kwargs):
    tifffile.imwrite("int.ome.tif", img, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    tifffile.imwrite("seg.ome.tif", mask, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    n = Nyxus3D(["*3D_GLRLM*"], **kwargs)
    if metaparam:
        n.set_metaparam(metaparam)
    return n.featurize_files([os.path.abspath("int.ome.tif")], [os.path.abspath("seg.ome.tif")], False)


def check(table, column, expected, what):
    got = table[column][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"{what} {column} = {got}, page says {expected}"


# Every plain-code value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["glrlm_3d"]
for code, facts in extras.items():
    check(features, code, facts["example"], "IBSI")

# The direction averages the page quotes
check(features, "3GLRLM_SRE_AVE", 0.9033, "IBSI")
check(features, "3GLRLM_RP_AVE", 0.8803, "IBSI")

# No binning (coarse_gray_depth=0, or 3glrlm/greydepth=0 for this family): the same values as IBSI mode
for kwargs, meta in (({"coarse_gray_depth": 0}, None), ({}, "3glrlm/greydepth=0")):
    table = run(img, mask, metaparam=meta, **kwargs)
    for code, facts in extras.items():
        check(table, code, facts["example"], f"no binning {kwargs} {meta}")

# Default settings: 64 MATLAB-style levels map 1, 2, 3, 4 to 17, 33, 49, 64; the runs are unchanged,
# the level-weighted features move
default = run(img, mask)
for code in ("3GLRLM_SRE", "3GLRLM_RP", "3GLRLM_SRE_AVE", "3GLRLM_RP_AVE"):
    check(default, code, features[code][0], "default")
check(default, "3GLRLM_GLV", 229.4, "default")
check(default, "3GLRLM_HGLRE", 1707, "default")

# A depth of 2 (MATLAB-style binning), from the coarse grey depth or the family's metaparameter
for kwargs, meta in (({"coarse_gray_depth": 2}, None), ({}, "3glrlm/greydepth=2")):
    binned = run(img, mask, metaparam=meta, **kwargs)
    check(binned, "3GLRLM_RP", 0.7407, f"depth 2 {kwargs} {meta}")
    check(binned, "3GLRLM_GLV", 0.1875, f"depth 2 {kwargs} {meta}")

# Two voxels of the bounding box outside the mask, 3glrlm/greydepth=2: they form no run
holes = np.ones_like(img)
holes[0, 0, 2] = holes[2, 0, 0] = 0
masked = run(img, holes, metaparam="3glrlm/greydepth=2")
check(masked, "3GLRLM_RP", 0.72, "masked greydepth=2")
check(masked, "3GLRLM_LGLRE", 0.4583, "masked greydepth=2")

# Constant ROI: every 3D GLRLM column, the _AVE columns included, is the substitute (0 by default)
const = np.full((2, 2, 2), 7, dtype=np.uint16)
table = run(const, np.ones_like(const), ibsi=True)
assert all(table[c][0] == 0 for c in table.columns if c.startswith("3GLRLM_")), "constant ROI"