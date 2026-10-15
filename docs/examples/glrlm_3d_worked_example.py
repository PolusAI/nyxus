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

# Default settings: no binning, the same values as IBSI mode; --coarseGrayDepth has no effect
for kwargs in ({}, {"coarse_gray_depth": 2}):
    table = run(img, mask, **kwargs)
    for code, facts in extras.items():
        check(table, code, facts["example"], f"default {kwargs}")

# 3glrlm/greydepth=2 (MATLAB-style binning)
binned = run(img, mask, metaparam="3glrlm/greydepth=2")
check(binned, "3GLRLM_RP", 0.7407, "greydepth=2")
check(binned, "3GLRLM_GLV", 0.1875, "greydepth=2")

# Two voxels of the bounding box outside the mask, 3glrlm/greydepth=2
holes = np.ones_like(img)
holes[0, 0, 2] = holes[2, 0, 0] = 0
masked = run(img, holes, metaparam="3glrlm/greydepth=2")
check(masked, "3GLRLM_RP", 0.8, "masked greydepth=2")
check(masked, "3GLRLM_LGLRE", 0.5125, "masked greydepth=2")

# Run percentage above 1: an ellipsoid with semi-axes 12, 30 and 30 voxels in a 24 x 60 x 60 box,
# intensities drawn uniformly from 1 to 255; the bounding-box voxels outside the mask form runs
z, y, x = np.mgrid[0:24, 0:60, 0:60]
inside = ((z - 11.5) / 12) ** 2 + ((y - 29.5) / 30) ** 2 + ((x - 29.5) / 30) ** 2 <= 1
assert inside.sum() == 45224
noise = np.random.default_rng(0).integers(1, 256, size=inside.shape)
ellipsoid = np.where(inside, noise, 0).astype(np.uint16)
emask = inside.astype(np.uint16)
for depth, rp in ((16, 1.131), (64, 1.182), (128, 1.191)):
    check(run(ellipsoid, emask, metaparam=f"3glrlm/greydepth={depth}"), "3GLRLM_RP", rp,
          f"ellipsoid greydepth={depth}")
check(run(ellipsoid, emask, metaparam="3glrlm/greydepth=64"), "3GLRLM_RP_AVE", 1.130,
      "ellipsoid greydepth=64")
unbinned = run(ellipsoid, emask)
check(unbinned, "3GLRLM_RP", 0.9962, "ellipsoid unbinned")
check(unbinned, "3GLRLM_RP_AVE", 0.9965, "ellipsoid unbinned")

# A ROI that fills its bounding box: a 20 x 20 x 20 cube of the same intensities
cube = noise[:20, :20, :20].astype(np.uint16)
check(run(cube, np.ones_like(cube), metaparam="3glrlm/greydepth=64"), "3GLRLM_RP", 0.9863,
      "cube greydepth=64")
