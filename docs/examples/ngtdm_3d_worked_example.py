# Example for docs/source/Math/f_ngtdm_3d.rst: 3D NGTDM features of the page's 3 x 3 x 3 worked
# example. The page shows the lines between the "example" markers. The assertions below pin every
# value the page quotes, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# planes z = 0, 1, 2 of a 3 x 3 x 3 ROI that fills its bounding box
vol = np.array([[[1, 2, 2], [3, 1, 4], [2, 2, 3]],
                [[4, 3, 1], [2, 4, 3], [1, 3, 2]],
                [[2, 1, 3], [4, 2, 1], [3, 4, 4]]], dtype=np.uint16)
mask = np.ones_like(vol)


def write(name, arr):
    path = str(Path(name).resolve())
    tifffile.imwrite(path, arr, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    return path


# coarse_gray_depth=0: no binning, so the grey levels are the intensities
nyx = Nyxus3D(["*3D_NGTDM*"], coarse_gray_depth=0)
features = nyx.featurize_files([write("int.ome.tif", vol)], [write("seg.ome.tif", mask)], False)
print(features[["3NGTDM_COARSENESS", "3NGTDM_CONTRAST", "3NGTDM_BUSYNESS"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["ngtdm_3d"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"

_n = [0]


def run(img, seg, ibsi=False, metaparams=(), depth=0):
    _n[0] += 1
    n = Nyxus3D(["*3D_NGTDM*"], ibsi=ibsi, coarse_gray_depth=depth)
    for m in metaparams:
        n.set_metaparam(m)
    return n.featurize_files([write(f"i{_n[0]}.ome.tif", img)], [write(f"s{_n[0]}.ome.tif", seg)], False)


def check(table, quoted, label):
    for name, expected in quoted.items():
        got = table[f"3NGTDM_{name}"][0]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
            f"{label} 3NGTDM_{name} = {got}, page says {expected}"


# IBSI mode: grey levels are lifted by one, which moves busyness only
check(run(vol, mask, ibsi=True), {"COARSENESS": 0.1549, "CONTRAST": 0.1889, "BUSYNESS": 1.502,
                                  "COMPLEXITY": 5.241, "STRENGTH": 0.7111}, "IBSI")

# Busyness depends on the grey levels themselves: every intensity shifted by 9
check(run(vol + 9, mask), {"BUSYNESS": 0.9682, "COARSENESS": 0.1549}, "shifted")
check(run(vol + 9, mask, ibsi=True), {"BUSYNESS": 0.8984}, "shifted IBSI")

# The default 64 MATLAB-style levels, from the coarse grey depth or from 3ngtdm/greydepth=64
grey64 = {"COARSENESS": 0.009831, "CONTRAST": 734.1, "BUSYNESS": 1.485, "COMPLEXITY": 1294.3,
          "STRENGTH": 11.13}
check(Nyxus3D(["*3D_NGTDM*"]).featurize_files(
    [write("d_i.ome.tif", vol)], [write("d_s.ome.tif", mask)], False), grey64, "default")
check(run(vol, mask, metaparams=["3ngtdm/greydepth=64"]), grey64, "greydepth=64")

# coarse_gray_depth reaches the family as 3ngtdm/greydepth does
grey8 = {"COARSENESS": 0.08807, "CONTRAST": 0.9837, "BUSYNESS": 1.518, "COMPLEXITY": 15.52,
         "STRENGTH": 1.198}
check(run(vol, mask, depth=8), grey8, "coarse_gray_depth=8")
check(run(vol, mask, metaparams=["3ngtdm/greydepth=8"]), grey8, "greydepth=8")

# Four PyRadiomics-style bins are the four intensities, so they give the unbinned values
check(run(vol, mask, depth=-4), {code[len("3NGTDM_"):]: facts["example"] for code, facts in extras.items()},
      "coarse_gray_depth=-4")

# 3ngtdm/radius=2
check(run(vol, mask, metaparams=["3ngtdm/radius=2"]),
      {"COARSENESS": 0.1583, "CONTRAST": 0.1860, "BUSYNESS": 1.445, "COMPLEXITY": 5.173,
       "STRENGTH": 0.7222}, "radius=2")

# The eight corner voxels left out of the mask: neither centres nor neighbours, as in PyRadiomics;
# IBSI mode moves busyness only
corners = np.ones_like(vol)
corners[::2, ::2, ::2] = 0
check(run(vol, corners),
      {"COARSENESS": 0.1843, "CONTRAST": 0.2407, "BUSYNESS": 1.097, "COMPLEXITY": 6.190,
       "STRENGTH": 0.9691}, "corners off")
check(run(vol, corners, ibsi=True),
      {"COARSENESS": 0.1843, "CONTRAST": 0.2407, "BUSYNESS": 1.121, "COMPLEXITY": 6.190,
       "STRENGTH": 0.9691}, "corners off (IBSI)")
check(run(vol, corners, depth=64),
      {"COARSENESS": 0.01173, "CONTRAST": 932.4, "BUSYNESS": 1.101, "COMPLEXITY": 1525.0,
       "STRENGTH": 15.18}, "corners off (64 levels)")

# A constant ROI: every feature is 0 in Python (the substitute in every binning mode but IBSI),
# whether or not it fills its box
const = np.full((3, 3, 3), 7, dtype=np.uint16)
for ibsi in (False, True):
    for seg, label in ((mask, "constant"), (corners, "constant, corners off")):
        check(run(const, seg, ibsi=ibsi),
              {"COARSENESS": 0.0, "CONTRAST": 0.0, "BUSYNESS": 0.0, "COMPLEXITY": 0.0, "STRENGTH": 0.0},
              f"{label} (ibsi={ibsi})")
for depth in (64, -8):
    check(run(const, mask, depth=depth),
          {"COARSENESS": 0.0, "CONTRAST": 0.0, "BUSYNESS": 0.0, "COMPLEXITY": 0.0, "STRENGTH": 0.0},
          f"constant (greydepth={depth})")

# A single-voxel ROI
single = np.zeros_like(vol)
single[1, 1, 1] = 1
check(run(const, single), {"COARSENESS": 0.0, "CONTRAST": 0.0, "BUSYNESS": 0.0, "COMPLEXITY": 0.0,
                           "STRENGTH": 0.0}, "single voxel")
