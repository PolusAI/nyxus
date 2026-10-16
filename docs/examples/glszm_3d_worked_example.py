# Example for docs/source/Math/f_glszm_3d.rst: 3D GLSZM features of the page's 3 x 3 x 3 worked
# example. The page shows the lines between the "example" markers. The assertions below pin every
# value the page quotes, except the results of the 32-bit overflow under "Very large zones", so the
# page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# Three 3 x 3 slices, indexed (z, y, x), with grey levels 1 to 4
vol = np.array([[[3, 3, 4],
                 [2, 2, 4],
                 [4, 3, 1]],
                [[4, 2, 1],
                 [3, 2, 2],
                 [1, 1, 3]],
                [[1, 2, 3],
                 [4, 4, 4],
                 [4, 3, 4]]], dtype=np.uint16)
mask = np.ones_like(vol)

tifffile.imwrite("int.ome.tif", vol, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
tifffile.imwrite("seg.ome.tif", mask, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_GLSZM*"], ibsi=True)
features = nyx.featurize_files([str(Path("int.ome.tif").resolve())],
                               [str(Path("seg.ome.tif").resolve())], False)
print(features[["3GLSZM_SAE", "3GLSZM_LAE", "3GLSZM_ZP", "3GLSZM_ZE"]])
# example-end

CODES = ["SAE", "LAE", "GLN", "GLNN", "SZN", "SZNN", "ZP", "GLV", "ZV", "ZE",
         "LGLZE", "HGLZE", "SALGLE", "SAHGLE", "LALGLE", "LAHGLE"]

# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["glszm_3d"]
assert len(extras) == 16, f"the page's index has {len(extras)} codes, the family has 16"
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"


_run = [0]


def glszm3d(v, m, meta=None, **kw):
    """All sixteen 3GLSZM values of one volume and mask, as {code suffix: value}."""
    _run[0] += 1
    iname, sname = Path(f"int{_run[0]}.ome.tif"), Path(f"seg{_run[0]}.ome.tif")
    tifffile.imwrite(iname, v.astype(np.uint16), ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    tifffile.imwrite(sname, m.astype(np.uint16), ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    n = Nyxus3D(["*3D_GLSZM*"], **kw)
    if meta:
        n.set_metaparam(meta)
    f = n.featurize_files([str(iname.resolve())], [str(sname.resolve())], False)
    return {c: float(f[f"3GLSZM_{c}"][0]) for c in CODES}


def same(a, b, rel=1e-12):
    return all(math.isclose(a[c], b[c], rel_tol=rel, abs_tol=1e-15) for c in CODES)


# The page's matrix P: 9 zones over 27 voxels, and zone entropy against an exact logarithm
P = np.array([[2, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 1],
              [1, 1, 0, 0, 0, 1]], dtype=float)
assert P.sum() == 9 and math.isclose(features["3GLSZM_ZP"][0], 9 / 27), "zone count differs from the page's P"
p = P[P > 0] / P.sum()
exact = float(-(p * np.log2(p)).sum())
assert math.isclose(exact, 2.9477, rel_tol=5e-5), f"exact zone entropy = {exact}, page says 2.9477"
assert math.isclose(features["3GLSZM_ZE"][0], 2.9485, rel_tol=5e-5), "3GLSZM_ZE differs from the page's 2.9485"

# 26-connectivity: the level-3 voxels at (z, y, x) = (1, 1, 0) and (0, 2, 1) touch only at a corner,
# and they join two groups of three into one zone of 6. 18- or 6-connectivity would split it.
ref = glszm3d(vol, mask, ibsi=True)
assert same(ref, {c: features[f"3GLSZM_{c}"][0] for c in CODES})
assert vol[1, 1, 0] == vol[0, 2, 1] == 3 and P[2, 5] == 1, "the corner-joined zone of level 3 and size 6"


def zone_count(v, max_axes):
    """Zones of equal level, joining voxels whose offset changes at most `max_axes` axes by 1."""
    offsets = [d for d in np.ndindex(3, 3, 3) if 0 < sum(abs(c - 1) for c in d) <= max_axes]
    seen, zones = np.zeros(v.shape, bool), 0
    for start in np.ndindex(v.shape):
        if seen[start]:
            continue
        zones += 1
        seen[start], stack = True, [start]
        while stack:
            z, y, x = stack.pop()
            for dz, dy, dx in offsets:
                q = (z + dz - 1, y + dy - 1, x + dx - 1)
                if all(0 <= q[i] < v.shape[i] for i in range(3)) and not seen[q] and v[q] == v[z, y, x]:
                    seen[q] = True
                    stack.append(q)
    return zones


assert [zone_count(vol, k) for k in (3, 2, 1)] == [9, 10, 15], "26-, 18- and 6-connected zone counts"

# Default settings: no binning, the same values as IBSI mode; --coarseGrayDepth has no effect
assert same(glszm3d(vol, mask), ref), "default settings differ from IBSI mode"
assert same(glszm3d(vol, mask, coarse_gray_depth=8), ref), "coarse_gray_depth changed 3GLSZM"
assert same(glszm3d(vol, mask, ibsi=True, meta="3glszm/greydepth=2"), ref), "IBSI mode did not ignore greydepth"

# Out of core: the same values
assert same(glszm3d(vol, mask, ibsi=True, ram_limit=0), ref), "out-of-core values differ"

# PyRadiomics-style binning, 3 bins: levels 1, 2, 3
f = glszm3d(vol, mask, meta="3glszm/greydepth=-3")
quoted = {"ZP": 0.1852, "SAE": 0.4286, "GLV": 0.64}
for name, expected in quoted.items():
    assert math.isclose(f[name], expected, rel_tol=5e-4), f"greydepth=-3 3GLSZM_{name} = {f[name]}, page says {expected}"

# MATLAB-style binning, 2 levels: the five voxels of intensity 1 bin to level 1 and form no zone
f = glszm3d(vol, mask, meta="3glszm/greydepth=2")
quoted = {"ZP": 1 / 27, "SAE": 0.002066, "LAE": 484.0, "LGLZE": 0.25}
for name, expected in quoted.items():
    assert math.isclose(f[name], expected, rel_tol=5e-4), f"greydepth=2 3GLSZM_{name} = {f[name]}, page says {expected}"

# A ROI that does not fill its bounding box: outside voxels belong to no zone and are not in Np
part = mask.copy()
part[0, 0, 0] = part[1, 1, 1] = part[2, 2, 2] = 0
f = glszm3d(vol, part, ibsi=True)
assert math.isclose(f["ZP"], 9 / 24), f"partial ROI 3GLSZM_ZP = {f['ZP']}, page says 0.375"
assert same(glszm3d(vol, part, ibsi=True, ram_limit=0), f), "partial ROI out-of-core values differ"

# Voxels of intensity 0 inside the mask: no zone, but counted in Np
zeros = vol.copy()
zeros[0, 0, 0] = zeros[1, 1, 1] = 0
f = glszm3d(zeros, mask, ibsi=True)
assert math.isclose(f["ZP"], 9 / 27), f"in-mask zeros 3GLSZM_ZP = {f['ZP']}, page says 0.3333"

# Constant ROI: every feature is the missing-value substitute (0 by default)
const = np.full((2, 2, 2), 7, dtype=np.uint16)
f = glszm3d(const, np.ones_like(const), ibsi=True)
assert all(v == 0 for v in f.values()), "constant ROI is not the substitute"

# Whole-volume mode (no mask): 3glszm/greydepth has no effect
wv = Path("wholevolume")
wv.mkdir()
tifffile.imwrite(wv / "vol.ome.tif", vol, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
n = Nyxus3D(["*3D_GLSZM*"])
n.set_metaparam("3glszm/greydepth=2")
f = n.featurize_directory(str(wv.resolve()), None, ".*")
assert math.isclose(f["3GLSZM_ZP"][0], 9 / 27), "whole-volume mode applied 3glszm/greydepth"

# Reading the metaparameter back fails, although setting it works
try:
    n.get_metaparam("3glszm/greydepth")
except Exception as e:
    assert "unrecognized" in str(e).lower(), f"get_metaparam failed differently: {e}"
else:
    raise AssertionError("get_metaparam('3glszm/greydepth') no longer fails")
