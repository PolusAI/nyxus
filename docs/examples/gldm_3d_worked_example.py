# Example for docs/source/Math/f_gldm_3d.rst: 3D GLDM features of the page's 3 x 4 x 4 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, in IBSI mode, with the default settings and with the 3gldm/greydepth metaparameter,
# so the page cannot drift from the code. The script writes its OME-TIFF files into the current
# directory.
import json
import math
import os

# example-start
from pathlib import Path

import numpy as np
import tifffile
from nyxus import Nyxus3D

# A 3 x 4 x 4 ROI (slices, rows, columns) with grey levels 1 to 4, inside a 3 x 6 x 6 volume
roi = np.array([[[1, 1, 2, 3], [1, 2, 2, 3], [4, 4, 2, 1], [4, 3, 3, 1]],
                [[1, 2, 2, 3], [1, 1, 3, 3], [4, 2, 2, 1], [4, 4, 3, 3]],
                [[2, 2, 3, 3], [1, 1, 3, 4], [4, 4, 1, 1], [2, 4, 4, 3]]], dtype=np.uint16)
img = np.zeros((3, 6, 6), dtype=np.uint16)
img[:, 1:5, 1:5] = roi
seg = (img > 0).astype(np.uint16)

# Nyxus3D reads volumes from files: OME-TIFF, slices along Z
int_file, seg_file = str(Path("int.ome.tif").resolve()), str(Path("seg.ome.tif").resolve())
tifffile.imwrite(int_file, img, ome=True, metadata={"axes": "ZYX"})
tifffile.imwrite(seg_file, seg, ome=True, metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_GLDM*"], ibsi=True)
features = nyx.featurize_files([int_file], [seg_file], False)
print(features[["3GLDM_SDE", "3GLDM_LDE", "3GLDM_DN", "3GLDM_GLV"]])
# example-end


def run(volume, mask, name, metaparam=None, **kwargs):
    """Featurizes one intensity/mask pair written under its own file names."""
    ip, sp = str(Path(f"{name}_int.ome.tif").resolve()), str(Path(f"{name}_seg.ome.tif").resolve())
    tifffile.imwrite(ip, volume.astype(np.uint16), ome=True, metadata={"axes": "ZYX"})
    tifffile.imwrite(sp, mask.astype(np.uint16), ome=True, metadata={"axes": "ZYX"})
    n = Nyxus3D(["*3D_GLDM*"], **kwargs)
    if metaparam:
        n.set_metaparam(metaparam)
    return n.featurize_files([ip], [sp], False)


def check(table, quoted, what):
    for name, expected in quoted.items():
        got = table[f"3GLDM_{name}"][0]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
            f"{what} 3GLDM_{name} = {got}, page says {expected}"


def voxels(table):
    """N_z, the number of voxels in the matrix, recovered as DN / DNN."""
    return table["3GLDM_DN"][0] / table["3GLDM_DNN"][0]


# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["gldm_3d"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"
assert round(voxels(features)) == 48

# The default settings (3gldm/greydepth = 0) give the IBSI values; --coarseGrayDepth has no effect
for table in (run(img, seg, "default"), run(img, seg, "coarse", coarse_gray_depth=8)):
    for code in extras:
        assert math.isclose(table[code][0], features[code][0], rel_tol=1e-12), f"default {code} differs from IBSI"

# Intensities times 250: no binning by default, so the grey-level features scale; 64 levels rescale
bright = np.where(seg > 0, img * 250, 0)
check(run(bright, seg, "bright"), {"SDE": 0.06470, "LDE": 36.54, "HGLE": 455729.2}, "x250 default")
check(run(bright, seg, "bright64", "3gldm/greydepth=64"), {"SDE": 0.06470, "HGLE": 1916.8}, "x250 greydepth=64")

# 3gldm/greydepth = -2: two PyRadiomics-style bins
check(run(img, seg, "radiomics2", "3gldm/greydepth=-2"),
      {"SDE": 0.02258, "LDE": 87.21, "GLN": 24.0, "GLV": 0.25}, "greydepth=-2")

# 3gldm/greydepth = 2: the 13 voxels of intensity 1 fall in level 1 and leave the matrix
matlab2 = run(img, seg, "matlab2", "3gldm/greydepth=2")
check(matlab2, {"SDE": 0.01297, "LDE": 146.2, "GLN": 35.0, "GLV": 0.0}, "greydepth=2")
assert round(voxels(matlab2)) == 35
probe = Nyxus3D(["*3D_GLDM*"])
assert probe.get_metaparam("3gldm/greydepth") == 0
probe.set_metaparam("3gldm/greydepth=2")
assert probe.get_metaparam("3gldm/greydepth") == 2
# ... and with greydepth = 1 every voxel does, so every feature is the no-value substitute (0)
check(run(img, seg, "matlab1", "3gldm/greydepth=1"), {"SDE": 0.0, "LDE": 0.0, "GLN": 0.0}, "greydepth=1")
# IBSI mode ignores the metaparameter
check(run(img, seg, "ibsi2", "3gldm/greydepth=2", ibsi=True), {"SDE": 0.06470, "LDE": 36.54}, "IBSI + greydepth=2")

# Three voxels of the bounding box left out of the mask
holes = seg.copy()
holes[0, 1, 1] = holes[2, 4, 4] = holes[1, 2, 3] = 0
masked = run(img, holes, "holes", ibsi=True)
check(masked, {"SDE": 0.07001, "LDE": 32.87}, "masked")
assert round(voxels(masked)) == 45

# The same three voxels given to a second ROI: ROI 1 is unchanged, so other ROIs' voxels never count
two = seg.copy()
two[0, 1, 1] = two[2, 4, 4] = two[1, 2, 3] = 2
check(run(img, two, "two", ibsi=True), {"SDE": 0.07001, "LDE": 32.87}, "two-ROI")

# A voxel of intensity 0 inside the mask
zero = img.copy()
zero[1, 2, 2] = 0
zeroed = run(zero, seg, "zero", ibsi=True)
check(zeroed, {"SDE": 0.06801, "LDE": 33.64}, "zero-voxel")
assert round(voxels(zeroed)) == 47

# Two isolated voxels: each is counted with dependence 1
iso = np.zeros_like(seg)
iso[0, 1, 1] = iso[2, 4, 4] = 1
check(run(img, iso, "isolated", ibsi=True), {"SDE": 1.0, "LDE": 1.0, "HGLE": 5.0}, "isolated")

# A constant ROI returns the no-value substitute (0) for every feature
check(run(np.where(seg > 0, 7, 0), seg, "constant", ibsi=True),
      {"SDE": 0.0, "LDE": 0.0, "GLN": 0.0, "DN": 0.0}, "constant")

# Out-of-core (a zero RAM budget) gives the in-memory values, with the metaparameter too
check(run(img, seg, "ooc", ram_limit=0), {"SDE": 0.06470, "LDE": 36.54}, "out-of-core")
check(run(img, seg, "ooc2", "3gldm/greydepth=2", ram_limit=0), {"SDE": 0.01297, "LDE": 146.2}, "out-of-core greydepth=2")
