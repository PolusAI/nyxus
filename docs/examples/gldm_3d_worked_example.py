# Example for docs/source/Math/f_gldm_3d.rst: 3D GLDM features of the page's 3 x 4 x 4 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, in IBSI mode, with the default settings, with --coarseGrayDepth and with the
# 3gldm/greydepth metaparameter, so the page cannot drift from the code. The script writes its
# OME-TIFF files into the current directory.
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

# --coarseGrayDepth=0, and 3gldm/greydepth=0 over the default, give the IBSI values
for table in (run(img, seg, "nobin", coarse_gray_depth=0), run(img, seg, "meta0", "3gldm/greydepth=0")):
    for code in extras:
        assert math.isclose(table[code][0], features[code][0], rel_tol=1e-12), f"unbinned {code} differs from IBSI"

# The default 64 levels put intensities 1 to 4 at levels 17, 33, 49 and 64: the same counts, so the
# dependence features keep their IBSI values and the grey-level features move
default = run(img, seg, "default")
check(default, {"SDE": 0.06470, "LDE": 36.54, "GLN": 12.08, "GLV": 308.4, "HGLE": 1916.8}, "default")
assert round(voxels(default)) == 48
probe = Nyxus3D(["*3D_GLDM*"])
assert probe.get_metaparam("3gldm/greydepth") == 64
# --coarseGrayDepth reaches the family, and the metaparameter starts at it
check(run(img, seg, "coarse8", coarse_gray_depth=8), {"SDE": 0.06470, "HGLE": 36.10}, "coarse_gray_depth=8")
assert Nyxus3D(["*3D_GLDM*"], coarse_gray_depth=8).get_metaparam("3gldm/greydepth") == 8

# Intensities times 250: the 64 levels do not follow the scale; without binning the grey-level
# features scale with the intensities
bright = np.where(seg > 0, img * 250, 0)
check(run(bright, seg, "bright"), {"SDE": 0.06470, "LDE": 36.54, "HGLE": 1916.8}, "x250 default")
check(run(bright, seg, "bright0", coarse_gray_depth=0), {"SDE": 0.06470, "HGLE": 455729.2}, "x250 no binning")

# 3gldm/greydepth = -2: two PyRadiomics-style bins
check(run(img, seg, "radiomics2", "3gldm/greydepth=-2"),
      {"SDE": 0.02258, "LDE": 87.21, "GLN": 24.0, "GLV": 0.25}, "greydepth=-2")

# 3gldm/greydepth = 2: intensity 1 is level 1 and intensities 2 to 4 are level 2; every voxel counts
matlab2 = run(img, seg, "matlab2", "3gldm/greydepth=2")
check(matlab2, {"SDE": 0.02213, "LDE": 116.1, "GLN": 29.04, "GLV": 0.1975}, "greydepth=2")
assert round(voxels(matlab2)) == 48
probe.set_metaparam("3gldm/greydepth=2")
assert probe.get_metaparam("3gldm/greydepth") == 2
# ... and with greydepth = 1 every voxel is level 1
check(run(img, seg, "matlab1", "3gldm/greydepth=1"), {"SDE": 0.006641, "GLN": 48.0, "GLV": 0.0}, "greydepth=1")
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

# A voxel of intensity 0 inside the mask: background without binning, a ROI voxel with it
zero = img.copy()
zero[1, 2, 2] = 0
zeroed = run(zero, seg, "zero", ibsi=True)
check(zeroed, {"SDE": 0.06801, "LDE": 33.64}, "zero-voxel")
assert round(voxels(zeroed)) == 47
zero_binned = run(zero, seg, "zero64")
check(zero_binned, {"SDE": 0.08743, "LDE": 32.96}, "zero-voxel default")
assert round(voxels(zero_binned)) == 48

# Two isolated voxels: each is counted with dependence 1
iso = np.zeros_like(seg)
iso[0, 1, 1] = iso[2, 4, 4] = 1
check(run(img, iso, "isolated", ibsi=True), {"SDE": 1.0, "LDE": 1.0, "HGLE": 5.0}, "isolated")

# A constant ROI returns the no-value substitute (0) for every feature
check(run(np.where(seg > 0, 7, 0), seg, "constant", ibsi=True),
      {"SDE": 0.0, "LDE": 0.0, "GLN": 0.0, "DN": 0.0}, "constant")

# Out-of-core (a zero RAM budget) gives the in-memory values, with the metaparameter too
check(run(img, seg, "ooc", ram_limit=0), {"SDE": 0.06470, "LDE": 36.54, "HGLE": 1916.8}, "out-of-core")
check(run(img, seg, "ooc2", "3gldm/greydepth=2", ram_limit=0), {"SDE": 0.02213, "LDE": 116.1}, "out-of-core greydepth=2")
check(run(zero, seg, "ooc_zero", ram_limit=0), {"SDE": 0.08743, "LDE": 32.96}, "out-of-core zero-voxel")
