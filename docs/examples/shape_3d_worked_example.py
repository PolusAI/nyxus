# Example for docs/source/Math/f_morphology_3d.rst: 3D shape features of the page's 44-voxel ROI.
# The page shows the lines between the "example" markers. The assertions below pin every Nyxus
# value the page quotes, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# The ROI, slice by slice (z = 0, 1, 2); rows are y, columns are x
roi = np.array([[[0, 1, 1, 1, 0, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 0]],
                [[0, 0, 1, 1, 0, 0],
                 [0, 1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 0]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=np.uint16)
seg = np.pad(roi, 1)                                   # 5 x 7 x 8 volume, ROI label 1
img = (np.arange(seg.size) % 7 + 1).astype(np.uint16).reshape(seg.shape)  # any values

tifffile.imwrite("int.ome.tif", img, ome=True, metadata={"axes": "ZYX"})
tifffile.imwrite("seg.ome.tif", seg, ome=True, metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_ALL_MORPHOLOGY*"])
features = nyx.featurize_files(["./int.ome.tif"], ["./seg.ome.tif"], False)
print(features.filter(like="3").T)
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["shape_3d"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code} = {got}, page says {facts['example']}"

f = {c: float(features[c][0]) for c in extras}
PACK = (math.pi / 6) / 0.5236                          # the voxel-volume factor, 0.9999976...


def box_area(w, h, d):
    """Mesh area of a solid w x h x d box of unit voxels: the box with its edges and corners bevelled."""
    return 2 * (w * h + h * d + w * d) - 2 * (w + h + d - 3) * (2 - math.sqrt(2)) - (6 - math.sqrt(3))


# Values the page quotes for the worked example
assert math.isclose(f["3AREA"], 66.365538, rel_tol=1e-7)
assert math.isclose(f["3VOXEL_VOLUME"], 44 * PACK, rel_tol=1e-14)
assert math.isclose(f["3MESH_VOLUME"], 475 / 12, rel_tol=1e-14)
assert math.isclose(f["3VOLUME_CONVEXHULL"], 122 / 3, rel_tol=1e-14)
z, y, x = np.nonzero(roi)
lam = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([x, y, z]).astype(float))))[::-1]
assert np.allclose(lam, [2.0920, 0.94139, 0.52260], rtol=1e-4)
for code, k in (("3MAJOR_AXIS_LEN", 0), ("3MINOR_AXIS_LEN", 1), ("3LEAST_AXIS_LEN", 2)):
    assert math.isclose(f[code], 4 * math.sqrt(lam[k]), rel_tol=1e-12)
A, V = f["3AREA"], f["3MESH_VOLUME"]
assert math.isclose(f["3AREA_2_VOLUME"], A / V, rel_tol=1e-14)
assert math.isclose(f["3COMPACTNESS1"], V / math.sqrt(math.pi * A ** 3), rel_tol=1e-14)
assert math.isclose(f["3COMPACTNESS2"], 36 * math.pi * V ** 2 / A ** 3, rel_tol=1e-14)
assert math.isclose(f["3SPHERICITY"], (36 * math.pi * V ** 2) ** (1 / 3) / A, rel_tol=1e-14)
assert math.isclose(f["3SPHERICITY"] * f["3SPHERICAL_DISPROPORTION"], 1.0, rel_tol=1e-14)


def shape3d(vol, single_roi=False, **kw):
    """3D shape features of label image `vol` (or, with single_roi, of the whole volume)."""
    tifffile.imwrite("s_int.ome.tif", (vol > 0).astype(np.uint16) + 1, ome=True, metadata={"axes": "ZYX"})
    tifffile.imwrite("s_seg.ome.tif", vol.astype(np.uint16), ome=True, metadata={"axes": "ZYX"})
    segs = ["./s_int.ome.tif"] if single_roi else ["./s_seg.ome.tif"]
    r = Nyxus3D(["*3D_ALL_MORPHOLOGY*"], **kw).featurize_files(["./s_int.ome.tif"], segs, single_roi)
    return {c: float(r[c][0]) for c in extras}


# A single voxel: the 0.5-isolevel surface is an octahedron of area sqrt(3) and volume 1/6; no axes
one = shape3d(np.pad(np.ones((1, 1, 1), np.uint16), 2))
assert math.isclose(one["3AREA"], math.sqrt(3), rel_tol=1e-14)
assert math.isclose(one["3MESH_VOLUME"], 1 / 6, rel_tol=1e-14)
assert math.isclose(one["3VOLUME_CONVEXHULL"], 1 / 6, rel_tol=1e-14)
assert one["3MAJOR_AXIS_LEN"] == one["3ELONGATION"] == one["3FLATNESS"] == 0

# A one-slice ROI (slice z = 0 of the example): LEAST_AXIS_LEN and FLATNESS are 0
flat = shape3d(np.pad(roi[:1], 1))
assert flat["3LEAST_AXIS_LEN"] == 0 and flat["3FLATNESS"] == 0
assert math.isclose(flat["3MESH_VOLUME"], 103 / 6, rel_tol=1e-14)

# A digital ball of radius 10: the mesh is a faceted sphere, with a sphericity below 1
g = np.mgrid[-10:11, -10:11, -10:11]
ball = shape3d(np.pad((np.sum(g ** 2, axis=0) <= 100).astype(np.uint16), 1))
assert math.isclose(ball["3AREA"], 1372.04, rel_tol=5e-6)
assert math.isclose(ball["3SPHERICITY"], 0.9099, rel_tol=5e-4)
assert math.isclose(ball["3MESH_VOLUME"], 4147.8333, rel_tol=1e-7)
assert math.isclose(ball["3ELONGATION"], 1.0, rel_tol=1e-12)

# A 6 x 4 x 5 box filling its volume: segmented ROI vs whole-volume (single-ROI) mode
box = np.ones((5, 4, 6), np.uint16)
seg_box, whole = shape3d(box), shape3d(box, single_roi=True)
# pins known defect https://github.com/PolusAI/nyxus/issues/494
assert math.isclose(seg_box["3MESH_VOLUME"], 120 - 12 / 2 - 5 / 6, rel_tol=1e-14)
assert whole["3MESH_VOLUME"] == seg_box["3MESH_VOLUME"]
for c in ("3AREA", "3SPHERICITY"):
    assert math.isclose(whole[c], seg_box[c], rel_tol=1e-14), c
assert math.isclose(seg_box["3AREA"], box_area(6, 4, 5), rel_tol=1e-14)        # 129.673
assert whole["3VOXEL_VOLUME"] == 120 and math.isclose(seg_box["3VOXEL_VOLUME"], 120 * PACK, rel_tol=1e-14)
assert math.isclose(seg_box["3MAJOR_AXIS_LEN"], 6.8599, rel_tol=5e-4)
assert whole["3MAJOR_AXIS_LEN"] == whole["3MINOR_AXIS_LEN"] == whole["3LEAST_AXIS_LEN"] == 0
assert whole["3ELONGATION"] == whole["3FLATNESS"] == 0

# Anisotropy: the voxels as acquired, with the coordinates scaled; anisotropy_z=2 doubles the volumes
aniso = shape3d(seg, anisotropy_z=2.0)
for c in ("3VOXEL_VOLUME", "3MESH_VOLUME", "3VOLUME_CONVEXHULL"):
    assert math.isclose(aniso[c], 2 * f[c], rel_tol=1e-12), c
# The area does not double: each triangle is stretched along z only
assert math.isclose(aniso["3AREA"], 103.43832, rel_tol=1e-7)
lam_z2 = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([x, y, 2.0 * z]).astype(float))))[::-1]
for code, k in (("3MAJOR_AXIS_LEN", 0), ("3MINOR_AXIS_LEN", 1), ("3LEAST_AXIS_LEN", 2)):
    assert math.isclose(aniso[code], 4 * math.sqrt(lam_z2[k]), rel_tol=1e-12), code
# The same in the first image plane: every plane is measured once, at its scaled position
assert math.isclose(shape3d(seg, anisotropy_z=3.0)["3VOXEL_VOLUME"], 3 * 44 * PACK, rel_tol=1e-14)
first = np.pad(roi, ((0, 1), (1, 1), (1, 1)))           # slice z = 0 of the ROI in image plane 0
assert math.isclose(shape3d(first, anisotropy_z=3.0)["3VOXEL_VOLUME"], 3 * 44 * PACK, rel_tol=1e-14)

# Whole-volume mode on the command line or through featurize_directory: a box one voxel larger on
# every axis, 7 x 5 x 6 for the 6 x 4 x 5 volume
os.makedirs("whole_dir", exist_ok=True)
tifffile.imwrite("whole_dir/v.ome.tif", np.full((5, 4, 6), 7, np.uint16), ome=True,
                 metadata={"axes": "ZYX"})
r = Nyxus3D(["*3D_ALL_MORPHOLOGY*"]).featurize_directory("whole_dir", "whole_dir", ".*")
# pins known defect https://github.com/PolusAI/nyxus/issues/493
assert r["3VOXEL_VOLUME"][0] == 210
assert math.isclose(r["3AREA"][0], box_area(7, 5, 6), rel_tol=1e-14)           # 192.158
assert math.isclose(r["3MESH_VOLUME"][0], 210 - 15 / 2 - 5 / 6, rel_tol=1e-14)
assert r["3MAJOR_AXIS_LEN"][0] == 0

# featurize_files(..., single_roi=True) skips zero voxels: the box is that of the nonzero voxels
part = np.zeros((5, 4, 6), np.uint16)
part[1:3, 1:3, 1:4] = 7                                 # a 3 x 2 x 2 block
tifffile.imwrite("part.ome.tif", part, ome=True, metadata={"axes": "ZYX"})
r = Nyxus3D(["*3D_ALL_MORPHOLOGY*"]).featurize_files(["./part.ome.tif"], ["./part.ome.tif"], True)
assert r["3VOXEL_VOLUME"][0] == 12
assert math.isclose(r["3AREA"][0], box_area(3, 2, 2), rel_tol=1e-14)           # 23.046

# single_roi=True measures that box whatever the mask's shape: a radius-5 ball (515 voxels) in a
# 15 x 15 x 15 volume, as a segmented ROI and as one ROI, which gives its 11 x 11 x 11 box
g5 = np.indices((15, 15, 15)) - 7
ball5 = (np.sum(g5 ** 2, axis=0) <= 25).astype(np.uint16)
tifffile.imwrite("ball5.ome.tif", 7 * ball5, ome=True, metadata={"axes": "ZYX"})
tifffile.imwrite("ball5_mask.ome.tif", ball5, ome=True, metadata={"axes": "ZYX"})
codes = ("3VOXEL_VOLUME", "3AREA", "3MAJOR_AXIS_LEN")
for single, expected in ((False, (514.999, 340.08, 8.909)),
                         # pins known defect https://github.com/PolusAI/nyxus/issues/494
                         (True, (1331, 686.58, 0))):
    r = Nyxus3D(["*3D_ALL_MORPHOLOGY*"]).featurize_files(["./ball5.ome.tif"], ["./ball5_mask.ome.tif"],
                                                         single)
    for code, value in zip(codes, expected):
        assert math.isclose(r[code][0], value, rel_tol=5e-4, abs_tol=1e-12), \
            f"single_roi={single} {code} = {r[code][0]}, page says {value}"

# A cavity is subtracted from the mesh volume
cube = np.pad(np.ones((5, 5, 5), np.uint16), 1)
solid = shape3d(cube)
cube[3, 3, 3] = 0
assert math.isclose(solid["3MESH_VOLUME"], 125 - 12 / 2 - 5 / 6, rel_tol=1e-14)   # 118.167
assert math.isclose(solid["3AREA"], box_area(5, 5, 5), rel_tol=1e-14)          # 131.673
hollow = shape3d(cube)
assert math.isclose(hollow["3MESH_VOLUME"], 118.0, rel_tol=1e-12)
# The cavity's octahedron adds its area sqrt(3)
assert math.isclose(hollow["3AREA"], solid["3AREA"] + math.sqrt(3), rel_tol=1e-14)

# The digital ball scores as more compact than a solid cube
for code in ("3COMPACTNESS1", "3COMPACTNESS2", "3SPHERICITY"):
    assert ball[code] > solid[code], code
assert ball["3SPHERICAL_DISPROPORTION"] < solid["3SPHERICAL_DISPROPORTION"]
assert math.isclose(solid["3SPHERICAL_DISPROPORTION"], 1.131, rel_tol=5e-4)
assert math.isclose(ball["3SPHERICAL_DISPROPORTION"], 1.099, rel_tol=5e-4)
for vals, expected in ((solid, (0.04412, 0.6918, 0.8844)), (ball, (0.04605, 0.7533, 0.9099))):
    for code, value in zip(("3COMPACTNESS1", "3COMPACTNESS2", "3SPHERICITY"), expected):
        assert math.isclose(vals[code], value, rel_tol=5e-4), f"{code} = {vals[code]}, page says {value}"
assert math.isclose(ball["3AREA_2_VOLUME"], 0.3308, rel_tol=5e-4)
