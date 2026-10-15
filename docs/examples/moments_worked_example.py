# Example for docs/source/Math/f_2dmoments.rst: shape and intensity moments of the page's 7x8 worked
# example. The page shows the lines between the "example" markers. The assertions below pin every
# value the page quotes, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

img = np.array([[0, 0, 6, 2, 5, 0, 0, 0],
                [0, 3, 7, 4, 1, 5, 2, 0],
                [4, 1, 2, 6, 3, 7, 4, 1],
                [5, 7, 3, 1, 6, 2, 5, 3],
                [2, 4, 6, 5, 2, 4, 1, 0],
                [0, 6, 1, 3, 7, 5, 0, 0],
                [0, 0, 3, 2, 4, 0, 0, 0]], dtype=np.uint32)
mask = (img > 0).astype(np.uint32)

nyx = Nyxus(["*GEOMOMS*"])
features = nyx.featurize(img, mask)
print(features[["SPAT_MOMENT_00", "SPAT_MOMENT_10", "IMOM_RM_00", "HU_M1", "IMOM_HU1"]])
# example-end


def same(got, expected, rel=5e-4):
    if expected == 0:
        return abs(got) <= 1e-9
    return math.isclose(got, expected, rel_tol=rel)


# Every value in the page's two feature indexes, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))
codes = []
for family in ("shape_moments", "intensity_moments"):
    for code, facts in extras[family].items():
        got = features[code][0]
        assert same(got, facts["example"]), f"{code} = {got}, page says {facts['example']}"
        codes.append(code)
assert len(codes) == 180

# Coordinates are measured from the bounding-box corner: the same ROI placed elsewhere in a larger
# image gives the same 180 values
big_img = np.zeros((14, 16), dtype=np.uint32)
big_img[3:10, 5:13] = img
moved = nyx.featurize(big_img, (big_img > 0).astype(np.uint32))
for code in codes:
    assert math.isclose(moved[code][0], features[code][0], rel_tol=1e-9, abs_tol=1e-9), code

# Values the page quotes for the worked example
assert features["SPAT_MOMENT_00"][0] == 40 and features["IMOM_RM_00"][0] == 150
assert same(features["SPAT_MOMENT_10"][0] / features["SPAT_MOMENT_00"][0], 3.275)
assert same(features["SPAT_MOMENT_01"][0] / features["SPAT_MOMENT_00"][0], 2.925)

# 48 x 40 rectangle: Nyxus, and the same moment with exact distances to the contour as traced
# (one pixel right of and below the ROI's border) and to the ROI's own border pixels
H, W = 40, 48
rect = np.ones((H, W), dtype=np.uint32)
w00 = Nyxus(["WEIGHTED_SPAT_MOMENT_00"]).featurize(rect, rect)["WEIGHTED_SPAT_MOMENT_00"][0]
yy, xx = np.mgrid[0:H, 0:W]
on_border = (xx == 0) | (yy == 0) | (xx == W - 1) | (yy == H - 1)
bx, by = xx[on_border].astype(float), yy[on_border].astype(float)
px, py = xx.ravel().astype(float)[:, None], yy.ravel().astype(float)[:, None]


def weight_sum(cx, cy):
    d = np.sqrt(np.min((px - cx[None, :]) ** 2 + (py - cy[None, :]) ** 2, axis=1))
    return float(np.sum(np.log(d + 0.001)))


assert same(w00, 2498.7), w00
assert same(weight_sum(bx + 1, by + 1), 2422.7)
assert same(weight_sum(bx, by), 1821.3)

# Digital disks: the sum of the weights changes sign between radius 5 and radius 8, and the
# odd-order normalized weighted moments follow it
for r, m00, ncm30 in ((5, -68.94, 0.0), (8, 6.806, 7881.0), (20, 1549.1, 0.007713)):
    n = 2 * r + 3
    yy, xx = np.mgrid[0:n, 0:n]
    disk = (((yy - r - 1) ** 2 + (xx - r - 1) ** 2) <= r * r).astype(np.uint32)
    got = Nyxus(["WEIGHTED_SPAT_MOMENT_00", "WT_NORM_CTR_MOM_30"]).featurize(disk, disk)
    assert same(got["WEIGHTED_SPAT_MOMENT_00"][0], m00), (r, got["WEIGHTED_SPAT_MOMENT_00"][0])
    assert same(got["WT_NORM_CTR_MOM_30"][0], ncm30, rel=2e-3), (r, got["WT_NORM_CTR_MOM_30"][0])

# Intensities times 10
scaled = nyx.featurize(img * 10, mask)
assert same(features["IMOM_HU1"][0], 0.04106) and same(scaled["IMOM_HU1"][0], 0.004106)
assert same(scaled["IMOM_NCM_20"][0] / features["IMOM_NCM_20"][0], 0.1)
assert same(scaled["IMOM_HU2"][0] / features["IMOM_HU2"][0], 0.01)
assert same(scaled["IMOM_NRM_10"][0] / features["IMOM_NRM_10"][0], 10 ** -0.5)

# A ROI whose intensities are all 0, and a single-pixel ROI
zero = nyx.featurize(np.zeros((5, 5), dtype=np.uint32), np.pad(np.ones((3, 3), dtype=np.uint32), 1))
assert all(zero[c][0] == 0 for c in extras["intensity_moments"])
assert zero["SPAT_MOMENT_00"][0] == 9
dot = np.zeros((3, 3), dtype=np.uint32)
dot[1, 1] = 7
single = nyx.featurize(dot, (dot > 0).astype(np.uint32))
assert same(single["WEIGHTED_SPAT_MOMENT_00"][0], -6.908)
assert single["CENTRAL_MOMENT_20"][0] == 0 and single["NORM_CENTRAL_MOMENT_20"][0] == 0

# mu01 and mu10 are rounding residues of order 1e-14
for code in ("CENTRAL_MOMENT_01", "CENTRAL_MOMENT_10", "IMOM_CM_01", "IMOM_CM_10"):
    assert abs(features[code][0]) < 1e-13, (code, features[code][0])

# The 48 x 40 rectangle has 172 border pixels; 85 ROI pixels lie on the traced (shifted) contour
on_traced = (bx + 1 < W) & (by + 1 < H)
assert len(bx) == 172 and on_traced.sum() == 85

# Out of core (ram_limit=0, file input): the weighted moments of a 245-pixel ellipse depend on where
# its bounding box starts; the 118 unweighted moments do not
import tifffile


def from_files(mask_image, **kw):
    tifffile.imwrite("mom_int.tif", (mask_image * 5).astype(np.uint16))
    tifffile.imwrite("mom_seg.tif", mask_image.astype(np.uint16))
    return Nyxus(["*GEOMOMS*"], **kw).featurize_files(
        [os.path.abspath("mom_int.tif")], [os.path.abspath("mom_seg.tif")], False)


def ellipse(n, cx, cy):
    yy, xx = np.mgrid[0:n, 0:n]
    return ((((xx - cx) / 10) ** 2 + ((yy - cy) / 8) ** 2) <= 1).astype(np.uint16)


weighted = [c for c in codes if c.startswith(("WEIGHTED_", "WT_NORM_", "IMOM_W"))]
assert len(weighted) == 62
for n, cx, cy, corner, ooc in ((30, 12, 10, (2, 2), 7.484), (26, 13, 13, (3, 5), 58.50),
                               (40, 20, 20, (10, 12), 338.2)):
    e = ellipse(n, cx, cy)
    ys, xs = np.nonzero(e)
    assert e.sum() == 245 and (xs.min(), ys.min()) == corner
    mem, out = from_files(e), from_files(e, ram_limit=0)
    assert same(mem["WEIGHTED_SPAT_MOMENT_00"][0], 49.81)
    assert same(out["WEIGHTED_SPAT_MOMENT_00"][0], ooc), (corner, out["WEIGHTED_SPAT_MOMENT_00"][0])
    for code in codes:
        if code not in weighted:
            assert math.isclose(out[code][0], mem[code][0], rel_tol=1e-9, abs_tol=1e-9), code
# At the image origin: the same ellipse still differs, the rectangle does not
e = ellipse(30, 12, 10)[2:, 2:]
assert same(from_files(e, ram_limit=0)["WEIGHTED_SPAT_MOMENT_00"][0], 43.75)
corner_rect = np.zeros((50, 60), dtype=np.uint16)
corner_rect[:40, :48] = 1
mem, out = from_files(corner_rect), from_files(corner_rect, ram_limit=0)
for code in codes:
    assert math.isclose(out[code][0], mem[code][0], rel_tol=1e-9, abs_tol=1e-9), code
