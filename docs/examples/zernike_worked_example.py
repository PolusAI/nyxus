# Example for docs/source/Math/f_zernike.rst: Zernike moments of the page's 5x5 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

img = np.array([[5, 2, 5, 4, 4],
                [3, 3, 3, 1, 3],
                [2, 1, 1, 1, 3],
                [4, 2, 2, 2, 3],
                [3, 5, 3, 3, 2]], dtype=np.uint32)
mask = np.ones_like(img)

nyx = Nyxus(["ZERNIKE2D"])
features = nyx.featurize(img, mask)
# 30 columns, ZERNIKE2D_Z0 to ZERNIKE2D_Z29
print(features[["ZERNIKE2D_Z0", "ZERNIKE2D_Z2", "ZERNIKE2D_Z3", "ZERNIKE2D_Z29"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["zernike"]
column = {"ZERNIKE2D": "ZERNIKE2D_Z0"}
for code, facts in extras.items():
    got = features[column.get(code, code)][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code} = {got}, page says {facts['example']}"

# The columns, in the order the page's table gives: order n ascending, then repetition m ascending,
# with n - m even
orders = [(n, m) for n in range(10) for m in range(n + 1) if (n - m) % 2 == 0]
names = [c for c in features.columns if c.startswith("ZERNIKE2D")]
assert names == ["ZERNIKE2D_Z%d" % k for k in range(30)], names
assert orders[3] == (2, 2) and orders[12] == (6, 0) and orders[29] == (9, 9)

# All 30 values on the worked example, as the page's table lists them
quoted = [0.3183, 0.0, 0.6051, 0.01349, 0.009830, 0.009666, 0.2363, 0.04984, 0.03165, 0.03977,
          0.03889, 0.0009361, 0.3727, 0.08806, 0.1363, 0.0003018, 0.07493, 0.06683, 0.001380,
          0.002329, 0.5813, 0.1047, 0.2508, 0.002560, 0.004304, 0.08618, 0.05453, 0.009942, 0.01636,
          0.0004792]
for k, expected in enumerate(quoted):
    got = features["ZERNIKE2D_Z%d" % k][0]
    assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
        f"Z{k} = {got}, page says {expected}"


def zern(image, roi):
    table = Nyxus(["ZERNIKE2D"]).featurize(image, roi)
    return [table["ZERNIKE2D_Z%d" % k][0] for k in range(30)]


# The disk geometry: centroid (2.9143, 2.8857) in 1-based pixel coordinates, radius 5
jj, ii = np.mgrid[:5, :5]
cx, cy = ((ii + 1) * img).sum() / img.sum(), ((jj + 1) * img).sum() / img.sum()
assert math.isclose(cx, 2.9143, rel_tol=5e-4) and math.isclose(cy, 2.8857, rel_tol=5e-4)
assert round(np.hypot(ii + 1 - cx, jj + 1 - cy).max() / 5, 2) == 0.59

# Z0 is 1/pi when every pixel lies in the disk; Z1 is 0 because the disk sits on the centroid
assert math.isclose(features["ZERNIKE2D_Z0"][0], 1 / math.pi, rel_tol=1e-12)
assert abs(features["ZERNIKE2D_Z1"][0]) < 1e-15

# A quarter turn and a mirror image leave every value unchanged; so does scaling the intensities
for other in (np.rot90(img), np.fliplr(img)):
    other = np.ascontiguousarray(other)
    assert np.allclose(zern(other, np.ones_like(other)), zern(img, mask), rtol=1e-12, atol=1e-15)
assert zern(img * 3, mask) == zern(img, mask)

# An elongated ROI: 3 x 9 bounding box, radius 3, so its ends fall outside the disk
long_img = np.array([[0, 0, 7, 8, 9, 2, 0, 0, 0],
                     [1, 3, 5, 0, 6, 4, 2, 6, 9],
                     [0, 2, 4, 6, 8, 0, 0, 3, 8]], dtype=np.uint32)
long_mask = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 0, 0, 1, 1]], dtype=np.uint32)
z = zern(long_img, long_mask)
assert math.isclose(z[0] * math.pi, 70 / 93, rel_tol=1e-12), z[0]
assert math.isclose(z[0], 0.2396, rel_tol=5e-4) and math.isclose(z[1], 0.09145, rel_tol=5e-4), z[:2]
# 5 of the 19 ROI pixels, holding 23 of the total intensity of 93, lie farther than R = 3 from the
# intensity centroid
jj, ii = np.mgrid[:3, :9]
w = (long_img * long_mask).astype(float)
cx, cy = ((ii + 1) * w).sum() / w.sum(), ((jj + 1) * w).sum() / w.sum()
out = (np.hypot(ii + 1 - cx, jj + 1 - cy) / 3 > 1) & (long_mask == 1)
assert (long_mask.sum(), out.sum(), w.sum(), w[out].sum()) == (19, 5, 93, 23)

# A centroid exactly on a pixel centre: that pixel is left out
sym = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.uint32)
z = zern(sym, np.ones_like(sym))
# pins known defect https://github.com/PolusAI/nyxus/issues/503
assert math.isclose(z[0] * math.pi, 0.75, rel_tol=1e-12), z[0]
assert math.isclose(z[2], 0.5040, rel_tol=5e-4), z[2]
# With the centre pixel kept, A20 = (3/pi) sum w (2 rho^2 - 1) would be 0.7427 (centroid (2, 2),
# R = 3)
jj, ii = np.mgrid[:3, :3]
rho2 = ((ii - 1) ** 2 + (jj - 1) ** 2) / 9
a20 = 3 / math.pi * (sym / sym.sum() * (2 * rho2 - 1)).sum()
assert math.isclose(abs(a20), 0.7427, rel_tol=5e-4), a20

# A constant ROI, such as a binary image inside its own foreground, gives 0 (the --noval value) in
# every column
const = np.full((4, 4), 7, dtype=np.uint32)
assert zern(const, np.ones_like(const)) == [0.0] * 30
assert zern(mask, mask) == [0.0] * 30
# The same 0/1 image under a mask that also covers its background is not constant
blob = np.zeros((7, 7), dtype=np.uint32)
blob[2:5, 1:6] = 1
assert zern(blob, blob) == [0.0] * 30
z = zern(blob, np.ones_like(blob))
assert math.isclose(z[0], 0.2971, rel_tol=5e-4) and math.isclose(z[2], 0.7873, rel_tol=5e-4), z[:3]
# A zero pixel in the mask stops a ROI of otherwise one intensity from counting as constant
const[0, 0] = 0
no_corner = np.ones_like(const)
no_corner[0, 0] = 0
assert zern(const, no_corner) == [0.0] * 30
assert zern(const, np.ones_like(const))[0] > 0.3

# An intensity-0 pixel inside the mask weighs the same as a pixel outside it, as long as it does not
# change the bounding box
dark = img.copy()
dark[2, 2] = 0
hole = mask.copy()
hole[2, 2] = 0
assert zern(dark, mask) == zern(dark, hole)
# A zero row inside the mask extends the bounding box, and with it the radius, from 4 to 5
dark = img.copy()
dark[4] = 0
short = mask.copy()
short[4] = 0
z_out, z_in = zern(dark, short), zern(dark, mask)
assert math.isclose(z_out[2], 0.5122, rel_tol=5e-4) and math.isclose(z_in[2], 0.6716, rel_tol=5e-4)

# Adding a constant to the intensities changes the weights, and so the moments
z = zern(img + 100, mask)
assert math.isclose(z[2], 0.6481, rel_tol=5e-4), z[2]

# IBSI mode changes nothing
ibsi = Nyxus(["ZERNIKE2D"], ibsi=True).featurize(img, mask)
for k in range(30):
    assert ibsi["ZERNIKE2D_Z%d" % k][0] == features["ZERNIKE2D_Z%d" % k][0], k
