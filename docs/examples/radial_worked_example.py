# Example for docs/source/Math/f_radial_distribution.rst: radial distribution features of the
# page's 47-pixel ROI. The page shows the lines between the "example" markers. The assertions below
# pin every value the page quotes, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

mask = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 1, 1, 1, 0, 0, 0, 0]], dtype=np.uint32)
img = np.array([[0, 0, 3, 4, 4, 3, 0, 0, 0],
                [0, 3, 5, 6, 6, 5, 3, 2, 0],
                [2, 4, 6, 8, 9, 7, 5, 3, 0],
                [3, 5, 7, 9, 9, 8, 6, 4, 2],
                [2, 4, 6, 8, 8, 7, 5, 3, 2],
                [0, 3, 4, 6, 6, 5, 4, 2, 0],
                [0, 0, 2, 3, 3, 0, 0, 0, 0]], dtype=np.uint32)

nyx = Nyxus(["FRAC_AT_D", "MEAN_FRAC", "RADIAL_CV"])
features = nyx.featurize(img, mask)
for code in ("FRAC_AT_D", "MEAN_FRAC", "RADIAL_CV"):
    print(code, features[[f"{code}_{k}" for k in range(8)]].to_numpy().round(4))
# example-end

CODES = ("FRAC_AT_D", "MEAN_FRAC", "RADIAL_CV")


def vec(table, code):
    return np.array([table[f"{code}_{k}"][0] for k in range(8)])


# Every value in the page's feature index, as recorded in docs/feature_extras.toml; the example of
# a multi-column code is its first column, ring 0
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["radial"]
for code, facts in extras.items():
    got = features[f"{code}_0"][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code}_0 = {got}, page says {facts['example']}"

# The 24 values of the worked-example table
quoted = {"FRAC_AT_D": [0.02128, 0.08511, 0.08511, 0.2128, 0.06383, 0.1702, 0.1277, 0.2340],
          "MEAN_FRAC": [7.0, 6.5, 6.25, 5.6, 4.667, 4.375, 3.833, 3.455],
          "RADIAL_CV": [2.646, 1.052, 1.079, 0.7284, 1.532, 0.9558, 1.223, 1.571]}
for code, values in quoted.items():
    for k, (got, want) in enumerate(zip(vec(features, code), values)):
        assert math.isclose(got, want, rel_tol=5e-4), f"{code}_{k} = {got}, page says {want}"


def formulas(img, mask, offset=1):
    """The page's formulas: contour = ROI pixels with a 4-neighbour outside the ROI, moved by
    (offset, offset); full scans for the centre and the radius. Returns the three vectors, the
    centre (x, y), R^2 and the pixel count of each ring."""
    m = mask > 0
    xs, ys = np.nonzero(m.T)                       # pixels column by column, as Nyxus.featurize
    inten = img[ys, xs].astype(float)
    p = np.pad(m, 1)
    edge = m & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    ky, kx = np.nonzero(edge)
    d2 = (xs[:, None] - kx[None, :] - offset) ** 2 + (ys[:, None] - ky[None, :] - offset) ** 2
    o = int(np.argmin(d2.max(1) - d2.min(1)))
    r2 = d2[o].max()
    d = np.hypot(xs - xs[o], ys - ys[o])
    ring = np.minimum((d / math.sqrt(r2) * 7).astype(int), 7)
    theta = np.mod(np.arctan2(ys - ys[o], xs - xs[o]), 2 * np.pi)
    wedge = (theta / (np.pi / 4)).astype(int)
    n = np.bincount(ring, minlength=8)
    s = np.bincount(ring, weights=inten, minlength=8)
    w = np.zeros((8, 8))
    np.add.at(w, (ring, wedge), inten)
    mu = w.mean(1)
    cv = np.sqrt(((w - mu[:, None]) ** 2).mean(1)) / (mu + 1e-9)
    out = {"FRAC_AT_D": n / (len(xs) + 1e-9), "MEAN_FRAC": s / (n + 1e-9), "RADIAL_CV": cv}
    return out, (int(xs[o]), int(ys[o])), r2, n.tolist()


# The formulas reproduce Nyxus on the worked example
mine, centre, r2, n = formulas(img, mask)
for code in CODES:
    assert np.allclose(mine[code], vec(features, code), rtol=1e-12, atol=1e-12), code
assert centre == (5, 4) and r2 == 17 and n == [1, 4, 4, 10, 3, 8, 6, 11]
d = np.hypot(*(np.nonzero(mask.T) - np.array(centre)[:, None]))
assert (d > math.sqrt(17) + 1e-12).sum() == 8 and np.isclose(d, math.sqrt(17)).sum() == 3
assert math.isclose(vec(features, "FRAC_AT_D").sum(), 1.0, rel_tol=1e-9)
assert math.isclose((vec(features, "FRAC_AT_D") * vec(features, "MEAN_FRAC")).sum() * 47, 224, rel_tol=1e-9)
assert math.isclose(vec(features, "RADIAL_CV")[3], math.sqrt(26) / 7, rel_tol=1e-9)

# pins known defect https://github.com/PolusAI/nyxus/issues/497
# The contour points (boundary pixels moved by +1, +1), the centre's spread and the runner-up's
inner = np.pad(mask > 0, 1)
edge = (mask > 0) & ~(inner[:-2, 1:-1] & inner[2:, 1:-1] & inner[1:-1, :-2] & inner[1:-1, 2:])
ky, kx = np.nonzero(edge)
cy, cx = ky + 1, kx + 1
assert len(cx) == 20
assert sum(not (y < 7 and x < 9 and mask[y, x]) for y, x in zip(cy, cx)) == 9
spread = {p: (cx - p[0]) ** 2 + (cy - p[1]) ** 2 for p in ((5, 4), (5, 3))}
assert spread[(5, 4)].min() == 5 and spread[(5, 4)].max() == 17
assert np.ptp(spread[(5, 3)]) == 16

# The contour offset: on the ROI's own boundary the centre and the rings would be different
on_roi, centre0, r20, n0 = formulas(img, mask, offset=0)
assert centre0 == (4, 3) and r20 == 17 and n0 == [1, 4, 4, 12, 4, 11, 8, 3]
assert math.isclose(on_roi["MEAN_FRAC"][0], 9.0, rel_tol=1e-6)
assert math.isclose(on_roi["RADIAL_CV"][3], 0.3709, rel_tol=5e-4)

# A 3 x 3 square with intensities 1..9 row by row: the centre is the bottom-right corner
sq_img = np.pad(np.arange(1, 10, dtype=np.uint32).reshape(3, 3), 2)
sq_mask = np.pad(np.ones((3, 3), dtype=np.uint32), 2)
sq = nyx.featurize(sq_img, sq_mask)
assert math.isclose(sq["MEAN_FRAC_0"][0], 9.0, rel_tol=1e-6)
assert np.allclose(vec(sq, "FRAC_AT_D") * 9, [1, 0, 0, 0, 2, 0, 0, 6], atol=1e-6)
sq0 = formulas(sq_img, sq_mask, offset=0)[0]
assert math.isclose(sq0["MEAN_FRAC"][0], 5.0, rel_tol=1e-6)
assert np.allclose(sq0["FRAC_AT_D"] * 9, [1, 0, 0, 0, 4, 0, 0, 4], atol=1e-6)

# A ROI with a hole: the coded searches find a smaller radius than the farthest contour pixel
holed = np.pad(np.array([[0, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 1, 0, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 0]], dtype=np.uint32), 2)
h_img = (holed * (1 + np.arange(holed.size).reshape(holed.shape) % 7)).astype(np.uint32)
h = nyx.featurize(h_img, holed)
h_full, h_centre, h_r2, h_n = formulas(h_img, holed)
assert h_centre == (7, 6) and h_r2 == 20 and h_n == [1, 4, 4, 12, 7, 4, 7, 11]
assert np.allclose(vec(h, "FRAC_AT_D") * 50, [1, 4, 4, 12, 4, 7, 6, 12], atol=1e-6)
assert math.isclose(h["FRAC_AT_D_4"][0], 0.08, rel_tol=1e-6)
assert math.isclose(h_full["FRAC_AT_D"][4], 0.14, rel_tol=1e-6)
# Nyxus's rings are those of centre (7, 6) with R^2 = 18: 11 pixels lie beyond R, against 8 for 20
hy, hx = np.nonzero(holed)
h_d = np.hypot(hx - 7, hy - 6)
assert np.bincount(np.minimum((h_d / math.sqrt(18) * 7).astype(int), 7), minlength=8).tolist() == \
    [1, 4, 4, 12, 4, 7, 6, 12]
assert (h_d > math.sqrt(18) + 1e-12).sum() == 11 and (h_d > math.sqrt(20) + 1e-12).sum() == 8

# A pixel of intensity 0 inside the mask counts as a pixel: the centre set to 0
zero = img.copy()
zero[4, 5] = 0
z = nyx.featurize(zero, mask)
assert z["MEAN_FRAC_0"][0] == 0 and z["RADIAL_CV_0"][0] == 0
assert np.allclose(vec(z, "FRAC_AT_D"), vec(features, "FRAC_AT_D"))

# A ROI of one intensity is not skipped
flat = nyx.featurize(np.pad(np.full((5, 5), 4, dtype=np.uint32), 2), np.pad(np.ones((5, 5), np.uint32), 2))
assert np.allclose(vec(flat, "FRAC_AT_D") * 25, [1, 0, 4, 4, 2, 4, 0, 10], atol=1e-6)
assert np.allclose(vec(flat, "MEAN_FRAC"), [4, 0, 4, 4, 4, 4, 0, 4], rtol=1e-6)
assert math.isclose(flat["RADIAL_CV_7"][0], 1.183, rel_tol=5e-4)

# ROIs of 1 or 2 pixels have no contour and report 0 in all 24 columns
for m in (np.array([[1]]), np.array([[1, 1]])):
    tiny = nyx.featurize(np.pad(5 * m, 2).astype(np.uint32), np.pad(m, 2).astype(np.uint32))
    assert all((vec(tiny, c) == 0).all() for c in CODES)

# File input. A tie between two centre candidates: an array is read column by column, a file tile by
# tile and row by row, so the first candidate differs
import tifffile


def from_files(intensity, roi, **kw):
    tifffile.imwrite("rd_int.tif", intensity.astype(np.uint32))
    tifffile.imwrite("rd_seg.tif", roi.astype(np.uint32))
    return Nyxus(list(CODES), **kw).featurize_files(
        [os.path.abspath("rd_int.tif")], [os.path.abspath("rd_seg.tif")], False)


tie = np.pad(np.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=np.uint32), 2)
tie_img = (tie * (1 + np.arange(tie.size).reshape(tie.shape) % 9)).astype(np.uint32)
assert tie.sum() == 10
assert math.isclose(nyx.featurize(tie_img, tie)["MEAN_FRAC_0"][0], 9.0, rel_tol=1e-6)
assert math.isclose(from_files(tie_img, tie)["MEAN_FRAC_0"][0], 2.0, rel_tol=1e-6)

# Out of core (ram_limit=0): the worked example at the image origin gets the in-memory values;
# moved 2 pixels right and 3 down it does not
at_origin = np.zeros((20, 20), np.uint32)
at_origin_mask = np.zeros((20, 20), np.uint32)
at_origin[:7, :9], at_origin_mask[:7, :9] = img, mask
out = from_files(at_origin, at_origin_mask, ram_limit=0)
for code in CODES:
    assert np.allclose(vec(out, code), vec(features, code), rtol=1e-9, atol=1e-12), code
moved = np.roll(np.roll(at_origin, 3, axis=0), 2, axis=1)
moved_mask = np.roll(np.roll(at_origin_mask, 3, axis=0), 2, axis=1)
assert np.allclose(vec(from_files(moved, moved_mask), "FRAC_AT_D"), vec(features, "FRAC_AT_D"))
out = from_files(moved, moved_mask, ram_limit=0)
# pins known defect https://github.com/PolusAI/nyxus/issues/509
assert math.isclose(out["FRAC_AT_D_7"][0], 0.3404, rel_tol=5e-4), out["FRAC_AT_D_7"][0]
assert math.isclose(out["MEAN_FRAC_0"][0], 6.0, rel_tol=1e-6), out["MEAN_FRAC_0"][0]
