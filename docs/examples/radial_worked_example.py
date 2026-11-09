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
quoted = {"FRAC_AT_D": [0.02128, 0.08511, 0.08511, 0.2553, 0.08511, 0.2340, 0.1702, 0.06383],
          "MEAN_FRAC": [9.0, 8.5, 7.5, 5.75, 4.0, 3.636, 2.5, 2.0],
          "RADIAL_CV": [2.646, 1.003, 1.004, 0.3709, 1.031, 0.5292, 0.7483, 1.291]}
for code, values in quoted.items():
    for k, (got, want) in enumerate(zip(vec(features, code), values)):
        assert math.isclose(got, want, rel_tol=5e-4), f"{code}_{k} = {got}, page says {want}"


def contour_pixels(mask):
    """The contour as the page describes it: the ROI pixels with a 4-neighbour outside the ROI,
    as (x, y) arrays."""
    m = mask > 0
    p = np.pad(m, 1)
    edge = m & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    ky, kx = np.nonzero(edge)
    return kx, ky


def formulas(img, mask):
    """The page's formulas with full scans for the centre and the radius. Returns the three
    vectors, the centre (x, y), R^2 and the pixel count of each ring."""
    m = mask > 0
    xs, ys = np.nonzero(m.T)                       # pixels column by column, as Nyxus.featurize
    inten = img[ys, xs].astype(float)
    kx, ky = contour_pixels(mask)
    d2 = (xs[:, None] - kx[None, :]) ** 2 + (ys[:, None] - ky[None, :]) ** 2
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


# The formulas reproduce Nyxus on the worked example: the searches find the full-scan centre and R
mine, centre, r2, n = formulas(img, mask)
for code in CODES:
    assert np.allclose(mine[code], vec(features, code), rtol=1e-12, atol=1e-12), code
assert centre == (4, 3) and r2 == 17 and n == [1, 4, 4, 12, 4, 11, 8, 3]
assert len(contour_pixels(mask)[0]) == 20 and img[3, 4] == 9
d = np.hypot(*(np.nonzero(mask.T) - np.array(centre)[:, None]))
assert (d > math.sqrt(17) + 1e-12).sum() == 0 and np.isclose(d, math.sqrt(17)).sum() == 3
assert math.isclose(vec(features, "FRAC_AT_D").sum(), 1.0, rel_tol=1e-9)
assert math.isclose((vec(features, "FRAC_AT_D") * vec(features, "MEAN_FRAC")).sum() * 47, 224, rel_tol=1e-9)
assert math.isclose(vec(features, "RADIAL_CV")[5], math.sqrt(7) / 5, rel_tol=1e-9)

# The centre's spread of squared distances to the contour, and the runner-up's
kx, ky = contour_pixels(mask)
spread = {p: (kx - p[0]) ** 2 + (ky - p[1]) ** 2 for p in ((4, 3), (4, 2))}
assert spread[(4, 3)].min() == 5 and spread[(4, 3)].max() == 17
assert np.ptp(spread[(4, 2)]) == 16

# A 3 x 3 square with intensities 1..9 row by row: the centre is the middle pixel
sq_img = np.pad(np.arange(1, 10, dtype=np.uint32).reshape(3, 3), 2)
sq_mask = np.pad(np.ones((3, 3), dtype=np.uint32), 2)
sq = nyx.featurize(sq_img, sq_mask)
assert math.isclose(sq["MEAN_FRAC_0"][0], 5.0, rel_tol=1e-6)
assert np.allclose(vec(sq, "FRAC_AT_D") * 9, [1, 0, 0, 0, 4, 0, 0, 4], atol=1e-6)

# A 5 x 4 rectangle: the searches stop short of the farthest contour pixel, R^2 = 5 instead of 8
rect = np.pad(np.ones((4, 5), dtype=np.uint32), 2)
r_img = (rect * (1 + np.arange(rect.size).reshape(rect.shape) % 7)).astype(np.uint32)
r = nyx.featurize(r_img, rect)
r_full, r_centre, r_r2, r_n = formulas(r_img, rect)
assert r_centre == (4, 3) and r_r2 == 8 and r_n == [1, 0, 4, 4, 3, 6, 0, 2]
assert np.allclose(vec(r, "FRAC_AT_D") * 20, [1, 0, 0, 4, 4, 0, 3, 8], atol=1e-6)
# Nyxus's rings are those of centre (4, 3) with R^2 = 5: 2 pixels lie beyond R, none for 8
ry, rx = np.nonzero(rect)
r_d = np.hypot(rx - 4, ry - 3)
assert np.bincount(np.minimum((r_d / math.sqrt(5) * 7).astype(int), 7), minlength=8).tolist() == \
    [1, 0, 0, 4, 4, 0, 3, 8]
assert (r_d > math.sqrt(5) + 1e-12).sum() == 2 and (r_d > math.sqrt(8) + 1e-12).sum() == 0

# A pixel of intensity 0 inside the mask counts as a pixel: the centre set to 0
zero = img.copy()
zero[3, 4] = 0
z = nyx.featurize(zero, mask)
assert z["MEAN_FRAC_0"][0] == 0 and z["RADIAL_CV_0"][0] == 0
assert np.allclose(vec(z, "FRAC_AT_D"), vec(features, "FRAC_AT_D"))

# A ROI of one intensity is not skipped
flat = nyx.featurize(np.pad(np.full((5, 5), 4, dtype=np.uint32), 2), np.pad(np.ones((5, 5), np.uint32), 2))
assert np.allclose(vec(flat, "FRAC_AT_D") * 25, [1, 0, 4, 4, 4, 8, 0, 4], atol=1e-6)
assert np.allclose(vec(flat, "MEAN_FRAC"), [4, 0, 4, 4, 4, 4, 0, 4], rtol=1e-6)
assert math.isclose(flat["RADIAL_CV_7"][0], 1.0, rel_tol=5e-4)

# A ROI of 1 pixel is its own centre, at radius 0; a ROI of 2 pixels puts one in ring 0, one in ring 7
for m, frac in ((np.array([[1]]), [1, 0, 0, 0, 0, 0, 0, 0]), (np.array([[1, 1]]), [0.5, 0, 0, 0, 0, 0, 0, 0.5])):
    tiny = nyx.featurize(np.pad(5 * m, 2).astype(np.uint32), np.pad(m, 2).astype(np.uint32))
    assert np.allclose(vec(tiny, "FRAC_AT_D"), frac, atol=1e-6)
    assert math.isclose(tiny["MEAN_FRAC_0"][0], 5.0, rel_tol=1e-6)
    assert math.isclose(tiny["RADIAL_CV_0"][0], math.sqrt(7), rel_tol=1e-6)

# File input. A tie between two centre candidates: an array is read column by column, a file tile by
# tile and row by row, so the first candidate differs
import tifffile


def from_files(intensity, roi, **kw):
    tifffile.imwrite("rd_int.tif", intensity.astype(np.uint32))
    tifffile.imwrite("rd_seg.tif", roi.astype(np.uint32))
    return Nyxus(list(CODES), **kw).featurize_files(
        [os.path.abspath("rd_int.tif")], [os.path.abspath("rd_seg.tif")], False)


tie = np.pad(np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint32), 2)
tie_img = (tie * (1 + np.arange(tie.size).reshape(tie.shape) % 9)).astype(np.uint32)
assert tie.sum() == 11
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
assert math.isclose(out["FRAC_AT_D_7"][0], 0.5532, rel_tol=5e-4), out["FRAC_AT_D_7"][0]
assert math.isclose(out["MEAN_FRAC_0"][0], 3.0, rel_tol=1e-6), out["MEAN_FRAC_0"][0]

# Whole-slide mode: the contour is the four corners of the bounding box, one pixel past the last
# column and row; the centre is the pixel equally far from all four, (10, 10) with R = 14.14 on a
# 20 x 20 image and (32, 24) with R = 40 on a 64 x 48 image. The ring counts Nyxus reports are
# those of that centre.
for (h, w), (ox, oy), ring0 in (((20, 20), (10, 10), 13), ((48, 64), (32, 24), 101)):
    tifffile.imwrite("rd_whole.tif", np.full((h, w), 7, np.uint32))
    t = Nyxus(["FRAC_AT_D"]).featurize_files([os.path.abspath("rd_whole.tif")],
                                             [os.path.abspath("rd_whole.tif")], True)
    counts = vec(t, "FRAC_AT_D") * h * w
    wy, wx = np.mgrid[0:h, 0:w]
    rings = np.minimum((np.hypot(wx - ox, wy - oy) / math.hypot(ox, oy) * 7).astype(int), 7)
    assert np.allclose(counts, np.bincount(rings.ravel(), minlength=8), atol=1e-6), (h, w, counts)
    assert round(counts[0]) == ring0, (h, w, counts[0])
assert math.isclose(math.hypot(10, 10), 14.14, rel_tol=5e-4)
