# Example for docs/source/Math/f_fractal_dimension.rst: fractal dimensions of the page's 10 x 9
# worked-example ROI. The page shows the lines between the "example" markers. The assertions below
# pin every value the page quotes, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

mask = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]], dtype=np.uint32)
img = mask * 100    # the intensities do not matter; only the mask does

nyx = Nyxus(["FRACT_DIM_BOXCOUNT", "FRACT_DIM_PERIMETER"])
features = nyx.featurize(img, mask)
print(features[["FRACT_DIM_BOXCOUNT", "FRACT_DIM_PERIMETER"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["fractal"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code} = {got}, page says {facts['example']}"

F = ["FRACT_DIM_BOXCOUNT", "FRACT_DIM_PERIMETER"]


def fd(m, codes=F):
    m = np.asarray(m, dtype=np.uint32)
    row = Nyxus(codes).featurize(m * 100, m)
    return [row[c][0] for c in codes]


def check(got, quoted, label):
    for g, q in zip(got, quoted):
        assert math.isclose(g, q, rel_tol=5e-4, abs_tol=1e-12), f"{label}: {got}, page says {quoted}"


def square(n):
    return np.pad(np.ones((n, n), np.uint32), 2)


def disk(r):
    n = 2 * r + 5
    yy, xx = np.ogrid[:n, :n]
    return (((xx - n // 2) ** 2 + (yy - n // 2) ** 2) <= r * r).astype(np.uint32)


# The divider walk at stride 1 is the PERIMETER of the contour family: 32.63 on the example
check(fd(mask, ["FRACT_DIM_PERIMETER", "PERIMETER"]), [1.131, 32.63], "example PERIMETER")

# Only the mask matters: other intensities, including 0 inside the mask, give the same values
varied = np.where(mask > 0, np.arange(mask.size).reshape(mask.shape) % 7, 0).astype(np.uint32)
row = Nyxus(F).featurize(varied, mask)
assert [row[c][0] for c in F] == [features[c][0] for c in F]

# Squares: box counting reaches 2 only when the side fills its power-of-two canvas or the grid
# shifts absorb the rest; the divider dimension of a square is close to 1
check(fd(square(16)), [2.0, 1.008], "square 16")
check(fd(square(20)), [1.593, 1.006], "square 20")
check(fd(square(32)), [2.0, 1.004], "square 32")
check(fd(square(33)), [1.582, 1.0], "square 33")
check(fd(square(100)), [1.851, 1.003], "square 100")

# Disks
check(fd(disk(8)), [1.430, 1.060], "disk 8")
check(fd(disk(40)), [1.686, 1.032], "disk 40")

# One-pixel-wide straight lines: box counting reaches 1 only at power-of-two lengths, and the
# divider walk reads above 1 because the walk runs along the line and back, and the chords that
# span an end fold back
check(fd(np.ones((1, 16))), [1.0, 1.035], "line 16")
check(fd(np.ones((1, 10))), [0.7551, 1.101], "line 10")
check(fd(np.ones((1, 100)))[:1], [0.9257], "line 100")

# Two separate squares form one ROI with two borders, each walked on its own: the divider value is
# that of one square
two = np.zeros((40, 80), np.uint32)
two[5:30, 5:30] = 1
two[5:30, 45:70] = 1
check(fd(two), [1.423, 1.0], "two squares")
check(fd(square(25)), [1.842, 1.0], "one square")

# A hole's border, shorter than the large strides, is measured whole
holed = np.zeros((30, 30), np.uint32)
holed[3:27, 3:27] = 1
check(fd(holed)[1:], [1.005], "24 x 24 square")
holed[10:20, 10:20] = 0
check(fd(holed)[1:], [1.045], "24 x 24 square with a 10 x 10 hole")

# A 40 x 40 square and one separate pixel. In memory an intensity of 4294967295 makes the box
# count's marker overflow its 32-bit conversion, which is undefined, so only 100 is checked here
lone = np.zeros((80, 80), np.uint32)
lone[0:40, 0:40] = 1
lone[70, 70] = 1
inten = lone * 100
check([Nyxus(F).featurize(inten, lone)["FRACT_DIM_BOXCOUNT"][0]], [1.4156], "lone pixel")

# Small ROIs
check(fd(np.ones((1, 1))), [0.0, 0.0], "1 pixel")
check(fd(np.ones((1, 2))), [0.0, 0.0], "2 pixels")
check(fd(np.ones((1, 3))), [1.0, 1.0], "1 x 3")
check(fd(np.ones((3, 3))), [2.0, 1.0], "3 x 3")


# The box counts of the worked example, per grid origin, and of the 20 x 20 square (S = 32)
def box_counts(m):
    ys, xs = np.nonzero(m)
    xs, ys = xs - xs.min(), ys - ys.min()
    side = 1 << int(max(xs.max(), ys.max())).bit_length()
    sizes = [side >> k for k in range(side.bit_length() - 1)]
    return [[len(set(zip((xs + ox) // s, (ys + oy) // s))) for ox, oy in
             ((0, 0), (s // 2, 0), (0, s // 2), (s // 2, s // 2))] for s in sizes]


assert box_counts(mask) == [[1, 2, 2, 4], [4, 3, 4, 4], [8, 7, 9, 8], [20, 18, 18, 21]]
assert [min(c) for c in box_counts(square(20))] == [1, 4, 9, 25, 100]
# The unshifted grid alone gives 1.3966
s_ln, n_ln = np.log([16, 8, 4, 2]), np.log([c[0] for c in box_counts(mask)])
assert math.isclose(-np.polyfit(s_ln, n_ln, 1)[0], 1.3966, rel_tol=5e-4)

# Sides 2^k - 1 read exactly 2 (squares) and 1 (lines); a 2 x 2 square, like any ROI with L = 2,
# gives 0; a smooth outline can read slightly below 1
for k in (3, 7, 15, 31, 63):
    check([fd(square(k))[0], fd(np.ones((1, k)))[0]], [2.0, 1.0], f"side {k}")
assert fd(np.ones((2, 2)))[0] == 0.0
check(fd(square(6))[1:], [0.9947], "square 6")

# File input: out of core (ram_limit=0), anisotropy, merged labels and whole-slide mode
import tifffile


def from_files(m, intensity=None, single_roi=False, codes=F + ["PERIMETER"], **kw):
    m = np.asarray(m, dtype=np.uint32)
    tifffile.imwrite("fd_int.tif", (m * 100 if intensity is None else intensity).astype(np.uint32))
    tifffile.imwrite("fd_seg.tif", m)
    table = Nyxus(codes, **kw).featurize_files([os.path.abspath("fd_int.tif")],
                                               [os.path.abspath("fd_seg.tif")], single_roi)
    return [[table[c][i] for c in codes] for i in range(len(table))]


check(from_files(mask, ram_limit=0)[0], [1.3966, 1.1311, 26.0], "example out of core")
# pins known defect https://github.com/PolusAI/nyxus/issues/509
check(from_files(holed, ram_limit=0)[0][1:2], [1.005], "square with a hole out of core")
for value in (100, 4294967295):
    inten = lone * 100
    inten[70, 70] = value
    check(from_files(lone, inten, ram_limit=0)[0][:1], [1.4156], f"lone pixel {value} out of core")
check(from_files(mask)[0][:2], [1.3732, 1.1311], "example, no anisotropy")
check(from_files(mask, anisotropy_x=2)[0][:2], [1.2459, 1.1566], "anisotropy_x=2")
check(from_files(mask, anisotropy_y=3)[0][:2], [1.4458, 1.1635], "anisotropy_y=3")
# With anisotropy the out-of-core contour differs, so a single-contour disk differs too
check(from_files(disk(8), anisotropy_x=2)[0][1:2], [1.0917], "disk 8, anisotropy_x=2")
# pins known defect https://github.com/PolusAI/nyxus/issues/509
check(from_files(disk(8), anisotropy_x=2, ram_limit=0)[0][1:2], [1.0482], "disk 8, anisotropy_x=2, out of core")
# Without anisotropy a single contour can still differ out of core, which walks a one-pixel-wide
# part once where the in-memory walk goes along and back: a 1 x 16 line and an 8 x 8 square with a
# one-pixel-wide branch, at the image origin
line16 = np.zeros((4, 19), np.uint32)
line16[0, 0:16] = 1
check(from_files(line16)[0][1:2], [1.035], "line 16")
# pins known defect https://github.com/PolusAI/nyxus/issues/509
check(from_files(line16, ram_limit=0)[0][1:2], [1.190], "line 16, out of core")
branch = np.zeros((18, 18), np.uint32)
branch[0:8, 0:8] = 1
branch[3, 8:16] = 1
check(from_files(branch)[0][1:2], [1.190], "square with a branch")
# pins known defect https://github.com/PolusAI/nyxus/issues/509
check(from_files(branch, ram_limit=0)[0][1:2], [1.102], "square with a branch, out of core")
labels = two.copy()
labels[5:30, 45:70] = 2
for row in from_files(labels, (labels > 0) * 100):
    check(row[:2], [1.8416, 1.0], "two labels")
check(from_files(labels, (labels > 0) * 100, mergerois=True)[0][:2], [1.4231, 1.0], "merged")
# pins known defect https://github.com/PolusAI/nyxus/issues/493: the box is 201 x 201, which rounds
# up to the same 256 canvas as 200 x 200
check(from_files(np.ones((200, 200)), single_roi=True)[0][:2], [1.8703, 1.0], "whole slide")
# The whole-slide box is one pixel larger than the image, so a 256 x 256 image sits on a 512 canvas
# pins known defect https://github.com/PolusAI/nyxus/issues/493
check(from_files(np.ones((256, 256)), single_roi=True)[0][:1], [1.8667], "whole slide 256")
check(fd(square(256))[:1], [2.0], "square 256")
