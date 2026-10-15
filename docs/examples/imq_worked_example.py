# Example for docs/source/Math/f_image_quality.rst: image quality features of the page's 5x5 worked
# example, read from a file in whole-image mode. The page shows the lines between the "example"
# markers. The assertions below pin every value the page quotes, so the page cannot drift from the
# code.
import json
import math
import os
from pathlib import Path

# example-start
import tempfile

import numpy as np
import tifffile
from nyxus import ImageQuality

img = np.array([[5, 2, 5, 4, 4],
                [3, 3, 3, 1, 3],
                [2, 1, 1, 1, 3],
                [4, 2, 2, 2, 3],
                [3, 5, 3, 3, 2]], dtype=np.uint16)
folder = tempfile.mkdtemp()
tifffile.imwrite(os.path.join(folder, "example.tif"), img)

# Whole-image mode: the intensity directory doubles as the label directory
imq = ImageQuality(["*ALL_IMQ*"])
features = imq.featurize_directory(folder, folder)
print(features[["FOCUS_SCORE", "LOCAL_FOCUS_SCORE", "MAX_SATURATION", "MIN_SATURATION", "SHARPNESS"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["image_quality"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code} = {got}, page says {facts['example']}"


def check(table, quoted, label, row=0):
    for name, expected in quoted.items():
        got = table[name].iloc[row]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
            f"{label} {name} = {got}, page says {expected}"


def whole_image(images):
    """ImageQuality in whole-image mode on {name: array}; returns {name: row}."""
    d = tempfile.mkdtemp()
    for name, a in images.items():
        tifffile.imwrite(os.path.join(d, name + ".tif"), a)
    table = ImageQuality(["*ALL_IMQ*"]).featurize_directory(d, d)
    return {os.path.basename(p.replace("\\", "/"))[:-4]: table.iloc[[k]]
            for k, p in enumerate(table["intensity_image"])}


# Whole-image mode scores the image plus a row and a column of 0: 36 pixels, 11 of them 0
check(features, {"MAX_SATURATION": 3 / 36, "MIN_SATURATION": 11 / 36}, "whole image")

# LOCAL_FOCUS_SCORE: the four 3x3 tiles of the 6x6 image matrix, each filtered with 0 outside it
matrix = np.zeros((6, 6))
matrix[:5, :5] = img


def laplacian_var(t):
    p = np.pad(t, 1)
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * t
    return lap.var()


tiles = [laplacian_var(matrix[r:r + 3, c:c + 3]) for r in (0, 3) for c in (0, 3)]
for got, expected in zip(tiles, (42.32, 32.89, 35.06, 16.77)):
    assert math.isclose(got, expected, rel_tol=5e-4), f"tile score {got}, page says {expected}"
check(features, {"LOCAL_FOCUS_SCORE": sum(tiles) / 4, "FOCUS_SCORE": laplacian_var(matrix)}, "tiles")

# The same 5x5 image as an array, or with a mask of ones: no extra row and column
array = ImageQuality(["*ALL_IMQ*"]).featurize(img.astype(np.uint32), np.ones((5, 5), dtype=np.uint32))
check(array, {"FOCUS_SCORE": 30.12, "LOCAL_FOCUS_SCORE": 22.69, "POWER_SPECTRUM_SLOPE": 0.0,
              "MAX_SATURATION": 0.12, "MIN_SATURATION": 0.16, "SHARPNESS": 1.384}, "array")
seg = tempfile.mkdtemp()
os.makedirs(os.path.join(seg, "int"))
os.makedirs(os.path.join(seg, "seg"))
tifffile.imwrite(os.path.join(seg, "int", "example.tif"), img)
tifffile.imwrite(os.path.join(seg, "seg", "example.tif"), np.ones_like(img))
check(ImageQuality(["*ALL_IMQ*"]).featurize_directory(os.path.join(seg, "int"), os.path.join(seg, "seg")),
      {"FOCUS_SCORE": 30.12, "MAX_SATURATION": 0.12, "MIN_SATURATION": 0.16}, "mask of ones")

# A mask narrower than its bounding box: the 2 off-mask pixels of the 4x3 box count as 0
plus = np.zeros((5, 5), dtype=np.uint16)
plus[1:4, 1:4] = 1
plus[0, 2] = 1
tifffile.imwrite(os.path.join(seg, "seg", "example.tif"), plus)
check(ImageQuality(["*ALL_IMQ*"]).featurize_directory(os.path.join(seg, "int"), os.path.join(seg, "seg")),
      {"MIN_SATURATION": 2 / 12, "MAX_SATURATION": 1 / 12, "FOCUS_SCORE": 41.02}, "plus mask")


# A 128x128 disk blurred step by step
def blur(a, sigma):
    """Separable Gaussian blur, kernel truncated at 4 sigma, edges mirrored."""
    if sigma == 0:
        return a.astype(np.float64)
    r = int(4 * sigma + 0.5)
    x = np.arange(-r, r + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    a = np.pad(a.astype(np.float64), r, mode="symmetric")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, a)


yy, xx = np.mgrid[:128, :128]
disk = np.where((yy - 63.5) ** 2 + (xx - 63.5) ** 2 < 40 ** 2, 255, 0)
sigmas = (0, 1, 2, 4)
series = whole_image({f"s{s}": np.round(blur(disk, s)).astype(np.uint16) for s in sigmas})
quoted = {"FOCUS_SCORE": (3970.0, 124.0, 17.04, 2.828),
          "POWER_SPECTRUM_SLOPE": (1.178, 1.653, 1.996, 2.324),
          "SHARPNESS": (4.748, 11.23, 3.755, 1.881)}
for code, values in quoted.items():
    for s, v in zip(sigmas, values):
        check(series[f"s{s}"], {code: v}, f"disk sigma={s}")
check(series["s1"], {"LOCAL_FOCUS_SCORE": 1404.0}, "disk sigma=1")

# The unblurred disk with no extra row and column (a mask of ones): the slope is not 1.178
ones = np.ones((128, 128), dtype=np.uint32)
check(ImageQuality(["*ALL_IMQ*"]).featurize(disk.astype(np.uint32), ones),
      {"POWER_SPECTRUM_SLOPE": 2.106}, "disk mask of ones")

# A constant image (as an array, so without the extra row and column)
const = np.full((30, 30), 7, dtype=np.uint32)
check(ImageQuality(["*ALL_IMQ*"]).featurize(const, np.ones_like(const)),
      {"FOCUS_SCORE": 6.098, "LOCAL_FOCUS_SCORE": 11.32, "POWER_SPECTRUM_SLOPE": 0.0,
       "MAX_SATURATION": 1.0, "MIN_SATURATION": 0.0, "SHARPNESS": 0.0}, "constant")

# Too small: LOCAL_FOCUS_SCORE needs both sides >= 2, POWER_SPECTRUM_SLOPE the shorter side >= 24
row = np.array([[3, 7, 1]], dtype=np.uint32)
check(ImageQuality(["*ALL_IMQ*"]).featurize(row, np.ones_like(row)),
      {"FOCUS_SCORE": 128.2, "LOCAL_FOCUS_SCORE": 0.0}, "1x3")
g = np.random.default_rng(1).integers(0, 50, (23, 40)).astype(np.uint32)
check(ImageQuality(["*ALL_IMQ*"]).featurize(g, np.ones_like(g)), {"POWER_SPECTRUM_SLOPE": 0.0}, "23x40")

# Whole-image mode names the image by the path it was found under, relative if the directory is
here = os.getcwd()
os.chdir(folder)
try:
    name = ImageQuality(["*ALL_IMQ*"]).featurize_directory(".", ".")["intensity_image"][0]
finally:
    os.chdir(here)
assert not os.path.isabs(name) and name.replace("\\", "/").endswith("example.tif"), name
