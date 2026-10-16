# Example for docs/source/Math/f_gabor.rst: GABOR on the page's worked example, the 5x5 image of the
# texture pages with its four corners left out of the mask. The page shows the lines between the
# "example" markers. The assertions below pin every value the page quotes, so the page cannot drift
# from the code.
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
mask[0, 0] = mask[0, 4] = mask[4, 0] = mask[4, 4] = 0   # the four corners are not in the ROI

# The default pairs: frequencies 4, 16, 32, 64 at 0, 45, 90, 135 degrees
nyx = Nyxus(["GABOR"])
features = nyx.featurize(img, mask)
print(features[["GABOR_0", "GABOR_1", "GABOR_2", "GABOR_3"]])
# example-end


def columns(table):
    return [float(table[c][0]) for c in table.columns if c.startswith("GABOR_")]


def check(got, expected, label):
    assert len(got) == len(expected), f"{label}: {len(got)} GABOR columns, page says {len(expected)}"
    for k, (g, e) in enumerate(zip(got, expected)):
        assert math.isclose(g, e, rel_tol=5e-4, abs_tol=1e-12), f"{label} GABOR_{k} = {g}, page says {e}"


# The page's feature index (the value of GABOR_0), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["gabor"]
column = {"GABOR": "GABOR_0"}
for code, facts in extras.items():
    got = features[column.get(code, code)][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"

# Every column of the worked example: 25 of 25 pixels pass in the first three filters, 23 in the
# last, over a baseline count of 24
check(columns(features), [25 / 24, 25 / 24, 25 / 24, 23 / 24], "worked example")

# Scaling the intensities changes nothing; adding an offset does
check(columns(Nyxus(["GABOR"]).featurize(img * 7, mask)), [25 / 24, 25 / 24, 25 / 24, 23 / 24], "x7")
check(columns(Nyxus(["GABOR"]).featurize(img + 100, mask)), [20 / 24, 25 / 24, 25 / 24, 23 / 24], "+100")

# The default frequencies are radians per pixel: wavelengths 2 pi / f of 1.57, 0.39, 0.20 and 0.10
# pixels, envelope widths sig2lam * wavelength of 1.26 down to 0.08 pixels
wavelengths = [2 * math.pi / f for f in (4, 16, 32, 64)]
assert [round(w, 2) for w in wavelengths] == [1.57, 0.39, 0.20, 0.10], wavelengths
assert round(0.8 * wavelengths[0], 2) == 1.26 and round(0.8 * wavelengths[-1], 2) == 0.08
# The baseline wavelength is 62.8 pixels; settings are read in single precision
assert round(2 * math.pi / 0.1, 1) == 62.8
stored = Nyxus(["GABOR"]).get_params("gabor_gamma", "gabor_freqs")
assert stored["gabor_gamma"] == float(np.float32(0.1)), stored
assert stored["gabor_freqs"] == [float(np.float32(f)) for f in (4, 16, 32, 64)], stored

# IBSI mode has no effect
check(columns(Nyxus(["GABOR"], ibsi=True).featurize(img, mask)), [25 / 24, 25 / 24, 25 / 24, 23 / 24], "ibsi")

# An odd kernel size: the response image is shifted by one pixel
check(columns(Nyxus(["GABOR"], gabor_kersize=15).featurize(img, mask)),
      [1.0417, 0.9167, 0.8333, 0.9167], "kersize 15")

# Stripes with a period of 4 pixels, varying along x (vertical stripes) and along y (horizontal)
yy, xx = np.mgrid[0:24, 0:24]
vertical = (10 + 8 * np.cos(2 * np.pi * xx / 4)).astype(np.uint32)
horizontal = vertical.T.copy()
full = np.ones_like(vertical)
check(columns(Nyxus(["GABOR"]).featurize(vertical, full)), [1.0017, 1.0017, 1.0017, 1.0017], "vertical, defaults")
check(columns(Nyxus(["GABOR"]).featurize(horizontal, full)), [0.16696, 1.0017, 1.0017, 1.0017],
      "horizontal, defaults")
matched = dict(gabor_freqs=[1.5708, 1.5708], gabor_thetas=[0, 90])
check(columns(Nyxus(["GABOR"], **matched).featurize(vertical, full)), [1.0017, 0.38957], "vertical, pi/2")
check(columns(Nyxus(["GABOR"], **matched).featurize(horizontal, full)), [0.38957, 1.0017], "horizontal, pi/2")
check(columns(Nyxus(["GABOR"], **matched, gabor_thold=0.3).featurize(vertical, full)), [0.65913, 0.0],
      "vertical, pi/2, thold 0.3")

# A constant ROI, and a two-pixel ROI whose single counted baseline pixel gives 2
const = np.full((5, 5), 7, dtype=np.uint32)
check(columns(Nyxus(["GABOR"]).featurize(const, mask)), [0.0, 0.0, 0.0, 0.0], "constant")
pair = np.array([[3, 8]], dtype=np.uint32)
check(columns(Nyxus(["GABOR"]).featurize(pair, np.ones_like(pair))), [2.0, 2.0, 2.0, 2.0], "two pixels")

# The Gabor settings belong to the process: the object built last decides them for every object
first = Nyxus(["GABOR"], **matched)
check(columns(first.featurize(vertical, full)), [1.0017, 0.38957], "first object alone")
second = Nyxus(["GABOR"])
check(columns(first.featurize(vertical, full)), [1.0017, 1.0017, 1.0017, 1.0017], "first object after second")
