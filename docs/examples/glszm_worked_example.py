# Example for docs/source/Math/f_glszm.rst: GLSZM features of the page's 5x5 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, in IBSI mode and with the default settings, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

img = np.array([[1, 2, 5, 2, 3],
                [3, 2, 1, 3, 1],
                [1, 3, 5, 5, 2],
                [1, 1, 1, 1, 2],
                [1, 2, 4, 3, 5]], dtype=np.uint32)
mask = np.ones_like(img)

nyx = Nyxus(["*ALL_GLSZM*"], ibsi=True)
features = nyx.featurize(img, mask)
print(features[["GLSZM_SAE", "GLSZM_LAE", "GLSZM_ZP", "GLSZM_ZE"]])
# example-end

# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["glszm"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"


# Zone entropy: Nyxus' approximate logarithm against an exact one on the page's matrix P
P = np.array([[3, 0, 0, 0, 0, 1],
              [2, 2, 0, 0, 0, 0],
              [1, 2, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [2, 1, 0, 0, 0, 0]], dtype=float)
p = P[P > 0] / P.sum()
exact = float(-(p * np.log2(p)).sum())
assert math.isclose(exact, 3.0566, rel_tol=5e-4), f"exact zone entropy = {exact}, page says 3.0566"
assert math.isclose(features["GLSZM_ZE"][0], 3.0544, rel_tol=5e-5), "GLSZM_ZE differs from the page's 3.0544"
assert P.sum() == 15 and math.isclose(features["GLSZM_ZP"][0], 15 / 25), "zone count differs from the page's P"


def glszm(image, roi, **kw):
    return Nyxus(["*ALL_GLSZM*"], **kw).featurize(image, roi)


# Grey level variance with the default 64-level binning
got = glszm(img, mask)["GLSZM_GLV"][0]
assert math.isclose(got, 341.05, rel_tol=5e-4), f"default GLSZM_GLV = {got}, page says 341.05"


# Zone growth: images whose zones are split by the east/south-east/south/south-west walk
split = np.array([[2, 2, 1],
                  [1, 1, 1]], dtype=np.uint32)
f = glszm(split, np.ones_like(split), ibsi=True)
quoted = {"ZP": 0.5, "SAE": 0.4537}
for name, expected in quoted.items():
    got = f[f"GLSZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"2x3 GLSZM_{name} = {got}, page says {expected}"

pyrad = np.array([[5, 2, 5, 4, 4],
                  [3, 3, 3, 1, 3],
                  [2, 1, 1, 1, 3],
                  [4, 2, 2, 2, 3],
                  [3, 5, 3, 3, 2]], dtype=np.uint32)
f = glszm(pyrad, np.ones_like(pyrad), ibsi=True)
quoted = {"ZP": 0.52, "SAE": 0.6596}
for name, expected in quoted.items():
    got = f[f"GLSZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"PyRadiomics image GLSZM_{name} = {got}, page says {expected}"

# Background as grey level 1: a 6x5 ROI whose four corners are outside the mask
roi_img = np.array([[3, 3, 7, 7, 7, 2],
                    [3, 5, 5, 7, 2, 2],
                    [6, 5, 5, 9, 9, 2],
                    [6, 6, 4, 9, 4, 4],
                    [1, 6, 4, 4, 4, 8]], dtype=np.uint32)
roi_mask = np.array([[0, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 0]], dtype=np.uint32)
default = glszm(roi_img, roi_mask)
quoted = {"ZP": 0.4, "SAE": 0.4144, "LGLZE": 0.3342}
for name, expected in quoted.items():
    got = default[f"GLSZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"default GLSZM_{name} = {got}, page says {expected}"
ibsi = glszm(roi_img, roi_mask, ibsi=True)
assert math.isclose(ibsi["GLSZM_ZP"][0], 8 / 26, rel_tol=5e-4), f"IBSI GLSZM_ZP = {ibsi['GLSZM_ZP'][0]}"

# Constant ROI: every feature is the missing-value substitute (0 by default)
const = np.full((4, 4), 7, dtype=np.uint32)
f = glszm(const, np.ones_like(const), ibsi=True)
assert all(f[f"GLSZM_{c}"][0] == 0 for c in ("SAE", "ZE", "GLV", "ZP")), "constant ROI is not the substitute"

# The 8-connected values for the 2 x 3 image: one zone of 4 pixels and one of 2
assert math.isclose(2 / 6, 0.3333, rel_tol=5e-4), "8-connected zone percentage"
assert math.isclose((1 / 16 + 1 / 4) / 2, 0.1563, rel_tol=5e-4), "8-connected small area emphasis"

# 32-bit overflow of large area emphasis, and small area emphasis on bounding boxes of 65 536 pixels
for n, wrapped, true in ((250, -194421147, 1953062501), (300, -245057295, 4049910001)):
    big = np.full((n, n), 5, dtype=np.uint32)
    big[n - 1, n - 1] = 9
    f = glszm(big, np.ones_like(big), ibsi=True)
    assert f["GLSZM_LAE"][0] == wrapped, f"{n}x{n} GLSZM_LAE = {f['GLSZM_LAE'][0]}, page says {wrapped}"
    assert ((n * n - 1) ** 2 + 1) / 2 == true, f"{n}x{n} true large area emphasis"
rng = np.random.default_rng(1)
for shape, low, high in (((255, 256), 0.45, 0.55), ((300, 300), 0, 0)):
    noise = rng.integers(1, 5, size=shape).astype(np.uint32)
    got = glszm(noise, np.ones_like(noise), ibsi=True)["GLSZM_SAE"][0]
    assert low <= got <= high, f"random texture {shape} GLSZM_SAE = {got}"

# The 6 x 5 ROI without its corners: IBSI mode on the image binned the default way, corners set to 0
binned = np.minimum(np.floor(64 * roi_img.astype(int) / 9) + 1, 64).astype(np.uint32) * roi_mask
f = glszm(binned, roi_mask, ibsi=True)
quoted = {"ZP": 0.3077, "SAE": 0.1215, "LGLZE": 0.001356}
for name, expected in quoted.items():
    got = f[f"GLSZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"corners excluded GLSZM_{name} = {got}, page says {expected}"

# Memory of the 32-bit matrix: N_g rows by one column per bounding-box pixel
assert round(4095 * 200 * 200 * 4 / 1e6) == 655 and round(64 * 300 * 300 * 4 / 1e6) == 23