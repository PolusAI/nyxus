# Example for docs/source/Math/f_glcm.rst: GLCM features of the page's 5x5 worked example.
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

nyx = Nyxus(["*ALL_GLCM*"], ibsi=True)
features = nyx.featurize(img, mask)
print(features[["GLCM_ASM_0", "GLCM_CONTRAST_0", "GLCM_JE_0", "GLCM_ASM_AVE"]])
# example-end

# Every 0-degree value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["glcm"]
for code, facts in extras.items():
    got = features[f"{code}_0"][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code}_0 = {got}, page says {facts['example']}"

# Values the page quotes for the default settings (64-level binning, ordered pairs)
default = Nyxus(["*ALL_GLCM*"]).featurize(img, mask)
quoted = {"CONTRAST": 473.45, "VARIANCE": 332.44, "JVAR": 345.41, "CLUSHADE": -585.66,
          "SUMVARIANCE": 882.25, "CLUTEND": 882.25}
for name, expected in quoted.items():
    got = default[f"GLCM_{name}_0"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"default GLCM_{name}_0 = {got}, page says {expected}"

# The default-settings values the IBSI definitions give on the same 64-level matrix of ordered pairs
binned = np.minimum(np.floor(64 * img.astype(int) / 5 + 1), 64).astype(int)
P = np.zeros((64, 64))
for r in range(5):
    for c in range(4):
        P[binned[r, c] - 1, binned[r, c + 1] - 1] += 1
p = P / P.sum()
i, j = np.arange(1, 65)[:, None], np.arange(1, 65)[None, :]
mu_x, mu_y = (i * p).sum(), (j * p).sum()
assert math.isclose(((i + j - mu_x - mu_y) ** 3 * p).sum(), 9490.19, rel_tol=5e-6), "IBSI cluster shade"
k = np.arange(2, 129)
p_sum = np.array([p[(i + j) == kk].sum() for kk in k])
sum_var = ((k - (k * p_sum).sum()) ** 2 * p_sum).sum()
assert math.isclose(sum_var, 867.43, rel_tol=5e-6), f"IBSI sum variance = {sum_var}, page says 867.43"

# The approximate logarithm against an exact one, on the symmetric IBSI-mode matrix of each direction
def exact_entropies(dx, dy):
    P = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            if 0 <= r + dy < 5 and 0 <= c + dx < 5:
                a, b = img[r, c] - 1, img[r + dy, c + dx] - 1
                P[a, b] += 1
                P[b, a] += 1
    p = P / P.sum()
    H = lambda v: float(-(v[v > 0] * np.log2(v[v > 0])).sum())
    px, py = p.sum(1), p.sum(0)
    i, j = np.arange(5)[:, None], np.arange(5)[None, :]
    pxpy = np.outer(px, py)
    hxy, hxy2 = H(p.ravel()), H(pxpy.ravel())
    hxy1 = float(-(p * np.log2(np.where(pxpy > 0, pxpy, 1))).sum())
    return {"JE": hxy, "SUMENTROPY": H(np.array([p[(i + j) == k].sum() for k in range(9)])),
            "DIFENTRO": H(np.array([p[abs(i - j) == k].sum() for k in range(5)])),
            "INFOMEAS2": math.sqrt(1 - math.exp(-2 * (hxy2 - hxy))),
            "INFOMEAS1": (hxy - hxy1) / max(H(px), H(py))}

rel = {}
for angle, (dx, dy) in zip((0, 45, 90, 135), ((1, 0), (1, 1), (0, 1), (-1, 1))):
    for code, exact in exact_entropies(dx, dy).items():
        rel.setdefault(code, []).append(abs(features[f"GLCM_{code}_{angle}"][0] / exact - 1))
assert math.isclose(exact_entropies(1, 0)["JE"], 3.80869, rel_tol=5e-6), "exact JE_0 differs from the page"
assert max(max(rel[c]) for c in ("JE", "SUMENTROPY", "DIFENTRO", "INFOMEAS2")) < 3.15e-3, rel
assert 3.75e-3 < min(rel["INFOMEAS1"]) and max(rel["INFOMEAS1"]) < 5.65e-3, rel["INFOMEAS1"]

# Constant-ROI test: the ROI's minimum and maximum binned to 64 levels, in IBSI mode too
def glcm_contrast(image, **meta):
    nyx = Nyxus(["*ALL_GLCM*"], ibsi=True)
    for key, value in meta.items():
        nyx.set_metaparam(f"glcm/{key}={value}")
    return nyx.featurize(image, np.ones_like(image))["GLCM_CONTRAST_0"][0]

checker = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint32)
assert glcm_contrast(checker + 1000) == 0, "a 1000/1001 ROI is not the substitute"
assert glcm_contrast(checker + 10) == 1, "a 10/11 ROI is the substitute"
assert glcm_contrast(checker + 10, greydepth=8) == 0, "glcm/greydepth does not reach the constant-ROI test"