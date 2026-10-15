# Example for docs/source/Math/f_gldzm.rst: GLDZM features of the page's 5x5 worked example.
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

nyx = Nyxus(["*ALL_GLDZM*"], ibsi=True)
features = nyx.featurize(img, mask)
print(features[["GLDZM_SDE", "GLDZM_LDE", "GLDZM_ZP", "GLDZM_ZDE"]])
# example-end

# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["gldzm"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"


# The page's matrix P reproduces the zone count, zone percentage and the exact-logarithm entropy
P = np.array([[3, 1],
              [4, 0],
              [3, 0],
              [1, 0],
              [2, 1]], dtype=float)
p = P[P > 0] / P.sum()
exact = float(-(p * np.log2(p)).sum())
assert math.isclose(features["GLDZM_ZDE"][0], exact, rel_tol=1e-12), "GLDZM_ZDE is not the exact entropy of P"
assert P.sum() == 15 and math.isclose(features["GLDZM_ZP"][0], 15 / 25), "zone count differs from the page's P"
d = np.arange(1, 3)
assert math.isclose(features["GLDZM_SDE"][0], (P.sum(0) / d**2).sum() / 15), "GLDZM_SDE differs from P"


def gldzm(image, roi, **kw):
    return Nyxus(["*ALL_GLDZM*"], **kw).featurize(image, roi)


# Default 64-level binning on the worked example
default = gldzm(img, mask)
quoted = {"GLV": 341.05, "HGLZE": 1529.0, "LGLZE": 0.002177, "GLM": 34.47}
for name, expected in quoted.items():
    got = default[f"GLDZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"default GLDZM_{name} = {got}, page says {expected}"


# Three pixels of the bounding box outside the mask
holes = np.ones_like(img)
holes[0, 0] = holes[0, 1] = holes[4, 4] = 0
masked_default = gldzm(img, holes)
masked_ibsi = gldzm(img, holes, ibsi=True)
quoted = [(masked_default, "ZP", 0.6818), (masked_default, "LGLZE", 0.1351),
          (masked_ibsi, "ZP", 0.5909), (masked_ibsi, "SDE", 0.8269), (masked_ibsi, "ZDM", 1.231)]
for table, name, expected in quoted:
    got = table[f"GLDZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"masked GLDZM_{name} = {got}, page says {expected}"

# The same 64-level binning without those three pixels: the example binned by hand to levels 13,
# 26, 39, 52 and 64, with the outside pixels set to 0, in IBSI mode so that 0 stays background
binned = np.minimum(64 * img // 5 + 1, 64).astype(np.uint32)
got = gldzm(binned * holes, holes, ibsi=True)["GLDZM_LGLZE"][0]
assert math.isclose(got, 0.002038, rel_tol=5e-4), f"masked binned GLDZM_LGLZE = {got}, page says 0.002038"

# Relation quoted on the page: GLSZM finds the same zones, so its grey level non-uniformity is 3.4 too
glszm = Nyxus(["*ALL_GLSZM*"], ibsi=True).featurize(img, mask)
assert math.isclose(glszm["GLSZM_GLN"][0], 3.4, rel_tol=5e-4), f"GLSZM_GLN = {glszm['GLSZM_GLN'][0]}"
assert math.isclose(features["GLDZM_GLNU"][0], 3.4, rel_tol=5e-4), f"GLDZM_GLNU = {features['GLDZM_GLNU'][0]}"


# Distance next to background: a 5x5 ROI whose centre pixel is outside the mask. The only pixel of
# level 2 touches that hole; IBSI gives its zone distance 1, Nyxus gives 2.
hole_img = np.ones((5, 5), dtype=np.uint32)
hole_img[2, 1] = 2
hole_mask = np.ones_like(hole_img)
hole_mask[2, 2] = 0
f = gldzm(hole_img, hole_mask, ibsi=True)
quoted = {"SDE": 0.625, "LDE": 2.5, "ZDM": 1.5, "ZP": 0.08333}
for name, expected in quoted.items():
    got = f[f"GLDZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"hole GLDZM_{name} = {got}, page says {expected}"

# The same ROI with the default binning: the hole becomes a zone of grey level 1 at distance 3
f = gldzm(hole_img * 10, hole_mask)
quoted = {"ZP": 0.125, "ZDM": 2.0}
for name, expected in quoted.items():
    got = f[f"GLDZM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"default hole GLDZM_{name} = {got}, page says {expected}"


# Constant ROI: every feature is the missing-value substitute (0 by default)
const = np.full((4, 4), 7, dtype=np.uint32)
f = gldzm(const, np.ones_like(const), ibsi=True)
assert all(f[f"GLDZM_{c}"][0] == 0 for c in ("SDE", "ZDE", "GLV", "ZP")), "constant ROI is not the substitute"
