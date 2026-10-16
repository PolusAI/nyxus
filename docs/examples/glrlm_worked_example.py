# Example for docs/source/Math/f_glrlm.rst: GLRLM features of the page's 5x5 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, in IBSI mode and with the default settings, so the page cannot drift from the code.
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

nyx = Nyxus(["*ALL_GLRLM*"], ibsi=True)
features = nyx.featurize(img, mask)
print(features[["GLRLM_SRE_0", "GLRLM_LRE_0", "GLRLM_RP_0", "GLRLM_SRE_AVE"]])
# example-end

# Every 0-degree value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["glrlm"]
for code, facts in extras.items():
    got = features[f"{code}_0"][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code}_0 = {got}, page says {facts['example']}"

# Values the page quotes for the default settings (64-level binning)
default = Nyxus(["*ALL_GLRLM*"]).featurize(img, mask)
quoted = {"SRE": 0.7549, "RP": 0.68, "GLV": 249.44, "HGLRE": 1756.7, "LGLRE": 0.001363}
for name, expected in quoted.items():
    got = default[f"GLRLM_{name}_0"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"default GLRLM_{name}_0 = {got}, page says {expected}"

# Values the page quotes for a mask that leaves three pixels of the bounding box out of the ROI
holes = np.ones_like(img)
holes[0, 0] = holes[0, 1] = holes[4, 4] = 0
masked_default = Nyxus(["*ALL_GLRLM*"]).featurize(img, holes)
masked_ibsi = Nyxus(["*ALL_GLRLM*"], ibsi=True).featurize(img, holes)
quoted = [(masked_default, "RP", 0.7273), (masked_default, "LGLRE", 0.1262),
          (masked_ibsi, "RP", 0.6364)]
for table, name, expected in quoted:
    got = table[f"GLRLM_{name}_0"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"masked GLRLM_{name}_0 = {got}, page says {expected}"

# The masked values with those pixels excluded: 0-degree runs of the binned ROI pixels only
binned = np.minimum(np.floor(64 * img.astype(int) / 5 + 1), 64).astype(int)
runs = []
for r in range(5):
    c = 0
    while c < 5:
        if not holes[r, c]:
            c += 1
            continue
        length = 1
        while c + length < 5 and holes[r, c + length] and binned[r, c + length] == binned[r, c]:
            length += 1
        runs.append(binned[r, c])
        c += length
assert math.isclose(len(runs) / 22, 0.6364, rel_tol=5e-4), "excluded run percentage"
assert math.isclose(sum(1 / g**2 for g in runs) / len(runs), 0.001426, rel_tol=5e-4), "excluded LGLRE"

# Run entropy: the approximate logarithm against an exact one on the page's matrix P
P = np.array([[1, 0, 1], [3, 0, 1], [4, 1, 1], [1, 1, 0], [3, 0, 0]], dtype=float)
p = P[P > 0] / P.sum()
exact = float(-(p * np.log2(p)).sum())
assert math.isclose(exact, 3.0575, rel_tol=5e-5), f"exact run entropy = {exact}, page says 3.0575"
assert math.isclose(features["GLRLM_RE_0"][0], 3.0582, rel_tol=5e-5), "GLRLM_RE_0 differs from the page"


def glrlm(image, **kw):
    image = np.asarray(image, dtype=np.uint32)
    return Nyxus(["*ALL_GLRLM*"], **kw).featurize(image, np.ones_like(image))


# 32-bit overflow in IBSI mode: one run of 8 pixels above eight runs of 1 pixel at levels 1 and 2
f = glrlm([[10000] * 8, [1, 2] * 4], ibsi=True)
assert math.isclose(f["GLRLM_LRHGLE_0"][0], 2.339e8, rel_tol=5e-4), "wrapped LRHGLE"
assert math.isclose((10000**2 * 64 + 4 * 1 + 4 * 4) / 9, 7.111e8, rel_tol=5e-4), "true LRHGLE"
assert glrlm([[8192] * 8, [1, 2] * 4], ibsi=True)["GLRLM_SRLGLE_0"][0] == 0, "SRLGLE at 8192"
assert math.isclose((1 / (8192**2 * 64) + 4 + 4 / 4) / 9, 0.5556, rel_tol=5e-4), "true SRLGLE"
assert glrlm([[65535, 7, 7, 3]], ibsi=True)["GLRLM_SRLGLE_0"][0] == 0, "SRLGLE with maximum 65535"
f = glrlm([[70000, 7, 8, 3]], ibsi=True)
assert f["GLRLM_LGLRE_0"][0] == 0 and f["GLRLM_LRLGLE_0"][0] == 0, "N_g above 65536"

# Constant ROI: every feature, the _AVE columns included, is the missing-value substitute
f = glrlm(np.full((4, 4), 7))
assert all(f[c][0] == 0 for c in f.columns if c.startswith("GLRLM_")), "constant ROI is not the substitute"

# Run entropy over the four directions: rows for 0, down-right for 45, columns for 90, down-left for 135
def exact_run_entropy(dy, dx):
    seen, runs = np.zeros(img.shape, bool), []
    for r in range(5):
        for c in range(5):
            if seen[r, c]:
                continue
            length, rr, cc = 0, r, c
            while 0 <= rr < 5 and 0 <= cc < 5 and img[rr, cc] == img[r, c] and not seen[rr, cc]:
                seen[rr, cc] = True
                length, rr, cc = length + 1, rr + dy, cc + dx
            runs.append((img[r, c], length))
    _, counts = np.unique(np.array(runs), axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

rel = [abs(features[f"GLRLM_RE_{a}"][0] / exact_run_entropy(dy, dx) - 1)
       for a, (dy, dx) in zip((0, 45, 90, 135), ((0, 1), (1, 1), (1, 0), (1, -1)))]
assert 2e-4 <= min(rel) and max(rel) <= 1.2e-3, f"run entropy relative differences {rel}"