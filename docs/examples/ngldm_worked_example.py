# Example for docs/source/Math/f_ngldm.rst: NGLDM features of the page's 5x5 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, in IBSI mode and with the default settings, so the page cannot drift from the code.
# The values for a negative or zero --coarseGrayDepth are not: they are command-line only.
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

nyx = Nyxus(["*ALL_NGLDM*"], ibsi=True)
features = nyx.featurize(img, mask)
print(features[["NGLDM_LDE", "NGLDM_HDE", "NGLDM_GLNU", "NGLDM_DCENT"]])
# example-end

# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["ngldm"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"

# Values the page quotes for the default settings (64 levels, grey level floor(64 x / 5))
default = Nyxus(["*ALL_NGLDM*"]).featurize(img, mask)
quoted = {"GLM": 35.4, "GLV": 243.36, "HGLCE": 1496.5, "LGLCE": 0.00182, "LDE": 0.3725, "HDHGLE": 5608.0}
for name, expected in quoted.items():
    got = default[f"NGLDM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"default NGLDM_{name} = {got}, page says {expected}"

# Values the page quotes for a mask that leaves three pixels of the bounding box out of the ROI:
# the same in both modes, because pixels outside the mask are never counted
holes = np.ones_like(img)
holes[0, 0] = holes[0, 1] = holes[4, 4] = 0
for ibsi in (True, False):
    masked = Nyxus(["*ALL_NGLDM*"], ibsi=ibsi).featurize(img, holes)
    for name, expected in {"LDE": 0.3273, "HDE": 5.818}.items():
        got = masked[f"NGLDM_{name}"][0]
        assert math.isclose(got, expected, rel_tol=5e-4), f"masked (ibsi={ibsi}) NGLDM_{name} = {got}, page says {expected}"

# Grey level 0: one ROI pixel of intensity 0 in IBSI mode, and one pixel below x_max / 64 with
# the default binning. Its pixels count in N_s but not in the low-grey-level numerators.
zero = img.copy()
zero[2, 2] = 0
got = Nyxus(["*ALL_NGLDM*"], ibsi=True).featurize(zero, mask)["NGLDM_LGLCE"][0]
assert math.isclose(got, 0.2323, rel_tol=5e-4), f"in-mask zero NGLDM_LGLCE = {got}, page says 0.2323"
# leaving the level-0 pixel out of N_s as well would scale it by 25 / 24
assert math.isclose(got * 25 / 24, 0.2420, rel_tol=5e-4)
low = img * 20
low[0, 0] = 1
got = Nyxus(["*ALL_NGLDM*"]).featurize(low, mask)["NGLDM_LGLCE"][0]
assert math.isclose(got, 0.00181, rel_tol=5e-4), f"low-intensity NGLDM_LGLCE = {got}, page says 0.00181"
assert math.isclose(got * 25 / 24, 0.00189, rel_tol=5e-3)

# The two grey-level conversions of the worked example's intensities 1 to 5 at the default 64
assert [64 * x // 5 for x in range(1, 6)] == [12, 25, 38, 51, 64]
assert [min(64 * x // 5 + 1, 64) for x in range(1, 6)] == [13, 26, 39, 52, 64]

# Relation to GLDM quoted on the page: same matrix in IBSI mode, entropy differs by the GLDM log2
gldm = Nyxus(["*ALL_GLDM*"], ibsi=True).featurize(img, mask)
assert math.isclose(gldm["GLDM_SDE"][0], features["NGLDM_LDE"][0], rel_tol=1e-12)
assert math.isclose(gldm["GLDM_DE"][0], 3.380, rel_tol=5e-4), f"GLDM_DE = {gldm['GLDM_DE'][0]}, page says 3.380"
