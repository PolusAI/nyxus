# Example for docs/source/Math/f_gldm.rst: GLDM features of the page's 5x5 worked example.
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

nyx = Nyxus(["*ALL_GLDM*"], ibsi=True)
features = nyx.featurize(img, mask)
print(features[["GLDM_SDE", "GLDM_LDE", "GLDM_DN", "GLDM_GLV"]])
# example-end

# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["gldm"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"

# Values the page quotes for the default settings (64-level binning)
default = Nyxus(["*ALL_GLDM*"]).featurize(img, mask)
quoted = {"SDE": 0.3725, "GLV": 236.6, "HGLE": 1552.8, "LGLE": 0.001612}
for name, expected in quoted.items():
    got = default[f"GLDM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"default GLDM_{name} = {got}, page says {expected}"

# Values the page quotes for a mask that leaves three pixels of the bounding box out of the ROI
holes = np.ones_like(img)
holes[0, 0] = holes[0, 1] = holes[4, 4] = 0
masked_default = Nyxus(["*ALL_GLDM*"]).featurize(img, holes)
masked_ibsi = Nyxus(["*ALL_GLDM*"], ibsi=True).featurize(img, holes)
quoted = [(masked_default, "SDE", 0.3273), (masked_ibsi, "SDE", 0.3273),
          (masked_default, "LDE", 5.818), (masked_ibsi, "LDE", 5.818)]
for table, name, expected in quoted:
    got = table[f"GLDM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"masked GLDM_{name} = {got}, page says {expected}"

# Values the page quotes for a pixel of intensity 0 inside the mask (IBSI mode)
zero = img.copy()
zero[2, 2] = 0
zero_ibsi = Nyxus(["*ALL_GLDM*"], ibsi=True).featurize(zero, mask)
for name, expected in {"SDE": 0.4282, "LDE": 4.583}.items():
    got = zero_ibsi[f"GLDM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"zero-pixel GLDM_{name} = {got}, page says {expected}"

# Values the page quotes for intensities 1001 to 1005, which the default binning merges into one level
narrow = Nyxus(["*ALL_GLDM*"]).featurize(img + 1000, mask)
for name, expected in {"SDE": 0.02778, "LDE": 49.0, "GLN": 25.0}.items():
    got = narrow[f"GLDM_{name}"][0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"narrow-range GLDM_{name} = {got}, page says {expected}"
