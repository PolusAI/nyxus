# Example for docs/source/Math/f_ngtdm.rst: NGTDM features of the page's 5x5 worked example.
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

nyx = Nyxus(["*ALL_NGTDM*"], ibsi=True)
features = nyx.featurize(img, mask)
print(features[["NGTDM_COARSENESS", "NGTDM_CONTRAST", "NGTDM_BUSYNESS"]])
# example-end

# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["ngtdm"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4), f"{code} = {got}, page says {facts['example']}"


def check(table, quoted, label):
    for name, expected in quoted.items():
        got = table[f"NGTDM_{name}"][0]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
            f"{label} NGTDM_{name} = {got}, page says {expected}"


# Values the page quotes for the default settings (64-level binning)
default = Nyxus(["*ALL_NGTDM*"]).featurize(img, mask)
check(default, {"COARSENESS": 0.01551, "CONTRAST": 309.1, "BUSYNESS": 0.6357, "COMPLEXITY": 1359.7,
                "STRENGTH": 17.47}, "default")

# Values the page quotes for a mask that leaves three pixels of the bounding box out of the ROI
holes = np.ones_like(img)
holes[0, 0] = holes[0, 1] = holes[4, 4] = 0
check(Nyxus(["*ALL_NGTDM*"]).featurize(img, holes), {"COARSENESS": 0.01324, "CONTRAST": 351.2}, "masked default")
check(Nyxus(["*ALL_NGTDM*"], ibsi=True).featurize(img, holes), {"CONTRAST": 0.1302}, "masked IBSI")

# The same 64-level binning without those three pixels: the example binned by hand to levels 13,
# 26, 39, 52 and 64, with the outside pixels set to 0, in IBSI mode so that 0 stays background
binned = np.minimum(64 * img // 5 + 1, 64).astype(np.uint32)
check(Nyxus(["*ALL_NGTDM*"], ibsi=True).featurize(binned * holes, holes),
      {"COARSENESS": 0.01685, "CONTRAST": 277.7}, "masked binned")

# The worked example with one more ROI pixel, of intensity 7, that has no ROI neighbour
wide = np.zeros((5, 7), dtype=np.uint32)
wide[:, :5] = img
wide[2, 6] = 7
lone = Nyxus(["*ALL_NGTDM*"], ibsi=True).featurize(wide, (wide > 0).astype(np.uint32))
check(lone, {"CONTRAST": 0.09734, "COARSENESS": 0.2006, "STRENGTH": 1.373,
             "BUSYNESS": features["NGTDM_BUSYNESS"][0], "COMPLEXITY": features["NGTDM_COMPLEXITY"][0]},
      "isolated pixel")

# A block of 1s plus an isolated 2: no grey-tone difference anywhere, and only one level with p_i > 0
block = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 2]], dtype=np.uint32)
flat = Nyxus(["*ALL_NGTDM*"], ibsi=True).featurize(block, (block > 0).astype(np.uint32))
check(flat, {"COARSENESS": 0.0, "CONTRAST": 0.0, "BUSYNESS": 0.0, "COMPLEXITY": 0.0,
             "STRENGTH": 0.0}, "zero differences")

# A constant ROI
const = np.full((4, 4), 7, dtype=np.uint32)
for ibsi in (True, False):
    check(Nyxus(["*ALL_NGTDM*"], ibsi=ibsi).featurize(const, np.ones_like(const)),
          {"COARSENESS": 0.0, "CONTRAST": 0.0, "BUSYNESS": 0.0, "COMPLEXITY": 0.0, "STRENGTH": 0.0},
          f"constant (ibsi={ibsi})")
