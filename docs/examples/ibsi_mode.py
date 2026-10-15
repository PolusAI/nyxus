# Example for docs/source/ibsi_mode.rst: what IBSI mode changes in the results.
# The page shows the lines between the "example" markers.
import numpy as np

# Two 64 x 64 images with grey levels 1 to 40, each with one square ROI
rng = np.random.default_rng(0)
intens = rng.integers(1, 41, (2, 64, 64)).astype(np.uint32)
seg = np.zeros_like(intens)
seg[:, 8:56, 8:56] = 1

# example-start
from nyxus import Nyxus

default = Nyxus(["*ALL*"]).featurize(intens, seg)
ibsi = Nyxus(["*ALL*"], ibsi=True).featurize(intens, seg)

# IBSI mode adds the intensity histogram family
added = [c for c in ibsi.columns if c not in default.columns]
print(len(default.columns), len(ibsi.columns), added[:3])

# Compare every feature column the two runs share
shared = [c for c in default.columns[9:] if c in ibsi.columns]
changed = [c for c in shared if not np.allclose(default[c], ibsi[c])]
print(sorted({c.split("_")[0] for c in changed}))
print(default.loc[0, ["MEAN", "GLCM_CONTRAST_AVE", "NGTDM_COARSENESS"]].tolist())
print(ibsi.loc[0, ["MEAN", "GLCM_CONTRAST_AVE", "NGTDM_COARSENESS"]].tolist())
# example-end

assert len(added) == 46 and all(c.startswith("IH_") for c in added)
assert sorted({c.split("_")[0] for c in changed}) == ["GLCM", "GLDM", "GLDZM", "GLRLM", "GLSZM", "NGLDM", "NGTDM"]
assert default["MEAN"].equals(ibsi["MEAN"])
assert default.loc[0, "GLCM_CONTRAST_AVE"] > 2 * ibsi.loc[0, "GLCM_CONTRAST_AVE"]
