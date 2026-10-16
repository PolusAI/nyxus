# Example for docs/source/Math/f_ngldm_3d.rst: 3D NGLDM features of the page's 3 x 3 x 3 worked
# example. The page shows the lines between the "example" markers. The assertions below pin every
# value the page quotes, in IBSI mode and with the default settings, so the page cannot drift from
# the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# Three slices of 3 x 3 voxels, (Z, Y, X), grey levels 1 to 4
vol = np.array([[[1, 1, 2],
                 [1, 2, 2],
                 [4, 3, 3]],
                [[1, 1, 2],
                 [3, 2, 2],
                 [4, 4, 3]],
                [[1, 2, 2],
                 [3, 3, 4],
                 [4, 4, 1]]], dtype=np.uint16)
mask = np.ones_like(vol)

tifffile.imwrite("int.ome.tif", vol, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
tifffile.imwrite("seg.ome.tif", mask, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_NGLDM*"], ibsi=True)
features = nyx.featurize_files([os.path.abspath("int.ome.tif")],
                               [os.path.abspath("seg.ome.tif")], False)
print(features[["3NGLDM_LDE", "3NGLDM_HDE", "3NGLDM_GLNU", "3NGLDM_DCENT"]])
# example-end

_n = [0]


def run(v, m, **kw):
    """Featurize volume v with label volume m; files go to the current directory."""
    _n[0] += 1
    a, b = os.path.abspath(f"int{_n[0]}.ome.tif"), os.path.abspath(f"seg{_n[0]}.ome.tif")
    tifffile.imwrite(a, v.astype(np.uint16), ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    tifffile.imwrite(b, m.astype(np.uint16), ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    return Nyxus3D(["*3D_NGLDM*"], **kw).featurize_files([a], [b], False)


def check(df, code, expected, what):
    got = df[code].iloc[0]
    assert math.isclose(got, expected, rel_tol=5e-4), f"{what} {code} = {got}, page says {expected}"


# Every value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["ngldm_3d"]
for code, facts in extras.items():
    check(features, code, facts["example"], "IBSI")

# Values the page quotes for the default settings (64 levels: grey level floor(64 x / 4))
default = run(vol, mask)
for name, expected in {"GLM": 38.52, "GLV": 308.3, "HGLCE": 1792.0, "LGLCE": 0.001453, "LDE": 0.08747}.items():
    check(default, f"3NGLDM_{name}", expected, "default")

# Out of core (ram_limit=0 sends every ROI down the oversized path): the same values
for ibsi in (True, False):
    ooc = run(vol, mask, ibsi=ibsi, ram_limit=0)
    ref = features if ibsi else default
    for code in extras:
        assert math.isclose(ooc[code].iloc[0], ref[code].iloc[0], rel_tol=1e-12, abs_tol=1e-15), \
            f"out-of-core (ibsi={ibsi}) {code} = {ooc[code].iloc[0]}, in memory {ref[code].iloc[0]}"

# A mask that leaves three voxels of the bounding box out of the ROI: the same in both modes,
# because voxels outside the mask are neither centres nor neighbours
holes = mask.copy()
holes[0, 0, 0] = holes[1, 1, 1] = holes[2, 2, 2] = 0
for ibsi in (True, False):
    masked = run(vol, holes, ibsi=ibsi)
    check(masked, "3NGLDM_LDE", 0.06111, f"masked (ibsi={ibsi})")
    check(masked, "3NGLDM_HDE", 20.75, f"masked (ibsi={ibsi})")

# The same ROI inside a larger volume whose other voxels all have intensity 4: those neighbours
# are outside the mask, so every value equals the worked example's
big = np.full((5, 5, 5), 4, dtype=np.uint16)
big[1:4, 1:4, 1:4] = vol
bseg = np.zeros_like(big)
bseg[1:4, 1:4, 1:4] = 1
embedded = run(big, bseg, ibsi=True)
for code in extras:
    assert math.isclose(embedded[code].iloc[0], features[code].iloc[0], rel_tol=1e-12), \
        f"embedded {code} = {embedded[code].iloc[0]}, worked example {features[code].iloc[0]}"

# Grey level 0: one ROI voxel of intensity 0 in IBSI mode, and one voxel below x_max / 64 with the
# default binning. Its voxels count in N_s but not in the low-grey-level numerators.
zero = vol.copy()
zero[1, 1, 1] = 0
check(run(zero, mask, ibsi=True), "3NGLDM_LGLCE", 0.3627, "in-mask zero")
low = vol * 20
low[0, 0, 0] = 1
check(run(low, mask), "3NGLDM_LGLCE", 0.001308, "low-intensity")

# Constant ROI: every feature is the missing-value substitute, 0 by default
const = run(np.full_like(vol, 7), mask, ibsi=True)
for code in extras:
    assert const[code].iloc[0] == 0, f"constant ROI {code} = {const[code].iloc[0]}"

# The 3D family has no metaparameter of its own
try:
    Nyxus3D(["*3D_NGLDM*"]).set_metaparam("3ngldm/greydepth=8")
except Exception:
    pass
else:
    raise AssertionError("3ngldm/greydepth was accepted")

# Relation quoted on the page: 3GLDM_SDE is the same as 3NGLDM_LDE on the worked example
gldm_a, gldm_b = os.path.abspath("int.ome.tif"), os.path.abspath("seg.ome.tif")
gldm = Nyxus3D(["*3D_GLDM*"], ibsi=True).featurize_files([gldm_a], [gldm_b], False)
assert math.isclose(gldm["3GLDM_SDE"].iloc[0], features["3NGLDM_LDE"].iloc[0], rel_tol=1e-12)
check(gldm, "3GLDM_DE", 3.557, "3D GLDM")
