# Example for docs/source/Math/f_glcm_3d.rst: 3D GLCM features of the page's 3 x 4 x 4 worked example.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, in IBSI mode and with the default settings, so the page cannot drift from the code.
# The script writes its OME-TIFF files into the current directory.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# 3 planes of 4 x 4 voxels, grey levels 1 to 4, indexed [z, y, x]
roi = np.array([[[1, 2, 4, 3], [2, 1, 3, 4], [4, 4, 1, 2], [3, 1, 2, 2]],
                [[2, 3, 1, 1], [4, 2, 2, 3], [1, 3, 4, 4], [2, 2, 3, 1]],
                [[3, 1, 2, 4], [1, 1, 4, 2], [2, 4, 3, 3], [4, 3, 1, 2]]], dtype=np.uint16)

# place it in a 5 x 6 x 6 volume with a one-voxel background border
img = np.zeros((5, 6, 6), dtype=np.uint16)
seg = np.zeros_like(img)
img[1:4, 1:5, 1:5] = roi
seg[1:4, 1:5, 1:5] = 1
tifffile.imwrite("int.ome.tif", img, ome=True, metadata={"axes": "ZYX"})
tifffile.imwrite("seg.ome.tif", seg, ome=True, metadata={"axes": "ZYX"})

nyx = Nyxus3D(["*3D_GLCM*"], ibsi=True)
features = nyx.featurize_files([os.path.abspath("int.ome.tif")],
                               [os.path.abspath("seg.ome.tif")], False)
print(features[["3GLCM_ASM", "3GLCM_CONTRAST", "3GLCM_JE", "3GLCM_CONTRAST_AVE"]])
# example-end


def run(intensity, mask, tag, ibsi, metaparams=(), **kwargs):
    i, m = os.path.abspath(f"int_{tag}.ome.tif"), os.path.abspath(f"seg_{tag}.ome.tif")
    tifffile.imwrite(i, intensity, ome=True, metadata={"axes": "ZYX"})
    tifffile.imwrite(m, mask, ome=True, metadata={"axes": "ZYX"})
    n = Nyxus3D(["*3D_GLCM*"], ibsi=ibsi, **kwargs)
    for p in metaparams:
        n.set_metaparam(p)
    return n.featurize_files([i], [m], False)


def check(table, column, expected, what):
    got = table[column][0]
    assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), f"{what} {column} = {got}, page says {expected}"


# Every plain-code value in the page's feature index (IBSI mode), as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["glcm_3d"]
for code, facts in extras.items():
    check(features, code, facts["example"], "IBSI")

# The direction average the page quotes
check(features, "3GLCM_CONTRAST_AVE", 2.5191, "IBSI")
check(features, "3GLCM_CORRELATION_AVE", -0.03872, "IBSI")

# Default settings (64-level binning, ordered pairs)
default = run(img, seg, "default", False)
for column, expected in {"3GLCM_CONTRAST": 478.61, "3GLCM_CONTRAST_AVE": 620.39,
                         "3GLCM_JVAR": 257.66, "3GLCM_VARIANCE": 334.03}.items():
    check(default, column, expected, "default")

# 3glcm/greydepth and coarse_gray_depth both set the number of levels
for table in (run(img, seg, "gd8", False, metaparams=["3glcm/greydepth=8"]),
              run(img, seg, "cgd8", False, coarse_gray_depth=8)):
    check(table, "3GLCM_CONTRAST", 5.7222, "8 levels")

# 3glcm/offset is accepted but the distance stays 1
check(run(img, seg, "off2", False, metaparams=["3glcm/offset=2"]), "3GLCM_CONTRAST", 478.61, "offset=2")

# A mask that leaves three voxels of the bounding box out of the ROI
holes = seg.copy()
holes[1, 1, 1] = holes[1, 1, 2] = holes[3, 4, 4] = 0
check(run(img, holes, "holes_def", False), "3GLCM_CONTRAST", 493.2, "masked default")
check(run(img, holes, "holes_ibsi", True), "3GLCM_CONTRAST", 2.0, "masked IBSI")

# A ROI one plane thick: the (1,1,1) direction has no pairs
thin = np.zeros_like(seg)
thin[2, 1:5, 1:5] = 1
flat = run(img, thin, "thin", True)
check(flat, "3GLCM_CONTRAST", 0.0, "one-plane")
check(flat, "3GLCM_CONTRAST_AVE", 0.67308, "one-plane")
