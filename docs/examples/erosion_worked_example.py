# Example for docs/source/Math/f_erosion.rst: erosion features of the page's irregular 8x9 ROI.
# The page shows the lines between the "example" markers. The assertions below pin every Nyxus
# value the page quotes, so the page cannot drift from the code.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
from nyxus import Nyxus

mask = np.array([[0, 0, 1, 1, 1, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 0, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1, 1, 0, 0],
                 [0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.uint32)
img = np.arange(1, mask.size + 1, dtype=np.uint32).reshape(mask.shape)  # any non-constant values

nyx = Nyxus(["EROSIONS_2_VANISH", "EROSIONS_2_VANISH_COMPLEMENT"])
features = nyx.featurize(img, mask)
print(features[["EROSIONS_2_VANISH", "EROSIONS_2_VANISH_COMPLEMENT"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["erosion"]
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code} = {got}, page says {facts['example']}"


def erosions(m, values=None):
    """EROSIONS_2_VANISH of mask `m` placed with a background margin of 3 pixels."""
    m = np.pad(np.asarray(m, dtype=np.uint32), 3)
    if values is None:
        values = np.arange(1, m.size + 1, dtype=np.uint32).reshape(m.shape)
    f = Nyxus(["EROSIONS_2_VANISH", "EROSIONS_2_VANISH_COMPLEMENT"]).featurize(values * m, m)
    assert f["EROSIONS_2_VANISH_COMPLEMENT"][0] == 0
    return f["EROSIONS_2_VANISH"][0]


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint32)


# Values the page quotes
assert erosions(mask) == 2
# pins known defect https://github.com/PolusAI/nyxus/issues/501
assert erosions(np.fliplr(mask)) == 3, "mirrored worked example"
assert erosions(np.rot90(mask, 2)) == 3, "worked example rotated by 180 degrees"
# pins known defect https://github.com/PolusAI/nyxus/issues/501
assert erosions(np.ones((7, 9))) == 1000, "rectangle filling its box"
notched = np.ones((7, 9)); notched[0, 0] = 0
assert erosions(notched) == 1000, "rectangle less one corner pixel"
assert erosions(disk(3)) == 3
assert erosions(disk(6)) == 6
assert erosions(disk(10)) == 10
assert erosions(disk(20)) == 21
tee = np.zeros((5, 5)); tee[1:, 1:] = 1; tee[0, 2] = tee[2, 0] = 1   # 4x4 square, two pixels sticking out
# pins known defect https://github.com/PolusAI/nyxus/issues/501
assert erosions(tee) == 1000, "pixels sticking out at the top and left"
assert erosions(np.rot90(tee, 2)) == 2, "pixels sticking out at the bottom and right"
for h, w in ((1, 1), (2, 5), (3, 3), (3, 9)):
    assert erosions(np.ones((h, w))) == 0, f"{h}x{w} rectangle"
for h, w in ((4, 4), (4, 9), (5, 5)):
    assert erosions(np.ones((h, w))) == 1000, f"{h}x{w} rectangle"

# A ROI of one intensity reports 0, whatever its shape
for m in (mask, disk(6), np.ones((7, 9))):
    # pins known defect https://github.com/PolusAI/nyxus/issues/502
    assert erosions(m, values=np.full(np.pad(m, 3).shape, 5, dtype=np.uint32)) == 0, "constant ROI"

# Whole-slide mode: the smaller side minus 3, constant images included (files in the working dir)
import tifffile
for (h, w), want in (((10, 10), 7), ((20, 20), 17), ((20, 40), 17), ((3, 9), 0)):
    for const in (True, False):
        img = (np.full((h, w), 5, dtype=np.uint16) if const
               else np.arange(1, h * w + 1, dtype=np.uint16).reshape(h, w))
        path = f"./ero_slide_{h}x{w}.tif"
        tifffile.imwrite(path, img)
        got = Nyxus(["EROSIONS_2_VANISH"]).featurize_files([path], [path], single_roi=True)
        # pins known defect https://github.com/PolusAI/nyxus/issues/493
        assert got["EROSIONS_2_VANISH"][0] == want, f"whole-slide {h}x{w}, constant={const}"
