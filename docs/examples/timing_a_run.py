# Example for docs/source/troubleshooting.rst: timing feature groups and thread counts.
# The page shows the lines between the "example" markers. Everything else creates input data
# and checks the result, so tests/python/test_2d_docs_examples_mechanics.py can run the
# script as written. The timings depend on the machine and are not checked.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(0)
os.makedirs("data/int", exist_ok=True)
os.makedirs("data/seg", exist_ok=True)
mask = np.zeros((512, 512), dtype=np.uint16)
yy, xx = np.ogrid[:100, :100]
disk = (yy - 50) ** 2 + (xx - 50) ** 2 < 40 ** 2
for k in range(16):
    r, c = divmod(k, 4)
    mask[r * 128 + 10:r * 128 + 110, c * 128 + 10:c * 128 + 110][disk] = k + 1
tifffile.imwrite("data/int/cells.tif", rng.integers(100, 4000, (512, 512), dtype=np.uint16))
tifffile.imwrite("data/seg/cells.tif", mask)

# example-start
import time

from nyxus import Nyxus

for group in ("*ALL_INTENSITY*", "*ALL_MORPHOLOGY*"):
    for threads in (1, 4):
        nyx = Nyxus([group], n_feature_calc_threads=threads)
        start = time.perf_counter()
        features = nyx.featurize_directory("data/int", "data/seg")
        seconds = time.perf_counter() - start
        print(f"{group:18} {threads} threads: {seconds:.2f} s for {len(features)} ROIs")
# example-end

assert len(features) == 16
assert nyx.get_params("n_feature_calc_threads")["n_feature_calc_threads"] == 4
assert "EROSIONS_2_VANISH" in features.columns
