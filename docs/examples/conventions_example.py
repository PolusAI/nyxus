# Example for docs/source/conventions.rst: three conventions on the 5x5 ROI of the texture pages.
# The page shows the lines between the "example" markers. The assertions below pin every value the
# page quotes, so the page cannot drift from the code.
import math

# example-start
import numpy as np
from nyxus import Nyxus

img = np.array([[5, 2, 5, 4, 4],
                [3, 3, 3, 1, 3],
                [2, 1, 1, 1, 3],
                [4, 2, 2, 2, 3],
                [3, 5, 3, 3, 2]], dtype=np.uint32)
mask = np.ones_like(img)
zero_inside = img.copy()
zero_inside[2, 2] = 0                 # a ROI pixel of intensity 0
constant = np.full_like(img, 7)       # a ROI of one intensity

codes = ["MEAN", "GLDM_SDE", "GLRLM_HGLRE_AVE"]
results = {}
for ibsi in (False, True):
    nyx = Nyxus(codes, ibsi=ibsi)
    for name, roi in [("example", img), ("zero inside", zero_inside), ("constant", constant)]:
        f = nyx.featurize(roi, mask)
        results[ibsi, name] = [float(f[c][0]) for c in codes]
        print(f"ibsi={ibsi!s:5}  {name:11}", *(f"{v:10.4f}" for v in results[ibsi, name]))
# example-end

# Every value the page quotes: (ibsi, roi) -> MEAN, GLDM_SDE, GLRLM_HGLRE_AVE
quoted = {
    (False, "example"):     (2.8, 0.3725, 1654.7051),
    (False, "zero inside"): (2.76, 0.4282, 1590.1715),
    (False, "constant"):    (7.0, 0.0, 0.0),
    (True, "example"):      (2.8, 0.3725, 9.9006),
    (True, "zero inside"):  (2.76, 0.4282, 9.9709),
    (True, "constant"):     (7.0, 0.0, 0.0),
}
for key, expected in quoted.items():
    for code, got, want in zip(codes, results[key], expected):
        assert math.isclose(got, want, rel_tol=5e-4, abs_tol=1e-12), f"{key} {code} = {got}, page says {want}"

# The zero inside the mask is a ROI pixel for MEAN (69 / 25) and is left out by GLDM in both modes
assert results[False, "zero inside"][0] == 69 / 25
assert results[False, "zero inside"][1] == results[True, "zero inside"][1]
