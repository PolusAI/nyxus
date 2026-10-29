# Example for docs/source/Math/f_intensity_3d.rst: 3D intensity features of the page's worked
# example, a 22-voxel ROI over three slices. The page shows the lines between the "example"
# markers. The assertions below pin the worked-example values the page quotes, so the page cannot
# drift from the code; the page's Validation section names the values that were measured once and
# are not pinned.
import json
import math
import os
from pathlib import Path

# example-start
import numpy as np
import tifffile
from nyxus import Nyxus3D

# a 3 x 4 x 4 volume (Z, Y, X)
img = np.array([[[1, 2, 2, 3], [2, 4, 5, 3], [3, 5, 6, 4], [0, 3, 4, 2]],
                [[2, 3, 4, 1], [3, 6, 6, 5], [4, 6, 5, 3], [2, 4, 3, 1]],
                [[1, 1, 2, 0], [2, 3, 4, 2], [3, 4, 5, 3], [1, 2, 2, 0]]], dtype=np.uint16)

# ROI 1: 4 voxels in slice 0, 12 in slice 1, 6 in slice 2
seg = np.zeros_like(img)
seg[0, 1:3, 1:3] = 1
seg[1] = 1
seg[1, [0, 0, 3, 3], [0, 3, 0, 3]] = 0
seg[2, 1:3, 1:4] = 1


def save(name, volume):
    # minisblack keeps a last axis of 3 or 4 from being read as colour samples
    path = os.path.abspath(name)
    tifffile.imwrite(path, volume, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})
    return path


nyx = Nyxus3D(["*3D_ALL_INTENSITY*"])
features = nyx.featurize_files([save("int.ome.tif", img)], [save("seg.ome.tif", seg)], False)
print(features[["3MEAN", "3MEDIAN", "3MODE", "3P10", "3P90", "3ROBUST_MEAN", "3ENTROPY"]])
# example-end

# Every value in the page's feature index, as recorded in docs/feature_extras.toml
extras = json.loads(Path(os.environ["NYXUS_DOCS_EXTRAS"]).read_text(encoding="utf-8"))["intensity_3d"]
assert len(extras) == 36, len(extras)
for code, facts in extras.items():
    got = features[code][0]
    assert math.isclose(got, facts["example"], rel_tol=5e-4, abs_tol=1e-12), \
        f"{code} = {got}, page says {facts['example']}"


def run(volume, mask, tag, **kwargs):
    return Nyxus3D(["*3D_ALL_INTENSITY*"], **kwargs).featurize_files(
        [save(f"int_{tag}.ome.tif", volume)], [save(f"seg_{tag}.ome.tif", mask)], False)


def check(row, quoted, label):
    for name, expected in quoted.items():
        got = row[name]
        assert math.isclose(got, expected, rel_tol=5e-4, abs_tol=1e-12), \
            f"{label} {name} = {got}, page says {expected}"


one = features.iloc[0]

# The voxel count and the values the page derives by hand
assert int((seg == 1).sum()) == 22
check(one, {"3INTEGRATED_INTENSITY": 93, "3MEAN": 4.22727, "3MEDIAN": 4.0, "3MODE": 3.0,
            "3MIN": 2.0, "3VARIANCE": 1.42208, "3P01": 2.21, "3P10": 3.0, "3P90": 6.0,
            "3ROBUST_MEAN": 4.33333, "3ENTROPY": 2.15810, "3COVERED_IMAGE_INTENSITY_RANGE": 1.0},
      "worked example")

# The percentiles are numpy's default linear rule, and the robust mean keeps [P10, P90]
x = img[seg == 1].astype(float)
for q in (1, 10, 25, 75, 90, 99):
    assert math.isclose(one[f"3P{q:02d}"], float(np.percentile(x, q)), rel_tol=1e-12), q
assert math.isclose(x[(x >= 3.0) & (x <= 6.0)].mean(), one["3ROBUST_MEAN"], rel_tol=1e-12)
# MATLAB's default prctile, numpy's 'hazen' rule, agrees on every percentile but 3P01
hazen = [float(np.percentile(x, q, method="hazen")) for q in (1, 10, 25, 75, 90, 99)]
assert hazen == [2, 3, 3, 5, 6, 6], hazen
assert np.median(np.abs(x - np.median(x))) == 1.0      # MEDIAN_ABSOLUTE_DEVIATION is 0.9545

# Four coarse bins instead of 64: only entropy and uniformity change
four = run(img, seg, "four", coarse_gray_depth=4).iloc[0]
check(four, {"3ENTROPY": 1.75266, "3UNIFORMITY": 0.318182, "3P10": 3.0, "3MEAN": 4.22727}, "4 bins")

# IBSI mode changes nothing
ibsi = run(img, seg, "ibsi", ibsi=True).iloc[0]
for code in extras:
    assert ibsi[code] == one[code], code

# Covered range with two ROIs: the denominator is the range of all ROI voxels; a voxel of 40
# outside every ROI is not used
img2, seg2 = img.copy(), seg.copy()
img2[0, 0, 0] = 40
seg2[0, 3, 0:2] = 2                       # ROI 2 holds intensities 0 and 3
two = run(img2, seg2, "two")
covered = dict(zip(two["ROI_label"], two["3COVERED_IMAGE_INTENSITY_RANGE"]))
assert math.isclose(covered[1], 4 / 6) and math.isclose(covered[2], 3 / 6), covered

# A constant ROI
flat = run(np.full_like(img, 7), seg, "flat").iloc[0]
check(flat, {"3MEAN": 7, "3MODE": 7, "3P01": 7, "3P99": 7, "3ROBUST_MEAN": 7, "3UNIFORMITY": 1.0,
             "3UNIFORMITY_PIU": 100.0, "3STANDARD_DEVIATION": 0.0, "3COV": 0.0, "3SKEWNESS": 0.0,
             "3KURTOSIS": 0.0, "3QCOD": 0.0, "3COVERED_IMAGE_INTENSITY_RANGE": 0.0}, "constant")
assert abs(flat["3ENTROPY"]) < 1e-15

# Small ROIs: one voxel in each of three slices
small = np.zeros((3, 5, 5), np.uint16)
small[:, 2, 2] = [1, 2, 4]
check(run(small, (small > 0).astype(np.uint16), "small").iloc[0],
      {"3SKEWNESS": 0.0, "3KURTOSIS": 0.0, "3EXCESS_KURTOSIS": 0.0, "3HYPERSKEWNESS": 0.3464,
       "3HYPERFLATNESS": 0.7099}, "3 voxels")

# A voxel of intensity 0 inside the mask is a ROI voxel
zero = img.copy()
zero[1, 1, 1] = 0
check(run(zero, seg, "zero").iloc[0], {"3MIN": 0.0, "3MEAN": 87 / 22}, "zero in mask")

# A signed volume read from a file is reported in its own units
neg = run(img.astype(np.int16) - 10, seg, "neg").iloc[0]
check(neg, {"3MIN": -8.0, "3MEAN": -5.77273, "3VARIANCE": 1.42208, "3UNIFORMITY_PIU": 133.333},
      "signed")

# Intensities above 65535: each voxel's square wraps modulo 2^32 before it is summed
big = run(img.astype(np.uint32) * 20000, seg, "big").iloc[0]
# pins known defect https://github.com/PolusAI/nyxus/issues/490
check(big, {"3MEAN": 84545.45, "3ENERGY": 48940915712.0, "3ROOT_MEAN_SQUARED": 47165.53}, "x 20000")
assert math.isclose(float((x * 20000) @ (x * 20000)), 1.692e11)

# IBSI's coefficient of variation divides sigma_b by the mean
assert round(one["3STANDARD_DEVIATION_BIASED"] / one["3MEAN"], 4) == 0.2756
# The percentiles and quartiles the page quotes
check(one, {"3P01": 2.21, "3P25": 3.0, "3P75": 5.0, "3P99": 6.0, "3INTERQUARTILE_RANGE": 2.0,
            "3QCOD": 0.25, "3ROBUST_MEAN_ABSOLUTE_DEVIATION": 0.952381}, "percentiles")
