"""Mechanics of the background a 3D texture family skips: the cells of the ROI's bounding box that
are not the ROI's.

Under MATLAB-style binning -- a positive grey depth, 64 by default -- intensity 0 bins to level 1,
the level the ROI's darkest voxels bin to as well, so the binned value cannot tell the background
from the ROI. 3D GLSZM, GLDM and GLRLM tell them apart by the ROI's voxel cloud, in RAM and out of
core, and these tests assert it.

The reference is the same ROI with its MATLAB levels written into the image and no binning. There
the background is intensity 0 and every ROI voxel is level 1 or above, so nothing is ambiguous. A
binned run that skips a ROI voxel at level 1, or counts a background cell as one, parts from it.

Kind: *mechanics* per tests/vetting/SPEC.md 2. Each case compares two routes to one configuration
and claims no oracle of its own.
"""
import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

# the in-RAM side's ram_limit; see test_ooc_mechanics.py for why it is not larger
RAM_LIMIT_LARGE_MB = 1000

# the default coarse grey depth, which the binned runs leave unset so that the default is what runs
DEFAULT_DEPTH = 64

FAMILIES = {"3GLSZM_": "*3D_GLSZM*", "3GLDM_": "*3D_GLDM*", "3GLRLM_": "*3D_GLRLM*"}

# 8 x 90 x 90 voxels: a footprint well over 1 MB, so ram_limit=1 streams the ROI out of core
Z, Y, X = 8, 90, 90


def _volume():
    """Intensities 1..200 over the whole volume, every one of them on many voxels."""
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    return (1 + (x * 7 + y * 13 + z * 29) % 200).astype(np.uint32)


def _ellipsoid():
    """A ROI that leaves background inside its bounding box."""
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    inside = (((z - (Z - 1) / 2) / (Z / 2)) ** 2 + ((y - (Y - 1) / 2) / (Y / 3)) ** 2
              + ((x - (X - 1) / 2) / (X / 3)) ** 2) <= 1.0
    return inside.astype(np.uint32)


def _box():
    """A ROI that fills its bounding box: the box is 1 voxel in from every face."""
    mask = np.zeros((Z, Y, X), np.uint32)
    mask[1:-1, 1:-1, 1:-1] = 1
    return mask


def _matlab_levels(inten, mask, n):
    """The MATLAB levels of the ROI's voxels, as bin_pixel_matlab() computes them: the scale runs
    from 0 to the ROI's maximum, and level = floor(n x / max + 1) clipped to 1..n."""
    mx = float(inten[mask > 0].max())
    lv = np.floor((n / mx) * inten.astype(np.float64) + 1.0)
    return np.clip(lv, 1, n).astype(np.uint32)


def _write(tmp_path, name, inten, mask):
    ip, mp = tmp_path / (name + "_i.ome.tif"), tmp_path / (name + "_m.ome.tif")
    tifffile.imwrite(str(ip), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(mp), mask, metadata={"axes": "ZYX"})
    return str(ip), str(mp)


def _row(groups, pair, ram_limit, **kw):
    df = nyxus.Nyxus3D(groups, ram_limit=ram_limit, **kw).featurize_files([pair[0]], [pair[1]], False)
    assert len(df) == 1
    return df.iloc[0]


def _assert_binned_equals_prebinned(tmp_path, mask):
    inten = np.where(mask > 0, _volume(), 0).astype(np.uint32)
    levels = _matlab_levels(inten, mask, DEFAULT_DEPTH)
    prebinned = np.where(mask > 0, levels, 0).astype(np.uint32)

    # every level 1..n is populated, so the reference's levels 1..max are the binned run's levels
    # and the two number their matrix rows alike
    assert set(np.unique(levels[mask > 0])) == set(range(1, DEFAULT_DEPTH + 1))

    binned = _write(tmp_path, "binned", inten, mask)
    reference = _write(tmp_path, "prebinned", prebinned, mask)
    groups = list(FAMILIES.values())

    want = _row(groups, reference, RAM_LIMIT_LARGE_MB, coarse_gray_depth=0)
    for side, limit in (("in RAM", RAM_LIMIT_LARGE_MB), ("out of core", 1)):
        got = _row(groups, binned, limit)
        for prefix in FAMILIES:
            cols = [c for c in want.index if c.startswith(prefix)]
            assert cols, prefix
            bad = [(c, got[c], want[c]) for c in cols
                   if abs(got[c] - want[c]) > 1e-9 * max(abs(want[c]), 1.0)]
            assert not bad, "%s %s: binned run parts from the prebinned one: %r" % (side, prefix, bad[:6])


def _box_cells(mask):
    zz, yy, xx = mask.nonzero()
    return (np.ptp(zz) + 1) * (np.ptp(yy) + 1) * (np.ptp(xx) + 1)


# A ROI that fills its bounding box has no background to tell apart, so this isolates the ROI's own
# level-1 voxels: a family that skips level 1 as background loses them and parts from the reference.
def test_3d_texture_background_level1_voxels_kept_mechanics(tmp_path):
    mask = _box()
    assert _box_cells(mask) == mask.sum(), "the fixture must fill its bounding box"
    _assert_binned_equals_prebinned(tmp_path, mask)


# The ellipsoid adds background inside its box. A family that skips only level 0 counts the box's
# corners as level-1 voxels, and one that skips level 1 loses the ROI's darkest voxels; either parts
# from the reference.
def test_3d_texture_background_by_roi_mask_mechanics(tmp_path):
    mask = _ellipsoid()
    assert _box_cells(mask) > mask.sum(), "the fixture must leave background inside its bounding box"
    _assert_binned_equals_prebinned(tmp_path, mask)
