"""3D GLCM under MATLAB-style binning against the co-occurrence matrix built from its definition, on a
ROI that leaves background inside its bounding box, in RAM and out of core.

MATLAB-style binning -- a positive grey depth, 64 by default -- sends intensity 0 to level 1, the level
the ROI's darkest voxels take too, so the binned value of a bounding-box cell cannot say whether it is
the ROI's. 3D GLCM takes the ROI from its voxel cloud: background cells hold level 0 and pair with
nothing. The reference below counts, for each of the 13 offsets at distance 1, the ordered pairs of ROI
voxels by their MATLAB levels, floor(64 x / max + 1) clipped to 1..64 over the ROI's maximum, and takes
ASM (sum p^2) and CONTRAST (sum (i - j)^2 p) of each normalised matrix, averaged over the 13 offsets.
Both are invariant under transposition, so the order in which a pair is counted does not matter, and
MATLAB-style binning keeps the matrix asymmetric on both sides.

What this discriminates: a family that reads the bounding box's background as level 1 counts every
ROI-background pair and parts from the reference on the ellipsoid. The ROI that fills its bounding box
is the negative case: it has no background to count, so it holds either way.

Kind: *analytic* per tests/vetting/SPEC.md 4: the reference is the definition evaluated by numpy.
"""
import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

RAM_LIMIT_LARGE_MB = 1000   # the in-RAM side, as in test_ooc_mechanics.py
RAM_LIMIT_OOC_MB = 0        # every ROI footprint is >= 0, so every ROI streams out of core
DEPTH = 64                  # the default coarse grey depth, which the run leaves unset

Z, Y, X = 6, 40, 40

# the 13 offsets of one hemisphere of the 26-neighbourhood, (dz, dy, dx)
OFFSETS = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
           if (dz, dy, dx) > (0, 0, 0)]


def _volume():
    """Intensities 1..200 over the whole volume, none of them 0."""
    z, y, x = np.indices((Z, Y, X))
    return (1 + (x * 7 + y * 13 + z * 29) % 200).astype(np.uint32)


def _ellipsoid():
    z, y, x = np.indices((Z, Y, X))
    return ((((z - (Z - 1) / 2) / (Z / 2)) ** 2 + ((y - (Y - 1) / 2) / (Y / 3)) ** 2
             + ((x - (X - 1) / 2) / (X / 3)) ** 2) <= 1.0).astype(np.uint32)


def _box():
    mask = np.zeros((Z, Y, X), np.uint32)
    mask[1:-1, 1:-1, 1:-1] = 1
    return mask


def _reference(inten, mask):
    """-> {"3GLCM_ASM_AVE": ..., "3GLCM_CONTRAST_AVE": ...} from the definition."""
    roi = mask > 0
    mx = float(inten[roi].max())
    lv = np.clip(np.floor(DEPTH / mx * inten.astype(np.float64) + 1.0), 1, DEPTH).astype(np.int64)
    i_idx, j_idx = np.indices((DEPTH, DEPTH))
    asm, con = [], []
    for dz, dy, dx in OFFSETS:
        a = (slice(max(0, -dz), Z - max(0, dz)), slice(max(0, -dy), Y - max(0, dy)), slice(max(0, -dx), X - max(0, dx)))
        b = (slice(max(0, dz), Z - max(0, -dz)), slice(max(0, dy), Y - max(0, -dy)), slice(max(0, dx), X - max(0, -dx)))
        both = roi[a] & roi[b]
        M = np.zeros((DEPTH, DEPTH))
        np.add.at(M, (lv[a][both] - 1, lv[b][both] - 1), 1.0)
        p = M / M.sum()
        asm.append((p ** 2).sum())
        con.append((((i_idx - j_idx) ** 2) * p).sum())
    return {"3GLCM_ASM_AVE": float(np.mean(asm)), "3GLCM_CONTRAST_AVE": float(np.mean(con))}


def _assert_matches_definition(tmp_path, mask):
    inten = np.where(mask > 0, _volume(), 0).astype(np.uint32)
    ip, mp = tmp_path / "i.ome.tif", tmp_path / "m.ome.tif"
    tifffile.imwrite(str(ip), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(mp), mask, metadata={"axes": "ZYX"})
    want = _reference(inten, mask)
    for side, limit in (("in RAM", RAM_LIMIT_LARGE_MB), ("out of core", RAM_LIMIT_OOC_MB)):
        df = nyxus.Nyxus3D(["*3D_GLCM*"], ram_limit=limit).featurize_files([str(ip)], [str(mp)], False)
        assert len(df) == 1
        got = df.iloc[0]
        bad = [(f, float(got[f]), w) for f, w in want.items() if abs(float(got[f]) - w) > 1e-9 * abs(w)]
        assert not bad, "%s: 3D GLCM parts from the definition: %r" % (side, bad)


def _box_cells(mask):
    zz, yy, xx = mask.nonzero()
    return (np.ptp(zz) + 1) * (np.ptp(yy) + 1) * (np.ptp(xx) + 1)


# the ellipsoid leaves background in its bounding box
def test_3d_glcm_matlab_background_analytic(tmp_path):
    mask = _ellipsoid()
    assert _box_cells(mask) > mask.sum(), "the fixture must leave background inside its bounding box"
    _assert_matches_definition(tmp_path, mask)


# negative case: a ROI that fills its bounding box has no background to count
def test_3d_glcm_matlab_box_analytic(tmp_path):
    mask = _box()
    assert _box_cells(mask) == mask.sum(), "the fixture must fill its bounding box"
    _assert_matches_definition(tmp_path, mask)
