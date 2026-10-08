"""Mechanics of the 3D NGTDM family through the Python API. Claims no oracle (SPEC 2)."""
import math

import numpy as np
import pytest

from test_3d_ngtdm_common import NGTDM_3D_FEATURES, featurize_3d_ngtdm

tifffile = pytest.importorskip("tifffile")

# What Nyxus reports for a feature with no value. The Python API has no setting for it, so a feature
# that comes back as exactly this from a ROI with no matrix is told apart from a computed one by the
# positive control below, which gives five values other than it through the same path.
SOFT_NAN = 0.0


def _make_corner_pair(tmp_path, with_neighbour):
    # Two voxels at opposite corners of a 24x60x60 volume, levels 1 and 3. The ROI's bounding box is
    # the whole volume, over 1 MB, so ram_limit=1 forces the oversized path; the corners are 59 apart
    # in Chebyshev distance, so at radius 1 neither has a neighbour and the NGTDM is empty.
    # 'with_neighbour' adds a third voxel at (0,0,1), level 2, beside the first corner: the bounding
    # box and the path stay the same, and two of the three voxels now have a neighbour.
    Z, Y, X = 24, 60, 60
    inten = np.zeros((Z, Y, X), dtype=np.uint16)
    mask = np.zeros((Z, Y, X), dtype=np.uint32)
    inten[0, 0, 0], inten[Z - 1, Y - 1, X - 1] = 1, 3
    mask[0, 0, 0] = mask[Z - 1, Y - 1, X - 1] = 1
    if with_neighbour:
        inten[0, 0, 1], mask[0, 0, 1] = 2, 1
    tag = "neighboured" if with_neighbour else "corners"
    intp = tmp_path / ("%s_int.ome.tif" % tag)
    segp = tmp_path / ("%s_seg.ome.tif" % tag)
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask, metadata={"axes": "ZYX"})
    return str(intp), str(segp)


def test_3d_ngtdm_ooc_empty_matrix_featurize_mechanics(tmp_path):
    """An oversized ROI none of whose voxels has a ROI neighbour has an empty NGTDM, and a
    featurisation reports every feature as the soft-NaN value. With one voxel added beside a corner,
    the same bounding box through the same path gives five values that are not it -- so the first
    result is the empty matrix, not the oversized path returning nothing for a sparse ROI.

    The output stage writes any non-finite value as the soft-NaN value, so this cannot tell
    osized_calculate()'s own empty-matrix return from a NaN; the gtest
    test_3d_ngtdm_ooc_empty_matrix_mechanics calls osized_calculate() directly for that."""
    got, oversized = featurize_3d_ngtdm(*_make_corner_pair(tmp_path, False), 1, 1)
    assert oversized, "the ROI was not processed as oversized"
    assert [got[f] for f in NGTDM_3D_FEATURES] == [SOFT_NAN] * 5, got

    got, oversized = featurize_3d_ngtdm(*_make_corner_pair(tmp_path, True), 1, 1)
    assert oversized, "the ROI was not processed as oversized"
    assert all(math.isfinite(v) and v != SOFT_NAN for v in got.values()), got
