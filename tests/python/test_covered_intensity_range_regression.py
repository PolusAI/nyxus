"""Regression coverage for COVERED_IMAGE_INTENSITY_RANGE (2D) / 3COVERED_IMAGE_INTENSITY_RANGE (3D)
on a genuinely SEGMENTED image -- a mask strictly smaller than the full image/volume, leaving
off-mask background voxels the feature's "whole slide" baseline must still see.

No prior test exercised this: every existing fixture asserting this feature was either whole-slide
(mask == full image, so there is no off-mask region to get wrong) or happened to produce the
degenerate value 1.0/near-1.0 by coincidence, not by design -- masking the bug this test targets.

Bug (found while investigating a stale vetting-registry note): gatherRoisMetrics_2_slideprops /
gatherRoisMetrics_2_slideprops_3D (slideprops.cpp) computed the "pre-ROI" (whole-slide) min/max
intensity by skipping off-mask voxels, so for any segmented image the "whole slide" baseline
silently collapsed to the ROI's own min/max -- COVERED_IMAGE_INTENSITY_RANGE was always exactly 1
whenever a real mask was in play. Fixed by reading every voxel's intensity for the baseline while
still gating ROI geometry (labels/area/AABB) on the mask.

This test computes the expected ratio independently from the raw arrays (not a canned golden), so
it can't go stale the same way.
"""
import os
import pathlib
import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")


def _ellipse_mask(shape, radii):
    idx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    centers = [(s - 1) / 2.0 for s in shape]
    acc = sum(((idx[d] - centers[d]) / radii[d]) ** 2 for d in range(len(shape)))
    return (acc <= 1.0).astype(np.uint32)


def test_covered_intensity_range_2d_segmented(tmp_path):
    Y, X = 90, 90
    y = np.arange(Y)[:, None]
    x = np.arange(X)[None, :]
    inten = (1 + (x % 256) + (y % 200) * 256).astype(np.uint32)
    mask = _ellipse_mask((Y, X), (Y / 3.0, X / 3.0))

    intdir, segdir = tmp_path / "int", tmp_path / "seg"
    intdir.mkdir(); segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)

    n = nyxus.Nyxus(["*ALL_INTENSITY*"])
    df = n.featurize_directory(str(intdir) + os.sep, str(segdir) + os.sep)

    roi = inten[mask.astype(bool)]
    expected = (float(roi.max()) - float(roi.min())) / (float(inten.max()) - float(inten.min()))

    assert df["COVERED_IMAGE_INTENSITY_RANGE"].iloc[0] == pytest.approx(expected, rel=1e-9)
    # sanity: this fixture's mask leaves real background outside it, so the bug (ratio == 1.0)
    # would be plainly wrong here -- guard against a future fixture change silently losing that.
    assert expected < 0.99


def test_covered_intensity_range_3d_segmented(tmp_path):
    Z, Y, X = 8, 90, 90
    z = np.arange(Z)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]
    inten = (1 + (x % 256) + (y % 200) * 256 + z * 10000).astype(np.uint32)
    inten = np.broadcast_to(inten, (Z, Y, X)).astype(np.uint32)
    mask = _ellipse_mask((Z, Y, X), (Z / 2.0, Y / 3.0, X / 3.0))

    intp = tmp_path / "vol_int.ome.tif"
    segp = tmp_path / "vol_seg.ome.tif"
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask, metadata={"axes": "ZYX"})

    n = nyxus.Nyxus3D(["*3D_ALL_INTENSITY*"], ram_limit=8000)
    df = n.featurize_files([str(intp)], [str(segp)], False)

    roi = inten[mask.astype(bool)]
    expected = (float(roi.max()) - float(roi.min())) / (float(inten.max()) - float(inten.min()))

    assert df["3COVERED_IMAGE_INTENSITY_RANGE"].iloc[0] == pytest.approx(expected, rel=1e-9)
    assert expected < 0.99
