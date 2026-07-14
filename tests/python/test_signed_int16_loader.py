"""Robustness test for signed-integer TIFF loading.

A signed int16 image (e.g. a CT with negative Hounsfield units) whose ROI contains negative values used
to be cast straight to the unsigned pipeline type PixIntens, wrapping e.g. -1024 to ~4.29e9. That absurd
maximum then sized the grey-bin / histogram allocation and segfaulted on macOS (Windows/Linux "survived"
by allocator luck while emitting garbage MIN/MAX/MEAN). The loader now clamps negatives of signed integer
file types to 0 (grayscale_tiff.h tile- and strip-loaders) as a safe fallback for the no-preserve-hu path.

This exercises the featurize_files (TIFF) path that the in-memory numpy path does not cover.
"""
import numpy as np
import pytest
import nyxus

tifffile = pytest.importorskip("tifffile")


def _write_pair(tmp_path, inten, lab):
    ip = tmp_path / "int_ct.tif"
    sp = tmp_path / "seg.tif"
    tifffile.imwrite(str(ip), inten)
    tifffile.imwrite(str(sp), lab)
    return [str(ip)], [str(sp)]


def test_signed_int16_negatives_do_not_wrap(tmp_path):
    H, W = 16, 16
    inten = np.full((H, W), -1024, np.int16)   # negative Hounsfield units across the ROI
    inten[6:10, 6:10] = 40                      # a patch of positive intensity
    lab = np.ones((H, W), np.uint16)            # ROI = whole image (includes the negatives)
    ifiles, sfiles = _write_pair(tmp_path, inten, lab)

    nyx = nyxus.Nyxus(features=["MIN", "MAX", "MEAN", "RANGE"], n_feature_calc_threads=1)
    r = nyx.featurize_files(ifiles, sfiles, False).iloc[0]

    # negatives clamped to 0 -> finite, non-wrapped features (was MAX/MEAN/RANGE ~4.29e9)
    assert r["MAX"] < 1e6, f"MAX wrapped to a huge value: {r['MAX']}"
    assert r["MAX"] == 40.0
    assert r["MIN"] == 0.0
    assert r["RANGE"] == 40.0
    assert r["MEAN"] == pytest.approx((0 * 240 + 40 * 16) / 256)


def test_unsigned_int16_is_unaffected(tmp_path):
    """The clamp must not touch valid unsigned data."""
    H, W = 16, 16
    inten = np.zeros((H, W), np.uint16)
    inten[4:12, 4:12] = 5000
    lab = np.ones((H, W), np.uint16)
    ifiles, sfiles = _write_pair(tmp_path, inten, lab)

    nyx = nyxus.Nyxus(features=["MIN", "MAX"], n_feature_calc_threads=1)
    r = nyx.featurize_files(ifiles, sfiles, False).iloc[0]
    assert r["MAX"] == 5000.0
    assert r["MIN"] == 0.0
