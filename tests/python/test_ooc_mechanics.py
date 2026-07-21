"""Mechanics tests for the out-of-core (oversized-ROI) path.

An ROI whose memory footprint reaches ram_limit is streamed through the disk-backed
OutOfRamPixelCloud instead of an in-memory pixel vector, and its features are computed by
the osized_calculate path. These tests exercise that plumbing (not an external oracle):

- test_ooc_2d_matches_in_ram: the out-of-core path must produce the same intensity features
  as the in-RAM path on the same file pair.
- test_ooc_montage_oversized_fails_loudly: the in-memory montage path has no out-of-core
  support and must raise on an oversized ROI rather than emit a silent all-zero row.

ram_limit is a process-global in Nyxus, so each test sets it explicitly on both sides to stay
order-independent.
"""
import os
import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")


def _make_pair(tmp_path):
    # Deterministic, non-degenerate: per-row offset + per-column gradient. 500x500 gives an
    # in-memory footprint well above 1 MB, so ram_limit=1 forces the out-of-core path.
    Y, X = 500, 500
    xg = (np.arange(X) % 256).astype(np.uint32)
    yg = ((np.arange(Y) % 200) * 256).astype(np.uint32)
    inten = (1 + xg[None, :] + yg[:, None]).astype(np.uint32)
    mask = np.ones((Y, X), np.uint32)  # a single ROI covering the whole image
    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)
    return str(intdir) + os.sep, str(segdir) + os.sep


def _feature_cols(df):
    cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("ROI_label", "t_index", "c_index")
    ]
    return cols, df[cols].to_numpy(dtype=float).ravel()


def test_ooc_2d_matches_in_ram(tmp_path):
    intdir, segdir = _make_pair(tmp_path)
    feats = ["*ALL_INTENSITY*"]

    n_ram = nyxus.Nyxus(feats)
    n_ram.set_params(ram_limit=8000)  # large -> in-RAM (trivial); explicit so test is order-independent
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    n_ooc.set_params(ram_limit=1)  # 1 MB -> forces the oversized / out-of-core path
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

    bad = [
        (c, p, q)
        for c, p, q in zip(cols, a, b)
        if abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9
    ]
    assert not bad, "out-of-core intensity features diverge from in-RAM: %r" % (bad[:8],)


def _make_volume_pair(tmp_path):
    # A 3D OME-TIFF volume (one IFD per Z) + a whole-volume mask. Z*Y*X = 8*90*90 = 64800
    # voxels; the in-memory 3D footprint is well over 1 MB, so ram_limit=1 forces the oversized
    # (out-of-core) volumetric path. Intensity is a deterministic, non-degenerate function of
    # (x,y,z) so every 3D intensity feature is meaningful.
    Z, Y, X = 8, 90, 90
    z = np.arange(Z)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]
    inten = (1 + (x % 256) + (y % 200) * 256 + z * 10000).astype(np.uint32)
    inten = np.broadcast_to(inten, (Z, Y, X)).astype(np.uint32)
    mask = np.ones((Z, Y, X), np.uint32)  # a single ROI covering the whole volume
    intp = tmp_path / "vol_int.ome.tif"
    segp = tmp_path / "vol_seg.ome.tif"
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask, metadata={"axes": "ZYX"})
    return str(intp), str(segp)


def test_ooc_3d_matches_in_ram(tmp_path):
    """The 3D out-of-core path (voxel cloud streamed to disk, keeping z) must produce the same
    3D intensity AND surface/morphology features as the in-RAM path on the same volume pair."""
    intp, segp = _make_volume_pair(tmp_path)
    feats = ["*3D_ALL_INTENSITY*", "*3D_ALL_MORPHOLOGY*"]

    # Nyxus3D takes ram_limit in the constructor (its set_params does not expose it)
    n_ram = nyxus.Nyxus3D(feats, ram_limit=8000)  # large -> in-RAM (trivial)
    df_ram = n_ram.featurize_files([intp], [segp], False)

    n_ooc = nyxus.Nyxus3D(feats, ram_limit=1)  # 1 MB -> forces the oversized / out-of-core volumetric path
    df_ooc = n_ooc.featurize_files([intp], [segp], False)

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

    bad = [
        (c, p, q)
        for c, p, q in zip(cols, a, b)
        if abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9
    ]
    assert not bad, "3D out-of-core intensity features diverge from in-RAM: %r" % (bad[:8],)


def test_ooc_montage_oversized_fails_loudly():
    """The in-memory (montage) path has no out-of-core support, so an ROI whose footprint
    reaches ram_limit must fail loudly rather than emit a silent all-zero feature row."""
    Y, X = 300, 300
    xg = (np.arange(X) % 256).astype(np.uint32)
    yg = ((np.arange(Y) % 200) * 256).astype(np.uint32)
    inten = (1 + xg[None, :] + yg[:, None]).astype(np.uint32)
    mask = np.ones((Y, X), np.uint32)

    # sanity: with a large ram_limit the montage path succeeds and is non-zero. Set it
    # explicitly (nyxus ram_limit is process-global) so this does not depend on test order.
    ok = nyxus.Nyxus(["*ALL_INTENSITY*"])
    ok.set_params(ram_limit=8000)
    df_ok = ok.featurize(inten, mask, intensity_names=["I"], label_names=["M"])
    assert df_ok["MEAN"].iloc[0] > 0

    # ram_limit=1 makes the single ROI oversized -> must raise, not return zeros
    n = nyxus.Nyxus(["*ALL_INTENSITY*"])
    n.set_params(ram_limit=1)
    with pytest.raises(Exception, match="oversized"):
        n.featurize(inten, mask, intensity_names=["I"], label_names=["M"])
