"""Regression tests for the TIFF tile/strip loader.

Guards the fix in src/nyx/grayscale_tiff.h where the 32-bit *unsigned* sample
case of NyxusGrayscaleTiffStripLoader::loadTileFromFile used copyRow<size_t>
instead of copyRow<uint32_t>. size_t is 8 bytes on LP64/Win64 while a 32-bit
sample is 4 bytes, so copyRow read ((size_t*)buf)[col] -- 2x past the libtiff
scanline buffer. That corrupted pixel values (columns shifted / garbage) and
could crash (heap over-read), for ANY uint32 TIFF read from disk.

This must go through a *file-based* API (featurize_files / featurize_directory)
because the in-memory Nyxus.featurize(numpy, ...) path never touches the tile
loader and so cannot exercise the bug. tifffile writes strip-based TIFFs by
default (no tile=), which selects the strip loader that carried the bug.
"""

import numpy as np
import pytest

import nyxus

# tifffile is only needed to synthesize the on-disk uint32 TIFF; skip cleanly if
# it is not available in the test environment.
tifffile = pytest.importorskip("tifffile")


# Intensities deliberately span past the 16-bit range, including a value above
# 2**31, so any wrong element size / truncation / column shift in the loader
# produces a detectably wrong MAX and INTEGRATED_INTENSITY (or a crash).
_INTENSITY = np.array(
    [
        [100,        70_000,      300_000,       16_777_216],
        [4_000_000,  5,           123_456_789,   3_000_000_000],
        [65_535,     65_536,      2_147_483_648, 1],
    ],
    dtype=np.uint32,
)


def _write_strip_tiff(path, arr):
    # No tile= -> strip-based TIFF -> NyxusGrayscaleTiffStripLoader (the fixed path).
    tifffile.imwrite(str(path), arr)


def test_uint32_strip_tiff_pixels_read_correctly(tmp_path):
    """A uint32 strip TIFF loaded from disk yields the exact input statistics."""
    int_dir = tmp_path / "int"
    seg_dir = tmp_path / "seg"
    int_dir.mkdir()
    seg_dir.mkdir()

    inten = _INTENSITY
    mask = np.ones_like(inten, dtype=np.uint32)  # single ROI covering the whole image
    _write_strip_tiff(int_dir / "img.tif", inten)
    _write_strip_tiff(seg_dir / "img.tif", mask)

    nyx = nyxus.Nyxus(
        features=["MAX", "MIN", "INTEGRATED_INTENSITY", "MEAN"],
        n_feature_calc_threads=1,
    )
    f = nyx.featurize_files(
        intensity_files=[str(int_dir / "img.tif")],
        mask_files=[str(seg_dir / "img.tif")],
        single_roi=False,
    )

    assert f.shape[0] == 1, "expected exactly one ROI"
    # With the old copyRow<size_t> these fail (shuffled/garbage pixels) or crash.
    assert f.at[0, "MAX"] == float(inten.max())          # 3_000_000_000
    assert f.at[0, "MIN"] == float(inten.min())          # 1
    assert np.isclose(f.at[0, "INTEGRATED_INTENSITY"], float(inten.sum()), rtol=0, atol=0.5)
    assert np.isclose(f.at[0, "MEAN"], float(inten.mean()), rtol=1e-9)


def test_uint32_tiff_from_disk_matches_in_memory(tmp_path):
    """The file loader must read the same values as the in-memory numpy path.

    Nyxus.featurize(numpy) never uses the tile loader, so it is a correct oracle:
    if the on-disk read matches it for a uint32 image, the loader typing is right.
    """
    int_dir = tmp_path / "int"
    seg_dir = tmp_path / "seg"
    int_dir.mkdir()
    seg_dir.mkdir()

    inten = _INTENSITY
    mask = np.ones_like(inten, dtype=np.uint32)
    _write_strip_tiff(int_dir / "img.tif", inten)
    _write_strip_tiff(seg_dir / "img.tif", mask)

    feats = ["MAX", "MIN", "INTEGRATED_INTENSITY", "MEAN", "MEDIAN", "RANGE"]

    from_disk = nyxus.Nyxus(features=feats, n_feature_calc_threads=1).featurize_files(
        intensity_files=[str(int_dir / "img.tif")],
        mask_files=[str(seg_dir / "img.tif")],
        single_roi=False,
    )
    in_memory = nyxus.Nyxus(features=feats, n_feature_calc_threads=1).featurize(
        inten, mask
    )

    for col in feats:
        assert np.isclose(
            from_disk.at[0, col], in_memory.at[0, col], rtol=1e-9, atol=1e-6
        ), f"disk vs in-memory mismatch for {col}"
