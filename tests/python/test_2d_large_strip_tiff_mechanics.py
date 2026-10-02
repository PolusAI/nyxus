"""Strip-based TIFFs whose width or height exceeds the loader's 1024-pixel tile.

A non-tiled TIFF is read by two strip loaders: the prescan (slideprops.cpp, on every entry
point) goes through RawImageLoader -> RawTiffStripLoader, and the feature passes through
ImageLoader -> NyxusGrayscaleTiffStripLoader, for the intensity and the mask image alike. Both
use a 1024 x 1024 tile grid. An image within 1024 in both axes is a single tile at (0,0); past
1024 in either axis the grid has more than one cell, and the tile's origin, its buffer pitch
and -- in a compressed file -- the strip a tile starts inside all start to matter. Tiled TIFFs
take different loaders and are unaffected either way, so the geometries here are the ones that
exercise the strip loaders' tile grid at all.

Nyxus.featurize(numpy, ...) never opens a file, so it is the oracle: it must report the same
values as featurize_files() on the very same pixels written to disk as a strip TIFF.

Every fixture is written at runtime -- a checked-in TIFF above 1024 pixels would be megabytes,
and the repository's largest fixture is 512 x 512.
"""

import numpy as np
import pytest

import nyxus

# tifffile only synthesizes the on-disk fixture; skip cleanly where it is absent
tifffile = pytest.importorskip("tifffile")

# The loader's tile grid is STRIP_TILE_WIDTH x STRIP_TILE_HEIGHT (src/nyx/raw_tiff.h).
_TILE = 1024

_FEATURES = [
    "MAX",
    "MIN",
    "MEAN",
    "MEDIAN",
    "RANGE",
    "INTEGRATED_INTENSITY",
    "STANDARD_DEVIATION",
    "AREA_PIXELS_COUNT",
]


def _intensity(w, h):
    """Row- and column-dependent pattern, with the extrema past the first tile.

    The base pattern varies along both axes, so a tile that reads the wrong rows or addresses
    the buffer at the wrong pitch changes the sum. The MIN and MAX live in the bottom-right
    corner, which only a correctly addressed last tile reaches, so they pin the tile origin
    on top of the aggregate.
    """
    y, x = np.mgrid[0:h, 0:w]
    a = ((x * 7 + y * 3) % 60000 + 100).astype(np.uint16)
    a[h - 10, w - 10] = 65535   # MAX
    a[h - 5, w - 50] = 1        # MIN
    return a


def _set_ram_limit_mb(nyx, mb):
    """Set the process-global ram_limit and verify it was accepted.

    Nyxus rejects a limit above the RAM currently available and keeps the instance's previous
    value, reporting the refusal without raising. Read the value back so that a refusal fails here,
    naming the cause, and the claim that both sides of the comparison run in RAM is checked rather
    than assumed."""
    nyx.set_params(ram_limit=mb)
    got = nyx.get_params("ram_limit")["ram_limit"]
    assert got == mb, (
        "ram_limit=%d MB was not accepted (still %d MB) -- Nyxus refuses a limit above available "
        "RAM." % (mb, got)
    )


def _write_pair(tmp_path, w, h, compression=None, rowsperstrip=None):
    """Write the intensity image and its whole-image mask as strip TIFFs; return the paths."""
    tag = f"{w}x{h}_{compression}_{rowsperstrip}"
    int_dir = tmp_path / f"int{tag}"
    seg_dir = tmp_path / f"seg{tag}"
    int_dir.mkdir()
    seg_dir.mkdir()

    inten = _intensity(w, h)
    mask = np.ones_like(inten, dtype=np.uint32)   # one ROI over the whole image

    # no tile= -> strip layout, which is what selects the strip loaders
    int_path, seg_path = str(int_dir / "img.tif"), str(seg_dir / "img.tif")
    tifffile.imwrite(int_path, inten, compression=compression, rowsperstrip=rowsperstrip)
    tifffile.imwrite(seg_path, mask, compression=compression, rowsperstrip=rowsperstrip)
    return inten, int_path, seg_path


def _featurize_files(int_path, seg_path):
    # these fixtures are a few megabytes, so a 512 MB ram_limit keeps both paths in RAM
    n_disk = nyxus.Nyxus(features=_FEATURES, n_feature_calc_threads=1)
    _set_ram_limit_mb(n_disk, 512)
    return n_disk.featurize_files(
        intensity_files=[int_path],
        mask_files=[seg_path],
        single_roi=False,
    )


def _featurize_both(tmp_path, w, h, compression=None, rowsperstrip=None):
    """Featurize one image twice: as a strip TIFF on disk, and as the numpy array it came from."""
    inten, int_path, seg_path = _write_pair(tmp_path, w, h, compression, rowsperstrip)

    from_disk = _featurize_files(int_path, seg_path)

    n_mem = nyxus.Nyxus(features=_FEATURES, n_feature_calc_threads=1)
    _set_ram_limit_mb(n_mem, 512)
    in_memory = n_mem.featurize(inten, np.ones_like(inten, dtype=np.uint32))

    return inten, from_disk, in_memory


def _assert_agrees(inten, from_disk, in_memory, what):
    assert from_disk.shape[0] == 1, f"{what}: expected exactly one ROI"

    for col in _FEATURES:
        assert np.isclose(
            from_disk.at[0, col], in_memory.at[0, col], rtol=1e-9, atol=1e-6
        ), f"{what}: disk vs in-memory mismatch for {col}"

    # absolute anchors, so the test still says something if both paths were to move together
    assert from_disk.at[0, "MAX"] == float(inten.max()) == 65535.0, what
    assert from_disk.at[0, "MIN"] == float(inten.min()) == 1.0, what
    assert from_disk.at[0, "AREA_PIXELS_COUNT"] == float(inten.size), what
    assert np.isclose(
        from_disk.at[0, "INTEGRATED_INTENSITY"], float(inten.astype(np.float64).sum()),
        rtol=1e-12, atol=0.5,
    ), what


@pytest.mark.parametrize(
    "w,h",
    [
        (_TILE + 76, 300),          # 2 x 1 tile grid: wide only
        (300, _TILE + 76),          # 1 x 2 tile grid: tall only
        (_TILE + 76, _TILE + 76),   # 2 x 2 tile grid, with partial edge tiles
    ],
    ids=["wide", "tall", "both"],
)
def test_2d_large_strip_tiff_matches_memory_mechanics(tmp_path, w, h):
    """A strip TIFF past 1024 pixels reads back exactly what the in-memory path measures."""
    inten, from_disk, in_memory = _featurize_both(tmp_path, w, h)
    _assert_agrees(inten, from_disk, in_memory, f"{w}x{h}")


def test_2d_strip_tiff_at_tile_boundary_matches_memory_mechanics(tmp_path):
    """Exactly 1024 x 1024 is still a single tile, and must agree just the same.

    This is the control for the sizes above: it is the largest geometry the loader covers with
    one tile, so it separates "past the tile boundary" from "large".
    """
    inten, from_disk, in_memory = _featurize_both(tmp_path, _TILE, _TILE)
    _assert_agrees(inten, from_disk, in_memory, f"{_TILE}x{_TILE}")


# A codec decodes a strip from its first row forward, and the second tile row starts at image row
# 1024. 100 rows per strip puts row 1024 inside a strip, one strip spanning the image puts every
# tile inside the same strip, and 600 x 2000 is past 1024 in height only, so its tiles are read
# top to bottom. 256 rows per strip divides 1024, so its second tile row starts on a strip
# boundary: it is the control that separates "compressed" from "starts inside a strip".
@pytest.mark.parametrize(
    "w,h,compression,rowsperstrip",
    [
        (_TILE + 76, _TILE + 76, "lzw", 100),
        (_TILE + 76, _TILE + 76, "lzw", _TILE + 76),
        (_TILE + 76, _TILE + 76, "packbits", 100),
        (600, 2000, "lzw", 100),
        (_TILE + 76, _TILE + 76, "lzw", 256),
    ],
    ids=["lzw100", "lzw_one_strip", "packbits100", "tall_lzw100", "lzw256_strip_aligned"],
)
def test_2d_large_compressed_strip_tiff_matches_memory_mechanics(tmp_path, w, h, compression, rowsperstrip):
    """A compressed strip TIFF past 1024 pixels reads back exactly what the in-memory path measures."""
    # tifffile writes LZW and PackBits through imagecodecs; only these cases need it
    pytest.importorskip("imagecodecs")
    inten, from_disk, in_memory = _featurize_both(tmp_path, w, h, compression, rowsperstrip)
    _assert_agrees(inten, from_disk, in_memory, f"{w}x{h} {compression} {rowsperstrip} rows/strip")


def test_2d_large_compressed_strip_tiff_corrupt_strip_raises_mechanics(tmp_path):
    """An LZW strip that does not decode makes featurize_files raise, not return values.

    Strip 10 holds image rows 1000..1099, which the second tile row starts inside. Its bytes are
    overwritten with 0xFF, a run of codes past the end of the LZW table.
    """
    pytest.importorskip("imagecodecs")
    w = h = _TILE + 76
    _, int_path, seg_path = _write_pair(tmp_path, w, h, "lzw", 100)

    with tifffile.TiffFile(int_path) as tf:
        page = tf.pages[0]
        offset, nbytes = page.dataoffsets[10], page.databytecounts[10]
    with open(int_path, "r+b") as f:
        f.seek(offset + (3 * nbytes) // 4)
        f.write(b"\xff" * 64)

    # precondition: the damaged strip really does not decode
    with pytest.raises(Exception):
        tifffile.imread(int_path)

    with pytest.raises(RuntimeError, match="TIFFReadScanline"):
        _featurize_files(int_path, seg_path)
