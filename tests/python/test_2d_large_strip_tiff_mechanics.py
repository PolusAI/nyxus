"""Strip-based TIFFs whose width or height exceeds the loader's 1024-pixel tile.

The prescan (slideprops.cpp, on every entry point) reads a non-tiled TIFF through
RawImageLoader -> RawTiffStripLoader, whose tile grid is 1024 x 1024. An image within 1024
in both axes is a single tile at (0,0); past 1024 in either axis the grid has more than one
cell, and the tile's origin and buffer pitch start to matter. Tiled TIFFs take a different
loader and are unaffected either way, so the geometries here are the ones that exercise the
strip loader's tile grid at all.

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


def _featurize_both(tmp_path, w, h):
    """Featurize one image twice: as a strip TIFF on disk, and as the numpy array it came from."""
    int_dir = tmp_path / f"int{w}x{h}"
    seg_dir = tmp_path / f"seg{w}x{h}"
    int_dir.mkdir()
    seg_dir.mkdir()

    inten = _intensity(w, h)
    mask = np.ones_like(inten, dtype=np.uint32)   # one ROI over the whole image

    # no tile= -> strip layout, which is what selects the strip loader
    tifffile.imwrite(str(int_dir / "img.tif"), inten)
    tifffile.imwrite(str(seg_dir / "img.tif"), mask)

    # ram_limit is a process-global that an earlier out-of-core test can leave at 0; these
    # fixtures are a few megabytes, so set it high enough that both paths run in-RAM and the
    # comparison is order-independent
    n_disk = nyxus.Nyxus(features=_FEATURES, n_feature_calc_threads=1)
    n_disk.set_params(ram_limit=512)
    from_disk = n_disk.featurize_files(
        intensity_files=[str(int_dir / "img.tif")],
        mask_files=[str(seg_dir / "img.tif")],
        single_roi=False,
    )

    n_mem = nyxus.Nyxus(features=_FEATURES, n_feature_calc_threads=1)
    n_mem.set_params(ram_limit=512)
    in_memory = n_mem.featurize(inten, mask)

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
