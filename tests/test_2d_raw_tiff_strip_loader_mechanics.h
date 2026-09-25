#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <tiffio.h>

#include "../src/nyx/raw_tiff.h"
#include "../src/nyx/helpers/fsystem.h"

//
// RawTiffStripLoader is the prescan-side reader for strip-based (non-tiled) TIFFs: RawImageLoader
// instantiates it, slideprops.cpp prescans through it, and both the CLI and the Python API prescan.
// Its tile grid is STRIP_TILE_WIDTH x STRIP_TILE_HEIGHT = 1024 x 1024, so an image within 1024 in
// both axes is a single tile at (0,0) and every tile index below is the one that matters: these
// fixtures are sized past 1024 in one axis and in both, which is the only way to reach a tile grid
// with more than one cell.
//
// What the tests below pin down:
//   - tile (i,j) carries image rows [i*1024, ...) x columns [j*1024, ...), not rows 0.. of every tile;
//   - the buffer's pitch is the tile width, which is what get_uint32_pixel(r*tw + c) addresses;
//   - a partial edge tile zero-fills the part of the buffer the image does not reach;
//   - free_tile() between tiles is what RawImageLoader does per tile, and the next read re-allocates;
//   - a tile index past the grid is refused rather than read out of bounds.
//

// pixel value at image (col, row); row-dependent so a tile reading the wrong rows is detectable
static inline uint32_t nyxus_ut_strip_enc (size_t col, size_t row)
{
    return (uint32_t) ((col * 7 + row * 3) % 65536);
}

// Write a strip-based (non-tiled), single-channel uint16 grayscale TIFF carrying nyxus_ut_strip_enc.
// SAMPLEFORMAT is set explicitly: the TIFF 6.0 default is SAMPLEFORMAT_UINT, but a writer that omits
// the tag leaves the loader to infer it, and the fixture is about tile addressing, not inference.
static inline void nyxus_ut_write_uint16_strip_tiff (const std::string& path, size_t w, size_t h)
{
    TIFF* t = TIFFOpen (path.c_str(), "w");
    if (t == nullptr)
        throw std::runtime_error("could not create TIFF: " + path);

    TIFFSetField (t, TIFFTAG_IMAGEWIDTH, (uint32_t) w);
    TIFFSetField (t, TIFFTAG_IMAGELENGTH, (uint32_t) h);
    TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, 16);
    TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField (t, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField (t, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField (t, TIFFTAG_ROWSPERSTRIP, 16);   // many strips, so the file is genuinely strip-based

    std::vector<uint16_t> row (w);
    for (size_t y = 0; y < h; y++)
    {
        for (size_t x = 0; x < w; x++)
            row[x] = (uint16_t) nyxus_ut_strip_enc (x, y);
        if (TIFFWriteScanline (t, row.data(), (uint32_t) y, 0) < 0)
        {
            TIFFClose (t);
            throw std::runtime_error("TIFFWriteScanline failed at row " + std::to_string(y));
        }
    }
    TIFFClose (t);
}

// RAII for the runtime-written fixture: the tests assert mid-flight, so the file is removed by the
// destructor rather than at the end of the test body
struct NyxusUtStripTiff
{
    std::string path;

    NyxusUtStripTiff (const char* stem, size_t w, size_t h)
    {
        path = (fs::temp_directory_path() / (std::string("nyxus_ut_") + stem + ".tif")).string();
        nyxus_ut_write_uint16_strip_tiff (path, w, h);
    }

    ~NyxusUtStripTiff()
    {
        std::error_code ec;
        fs::remove (path, ec);
    }
};

// Walk the whole tile grid of a strip TIFF of the given geometry and check every pixel against
// nyxus_ut_strip_enc, freeing the tile between reads exactly as RawImageLoader does.
static inline void nyxus_ut_assert_strip_grid (const char* stem, size_t W, size_t H)
{
    NyxusUtStripTiff fixture (stem, W, H);

    auto ldr = RawTiffStripLoader (1 /*n_threads*/, fixture.path);

    ASSERT_EQ (ldr.fullWidth(0), W) << stem;
    ASSERT_EQ (ldr.fullHeight(0), H) << stem;

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);

    // the grid has more than one cell only past 1024, which is the point of these geometries
    const size_t nCols = (W + tw - 1) / tw,
        nRows = (H + th - 1) / th;
    ASSERT_GT (nCols * nRows, (size_t)1) << stem << ": geometry does not produce a multi-tile grid";

    for (size_t tr = 0; tr < nRows; tr++)
        for (size_t tc = 0; tc < nCols; tc++)
        {
            ASSERT_NO_THROW (ldr.loadTileFromFile (tr, tc, 0, 0)) << stem << " tile (" << tr << "," << tc << ")";

            const size_t row0 = tr * th,
                col0 = tc * tw,
                validH = (std::min) (th, H - row0),
                validW = (std::min) (tw, W - col0);

            for (size_t r = 0; r < th; r++)
                for (size_t c = 0; c < tw; c++)
                {
                    // inside the image: the pixel of global (col0+c, row0+r); outside: zero fill
                    const uint32_t expected = (r < validH && c < validW) ?
                        nyxus_ut_strip_enc (col0 + c, row0 + r) : 0u;
                    ASSERT_EQ (ldr.get_uint32_pixel (r * tw + c), expected)
                        << stem << " tile (" << tr << "," << tc << ") at in-tile (" << c << "," << r << ")";
                }

            // RawImageLoader::free_tile_buffers() runs after every tile, so the next read must work
            ldr.free_tile();
        }
}

// A striped TIFF past 1024 in both axes: a 2x2 grid whose right column and bottom row are partial.
void test_2d_raw_tiff_strip_loader_tile_grid_mechanics()
{
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x1100", 1100, 1100));
}

// Past 1024 in one axis only: either dimension alone splits the grid, and the short axis stays a
// single tile whose column (or row) range is the whole image.
void test_2d_raw_tiff_strip_loader_one_axis_grid_mechanics()
{
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x300", 1100, 300));
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip300x1100", 300, 1100));
}

// Negative: a tile index past the grid names no image rows or columns and is refused.
void test_2d_raw_tiff_strip_loader_tile_out_of_grid_refused_mechanics()
{
    NyxusUtStripTiff fixture ("strip1100x1100_oob", 1100, 1100);

    auto ldr = RawTiffStripLoader (1 /*n_threads*/, fixture.path);

    EXPECT_ANY_THROW (ldr.loadTileFromFile (2, 0, 0, 0));   // tile row past the last row of the grid
    EXPECT_ANY_THROW (ldr.loadTileFromFile (0, 2, 0, 0));   // tile column past the last column
    EXPECT_ANY_THROW (ldr.loadTileFromFile (9, 9, 0, 0));   // both

    ldr.free_tile();
}
