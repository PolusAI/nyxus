#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <tiffio.h>

#include "../src/nyx/grayscale_tiff.h"
#include "../src/nyx/raw_tiff.h"
#include "../src/nyx/helpers/fsystem.h"

//
// SampleFormat (TIFF tag 339) is optional: TIFF 6.0 says an absent tag means 1, unsigned integer,
// so a file that omits it is conformant, and that is what most writers emit for unsigned data --
// `tifffile` declines to write the tag at all in that case, and the checked-in tiled masks
// tests/data/hounsfield/ct_small_mask.tif and mask.tif carry no 339.
//
// All four TIFF loaders read the tag into sampleFormat_, which stays 0 when TIFFGetField finds
// nothing, and all four then map 0 onto 1 before dispatching on it:
//
//   RawTiffStripLoader            raw_tiff.h        prescan side, strip layout
//   RawTiffTileLoader             raw_tiff.h        prescan side, tile layout
//   NyxusGrayscaleTiffStripLoader grayscale_tiff.h  feature side, strip layout
//   NyxusGrayscaleTiffTileLoader  grayscale_tiff.h  feature side, tile layout
//
// Which of the four a run reaches is decided by the file's layout and by whether the request is
// whole-image or segmented, neither of which says anything about the pixel format, so the two tile
// loaders are the ones these tests drive: their fixtures are tiled and tagless, and each must read
// the pixels back unsigned.
//
// The fallback supplies a format only where the file names none. The negatives below hold that
// line: a file that DOES declare a format keeps it (so the mask-side float refusal still fires),
// and a bit depth the loader cannot type is still refused.
//
// Those two negatives are also the only tests that take a tile loader's refusal paths, which is
// where the open TIFF handle and the libtiff tile buffer are released. A gtest cannot see a leak;
// the AddressSanitizer gate runs this file with leak detection on, and that is what checks it.
//

// pixel value at image (col, row); row-dependent, so a read that lands on the wrong row shows up
static inline uint16_t nyxus_ut_tagless_enc (size_t col, size_t row)
{
    return (uint16_t) ((col * 11 + row * 101) % 65536);
}

// Write a tiled, single-channel grayscale TIFF of the given bit depth. `sampleFormat` < 0 leaves
// tag 339 out of the file entirely, which is the case under test; >= 0 writes it.
static inline void nyxus_ut_write_tiled_tiff (
    const std::string& path,
    size_t w, size_t h, size_t tileDim,
    int bitsPerSample,
    int sampleFormat)
{
    TIFF* t = TIFFOpen (path.c_str(), "w");
    if (t == nullptr)
        throw std::runtime_error("could not create TIFF: " + path);

    TIFFSetField (t, TIFFTAG_IMAGEWIDTH, (uint32_t) w);
    TIFFSetField (t, TIFFTAG_IMAGELENGTH, (uint32_t) h);
    TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, (uint16_t) bitsPerSample);
    TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
    if (sampleFormat >= 0)
        TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, (uint16_t) sampleFormat);
    TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField (t, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField (t, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField (t, TIFFTAG_TILEWIDTH, (uint32_t) tileDim);
    TIFFSetField (t, TIFFTAG_TILELENGTH, (uint32_t) tileDim);

    const tmsize_t tszb = TIFFTileSize (t);
    std::vector<uint8_t> tile ((size_t) tszb, 0);

    for (size_t row0 = 0; row0 < h; row0 += tileDim)
        for (size_t col0 = 0; col0 < w; col0 += tileDim)
        {
            std::fill (tile.begin(), tile.end(), (uint8_t) 0);

            // only the typed payloads the loaders can read are filled in; a bit depth outside them
            // is written as zeros, since those fixtures exist to be refused, not to be compared
            if (bitsPerSample == 16)
            {
                uint16_t* p = (uint16_t*) tile.data();
                for (size_t r = 0; r < tileDim; r++)
                    for (size_t c = 0; c < tileDim; c++)
                        p[r * tileDim + c] = (row0 + r < h && col0 + c < w) ?
                            nyxus_ut_tagless_enc (col0 + c, row0 + r) : (uint16_t) 0;
            }
            else if (bitsPerSample == 32 && sampleFormat == SAMPLEFORMAT_IEEEFP)
            {
                float* p = (float*) tile.data();
                for (size_t r = 0; r < tileDim; r++)
                    for (size_t c = 0; c < tileDim; c++)
                        p[r * tileDim + c] = (float) nyxus_ut_tagless_enc (col0 + c, row0 + r);
            }

            if (TIFFWriteTile (t, tile.data(), (uint32_t) col0, (uint32_t) row0, 0, 0) < 0)
            {
                TIFFClose (t);
                throw std::runtime_error("TIFFWriteTile failed at (" + std::to_string(col0) + "," + std::to_string(row0) + ")");
            }
        }

    TIFFClose (t);
}

// RAII for the runtime-written fixture: the tests assert mid-flight, so the file is removed by the
// destructor rather than at the end of the test body
struct NyxusUtTiledTiff
{
    std::string path;

    NyxusUtTiledTiff (const char* stem, size_t w, size_t h, size_t tileDim, int bps, int sf)
    {
        path = (fs::temp_directory_path() / (std::string("nyxus_ut_") + stem + ".tif")).string();
        nyxus_ut_write_tiled_tiff (path, w, h, tileDim, bps, sf);
    }

    ~NyxusUtTiledTiff()
    {
        std::error_code ec;
        fs::remove (path, ec);
    }
};

// The feature-side tile loader, which is what a segmented request reaches for a tiled file: a
// tagless fixture reads back as unsigned, pixel for pixel, over the whole 2x2 tile grid.
void test_2d_tagless_tiled_tiff_grayscale_tile_loader_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagless_tiled_gray", W, H, TD, 16, -1 /*no tag 339*/);

    NyxusGrayscaleTiffTileLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path,
        true /*permit_fp*/, 0.0, 0.0, 0.0);

    ASSERT_EQ (ldr.fullWidth(0), W);
    ASSERT_EQ (ldr.fullHeight(0), H);

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    for (size_t tr = 0; tr * th < H; tr++)
        for (size_t tc = 0; tc * tw < W; tc++)
        {
            ASSERT_NO_THROW (ldr.loadTileFromFile (tile, tr, tc, 0, 0)) << "tile (" << tr << "," << tc << ")";

            for (size_t r = 0; r < th; r++)
                for (size_t c = 0; c < tw; c++)
                    ASSERT_EQ ((*tile)[r * tw + c], (uint32_t) nyxus_ut_tagless_enc (tc * tw + c, tr * th + r))
                        << "tile (" << tr << "," << tc << ") at in-tile (" << c << "," << r << ")";
        }
}

// The prescan-side tile loader, which slideprops.cpp reaches for the same file: same fixture, same
// values. Both sides have to agree, since the prescan decides the slide's intensity range.
void test_2d_tagless_tiled_tiff_raw_tile_loader_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagless_tiled_raw", W, H, TD, 16, -1 /*no tag 339*/);

    RawTiffTileLoader ldr (fixture.path);

    ASSERT_EQ (ldr.fullWidth(0), W);
    ASSERT_EQ (ldr.fullHeight(0), H);

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);

    for (size_t tr = 0; tr * th < H; tr++)
        for (size_t tc = 0; tc * tw < W; tc++)
        {
            ASSERT_NO_THROW (ldr.loadTileFromFile (tr, tc, 0, 0)) << "tile (" << tr << "," << tc << ")";

            for (size_t r = 0; r < th; r++)
                for (size_t c = 0; c < tw; c++)
                    ASSERT_EQ (ldr.get_uint32_pixel (r * tw + c), (uint32_t) nyxus_ut_tagless_enc (tc * tw + c, tr * th + r))
                        << "tile (" << tr << "," << tc << ") at in-tile (" << c << "," << r << ")";

            ldr.free_tile();
        }
}

// Negative: a file that declares a format keeps it. A tiled float fixture is still a float fixture,
// so the mask-side refusal of real-valued pixels fires as before -- the fallback reads no tag as
// unsigned, it does not rewrite a tag the file carries.
void test_2d_tagless_tiled_tiff_declared_float_still_refused_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagged_float_tiled", W, H, TD, 32, SAMPLEFORMAT_IEEEFP);

    NyxusGrayscaleTiffTileLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path,
        false /*permit_fp: a mask may not carry real-valued pixels*/, 0.0, 0.0, 0.0);

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    EXPECT_ANY_THROW (ldr.loadTileFromFile (tile, 0, 0, 0, 0));
}

// Negative: the fallback supplies a sample format, not blanket acceptance. A tagless tiled fixture
// at a bit depth neither loader can type is still refused, on both sides.
void test_2d_tagless_tiled_tiff_unsupported_depth_still_refused_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagless_tiled_4bit", W, H, TD, 4 /*bits per sample*/, -1 /*no tag 339*/);

    // prescan side: the typed getter is resolved in the constructor, so that is where it refuses
    EXPECT_ANY_THROW (RawTiffTileLoader ldr (fixture.path));

    // feature side: the constructor accepts the header and the typed read refuses
    NyxusGrayscaleTiffTileLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path,
        true /*permit_fp*/, 0.0, 0.0, 0.0);
    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    EXPECT_ANY_THROW (ldr.loadTileFromFile (tile, 0, 0, 0, 0));
}
