#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>

#include "../src/nyx/grayscale_tiff.h"     // NyxusGrayscaleTiff{Tile,Strip}Loader
#include "../src/nyx/raw_tiff.h"           // RawTiff{Tile,Strip}Loader; also <tiffio.h>, <cstdint>, <memory>, <string>, <vector>
#include "../src/nyx/helpers/fsystem.h"

//
// SampleFormat (TIFF tag 339) is optional: TIFF 6.0 says an absent tag means 1, unsigned integer,
// so a file that omits it is conformant, and that is what most writers emit for unsigned data --
// `tifffile` declines to write the tag at all in that case, and the checked-in tiled masks
// tests/data/hounsfield/ct_small_mask.tif and mask.tif carry no 339.
//
// All four TIFF loaders read the tag into sampleFormat_, which stays 0 when TIFFGetField finds
// nothing, and all four read that 0 as 1:
//
//   RawTiffStripLoader            raw_tiff.h        prescan side, strip layout
//   RawTiffTileLoader             raw_tiff.h        prescan side, tile layout
//   NyxusGrayscaleTiffStripLoader grayscale_tiff.h  feature side, strip layout
//   NyxusGrayscaleTiffTileLoader  grayscale_tiff.h  feature side, tile layout
//
// Which of the four a run reaches is decided by the file's layout and by which side of the run is
// asking -- the prescan reads through RawImageLoader, feature extraction through ImageLoader -- and
// the pixel format is neither. A tiled file lands on a tile loader on both sides, so the two tile
// loaders are what this file drives: tiled, tagless fixtures that must read back unsigned, pixel for
// pixel, in the intensity configuration and in the mask one (real-valued pixels prohibited), and the
// refusals around that default. The tile loaders read SAMPLEFORMAT_VOID (4), untyped
// data, as unsigned too; every other declared format reaches their dispatch as written, so a float
// mask and the complex formats are still refused, as are a bit depth with no typed reader, a
// three-sample file and a file that is not tiled.
//
// Those refusals are thrown from the tile constructors, where the open TIFF handle has to be closed,
// and from the feature-side tile read, where libtiff's tile buffer has to be released. A gtest cannot
// see a leak; running this file under AddressSanitizer with leak detection on
// (`ASAN_OPTIONS=detect_leaks=1`) is what checks those releases.
//
// Strip-layout tests close the file. They pin the agreement the tile loaders are held to: an absent
// tag and a declared VOID read as unsigned on the strip loaders as well, and a declared complex
// format is refused there too.
//

// pixel value at image (col, row); row-dependent, so a read that lands on the wrong row shows up
static inline uint16_t nyxus_ut_tagless_enc (size_t col, size_t row)
{
    return (uint16_t) ((col * 11 + row * 101) % 65536);
}

// Write a tiled grayscale TIFF of the given bit depth and sample count. `sampleFormat` < 0 leaves
// tag 339 out of the file entirely, which is the case under test; >= 0 writes the value given.
static inline void nyxus_ut_write_tiled_tiff (
    const std::string& path,
    size_t w, size_t h, size_t tileDim,
    int bitsPerSample,
    int sampleFormat,
    int samplesPerPixel = 1)
{
    TIFF* t = TIFFOpen (path.c_str(), "w");
    if (t == nullptr)
        throw std::runtime_error("could not create TIFF: " + path);

    TIFFSetField (t, TIFFTAG_IMAGEWIDTH, (uint32_t) w);
    TIFFSetField (t, TIFFTAG_IMAGELENGTH, (uint32_t) h);
    TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, (uint16_t) bitsPerSample);
    TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, (uint16_t) samplesPerPixel);
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

            // only the typed payloads the loaders can read are filled in; a bit depth or a sample
            // count outside them is written as zeros, since those fixtures exist to be refused,
            // not to be compared
            if (bitsPerSample == 16 && samplesPerPixel == 1)
            {
                uint16_t* p = (uint16_t*) tile.data();
                for (size_t r = 0; r < tileDim; r++)
                    for (size_t c = 0; c < tileDim; c++)
                        p[r * tileDim + c] = (row0 + r < h && col0 + c < w) ?
                            nyxus_ut_tagless_enc (col0 + c, row0 + r) : (uint16_t) 0;
            }
            else if (bitsPerSample == 32 && samplesPerPixel == 1 && sampleFormat == SAMPLEFORMAT_IEEEFP)
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

// Write a one-sample strip (scanline) grayscale TIFF. `sampleFormat` follows the tiled writer's
// convention. Only 16 bits per sample carries the encoded payload; any other depth is written as
// zeros, for the fixtures that exist to be refused.
static inline void nyxus_ut_write_sample_format_strip_tiff (
    const std::string& path,
    size_t w, size_t h,
    int sampleFormat,
    int bitsPerSample = 16)
{
    TIFF* t = TIFFOpen (path.c_str(), "w");
    if (t == nullptr)
        throw std::runtime_error("could not create TIFF: " + path);

    TIFFSetField (t, TIFFTAG_IMAGEWIDTH, (uint32_t) w);
    TIFFSetField (t, TIFFTAG_IMAGELENGTH, (uint32_t) h);
    TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, (uint16_t) bitsPerSample);
    TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, (uint16_t) 1);
    if (sampleFormat >= 0)
        TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, (uint16_t) sampleFormat);
    TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField (t, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField (t, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField (t, TIFFTAG_ROWSPERSTRIP, 1);      // scanline layout, so TIFFIsTiled() is false

    std::vector<uint8_t> row ((size_t) TIFFScanlineSize (t), 0);

    for (size_t y = 0; y < h; y++)
    {
        if (bitsPerSample == 16)
        {
            uint16_t* p = (uint16_t*) row.data();
            for (size_t x = 0; x < w; x++)
                p[x] = nyxus_ut_tagless_enc (x, y);
        }

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
struct NyxusUtTiledTiff
{
    std::string path;

    NyxusUtTiledTiff (const char* stem, size_t w, size_t h, size_t tileDim, int bps, int sf, int spp = 1)
    {
        path = (fs::temp_directory_path() / (std::string("nyxus_ut_") + stem + ".tif")).string();
        nyxus_ut_write_tiled_tiff (path, w, h, tileDim, bps, sf, spp);
    }

    ~NyxusUtTiledTiff()
    {
        std::error_code ec;
        fs::remove (path, ec);
    }
};

// the same for a strip fixture
struct NyxusUtSampleFormatStripTiff
{
    std::string path;

    NyxusUtSampleFormatStripTiff (const char* stem, size_t w, size_t h, int sf, int bps = 16)
    {
        path = (fs::temp_directory_path() / (std::string("nyxus_ut_") + stem + ".tif")).string();
        nyxus_ut_write_sample_format_strip_tiff (path, w, h, sf, bps);
    }

    ~NyxusUtSampleFormatStripTiff()
    {
        std::error_code ec;
        fs::remove (path, ec);
    }
};

// Expect `attempt` to throw a std::runtime_error whose message contains `reason`. Every loader
// constructor check throws that type, so the message is what says which check refused the file.
template <typename F>
static inline void nyxus_ut_expect_refusal (F&& attempt, const char* who, const char* reason)
{
    try
    {
        attempt();
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_NE (std::string (e.what()).find (reason), std::string::npos)
            << who << " refused the file for another reason than '" << reason << "': " << e.what();
        return;
    }
    catch (...)
    {
        ADD_FAILURE() << who << " refused the file with something other than std::runtime_error";
        return;
    }
    ADD_FAILURE() << who << " accepted the file";
}

// Read every tile of a tagless-encoded fixture through the feature-side tile loader and compare it,
// pixel for pixel, against nyxus_ut_tagless_enc. `permit_fp` is false in the mask configuration.
static inline void nyxus_ut_expect_grid_unsigned (const std::string& path, bool permit_fp, size_t W, size_t H)
{
    NyxusGrayscaleTiffTileLoader<uint32_t> ldr (1 /*n_threads*/, path, permit_fp, 0.0, 0.0, 0.0);

    ASSERT_EQ (ldr.fullWidth(0), W);
    ASSERT_EQ (ldr.fullHeight(0), H);

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    for (size_t tr = 0; tr * th < H; tr++)
        for (size_t tc = 0; tc * tw < W; tc++)
        {
            ASSERT_NO_THROW (ldr.loadTileFromFile (tile, tr, tc, 0, 0/*channel*/, 0/*timeframe*/, 0)) << "tile (" << tr << "," << tc << ")";

            for (size_t r = 0; r < th; r++)
                for (size_t c = 0; c < tw; c++)
                    ASSERT_EQ ((*tile)[r * tw + c], (uint32_t) nyxus_ut_tagless_enc (tc * tw + c, tr * th + r))
                        << "tile (" << tr << "," << tc << ") at in-tile (" << c << "," << r << ")";
        }
}

// The feature-side tile loader, which is what a segmented request reaches for a tiled file: a
// tagless fixture reads back as unsigned, pixel for pixel, over the whole 2x2 tile grid.
void test_2d_tagless_tiled_tiff_grayscale_tile_loader_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagless_tiled_gray", W, H, TD, 16, -1 /*no tag 339*/);

    nyxus_ut_expect_grid_unsigned (fixture.path, true /*permit_fp*/, W, H);
}

// The same fixture opened the way a segmented request opens its mask, with real-valued pixels
// prohibited. The absent tag is read as unsigned in the constructor, so the loader's mask gate on
// sample formats >= 3 lets it through and the whole grid reads back pixel for pixel.
void test_2d_tagless_tiled_tiff_mask_tile_loader_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagless_tiled_mask", W, H, TD, 16, -1 /*no tag 339*/);

    nyxus_ut_expect_grid_unsigned (fixture.path, false /*permit_fp: the mask configuration*/, W, H);
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
            ASSERT_NO_THROW (ldr.loadTileFromFile (tr, tc, 0, 0/*channel*/, 0/*timeframe*/, 0)) << "tile (" << tr << "," << tc << ")";

            for (size_t r = 0; r < th; r++)
                for (size_t c = 0; c < tw; c++)
                    ASSERT_EQ (ldr.get_uint32_pixel (r * tw + c), (uint32_t) nyxus_ut_tagless_enc (tc * tw + c, tr * th + r))
                        << "tile (" << tr << "," << tc << ") at in-tile (" << c << "," << r << ")";

            ldr.free_tile();
        }
}

// Negative: a file that declares a format keeps it. A tiled float fixture is still a float fixture,
// so the mask-side refusal of real-valued pixels fires -- the default reads no tag as unsigned, it
// does not rewrite a tag the file carries.
void test_2d_tagless_tiled_tiff_declared_float_still_refused_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagged_float_tiled", W, H, TD, 32, SAMPLEFORMAT_IEEEFP);

    NyxusGrayscaleTiffTileLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path,
        false /*permit_fp: a mask may not carry real-valued pixels*/, 0.0, 0.0, 0.0);

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    // the mask gate throws a std::string; the format dispatch throws a std::runtime_error, so the
    // type says which of the two refused the file
    EXPECT_THROW (ldr.loadTileFromFile (tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0), std::string);
}

// Negative: the default supplies a sample format, not blanket acceptance. A tagless tiled fixture
// at a bit depth neither loader can type is refused, on both sides, by the typed dispatch -- which
// names the unsigned format (1) and the bit depth, so the refusal is the one a file reaches only
// once the absent tag has been read as unsigned.
void test_2d_tagless_tiled_tiff_unsupported_depth_still_refused_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tagless_tiled_4bit", W, H, TD, 4 /*bits per sample*/, -1 /*no tag 339*/);

    // prescan side: the typed getter is resolved in the constructor, so that is where it refuses
    nyxus_ut_expect_refusal ([&] { RawTiffTileLoader ldr (fixture.path); },
        "RawTiffTileLoader", "sample format 1 with 4 bits per sample");

    // feature side: the constructor accepts the header and the typed read refuses
    NyxusGrayscaleTiffTileLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path,
        true /*permit_fp*/, 0.0, 0.0, 0.0);
    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    nyxus_ut_expect_refusal ([&] { ldr.loadTileFromFile (tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0); },
        "NyxusGrayscaleTiffTileLoader", "sample format 1 with 4 bits per sample");
}

// Negative: the sample format default does not extend to the sample count. A file that declares
// three samples is not grayscale, and both tile loaders refuse it in the constructor, where the
// greyscale test lives, with a std::runtime_error that says so.
void test_2d_tagless_tiled_tiff_multisample_still_refused_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tiled_3spp", W, H, TD, 16, -1 /*no tag 339*/, 3 /*samples per pixel*/);

    nyxus_ut_expect_refusal ([&] { NyxusGrayscaleTiffTileLoader<uint32_t> gldr (1 /*n_threads*/, fixture.path,
        true /*permit_fp*/, 0.0, 0.0, 0.0); }, "NyxusGrayscaleTiffTileLoader", "not greyscale");
    nyxus_ut_expect_refusal ([&] { RawTiffTileLoader rldr (fixture.path); },
        "RawTiffTileLoader", "not greyscale");
}

// Negative: both tile loaders test the layout first, so a strip file -- here tagless, which the
// strip loaders accept -- is refused by each tile constructor as not tiled, before any other header
// field is read.
void test_2d_tagless_tiled_tiff_strip_file_refused_by_tile_loaders_mechanics()
{
    const size_t W = 32, H = 32;
    NyxusUtSampleFormatStripTiff fixture ("tagless_strip_for_tile_loaders", W, H, -1 /*no tag 339*/);

    nyxus_ut_expect_refusal ([&] { NyxusGrayscaleTiffTileLoader<uint32_t> gldr (1 /*n_threads*/, fixture.path,
        true /*permit_fp*/, 0.0, 0.0, 0.0); }, "NyxusGrayscaleTiffTileLoader", "not tiled");
    nyxus_ut_expect_refusal ([&] { RawTiffTileLoader rldr (fixture.path); },
        "RawTiffTileLoader", "not tiled");
}

// A declared SAMPLEFORMAT_VOID (4) says what an absent tag says, so it reads as unsigned integer on
// both sides rather than being refused. This is the same fixture shape as the tagless one, with the
// tag present and set to 4.
void test_2d_declared_void_format_tiled_read_as_unsigned_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tiled_void_fmt", W, H, TD, 16, SAMPLEFORMAT_VOID);

    // feature side
    NyxusGrayscaleTiffTileLoader<uint32_t> gldr (1 /*n_threads*/, fixture.path,
        true /*permit_fp*/, 0.0, 0.0, 0.0);
    const size_t tw = gldr.tileWidth(0),
        th = gldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    ASSERT_NO_THROW (gldr.loadTileFromFile (tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    for (size_t r = 0; r < th; r++)
        for (size_t c = 0; c < tw; c++)
            ASSERT_EQ ((*tile)[r * tw + c], (uint32_t) nyxus_ut_tagless_enc (c, r))
                << "at in-tile (" << c << "," << r << ")";

    // prescan side
    RawTiffTileLoader rldr (fixture.path);
    ASSERT_NO_THROW (rldr.loadTileFromFile (0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    ASSERT_EQ (rldr.get_uint32_pixel (1 * tw + 1), (uint32_t) nyxus_ut_tagless_enc (1, 1));
    rldr.free_tile();
}

// A declared SAMPLEFORMAT_VOID mask. VOID is 4, so the loader's mask gate on sample formats >= 3
// would refuse it as declared; it gets through because the constructor reads it as unsigned before
// any tile is read. The whole grid reads back pixel for pixel with real-valued pixels prohibited.
void test_2d_declared_void_format_tiled_mask_read_as_unsigned_mechanics()
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture ("tiled_void_fmt_mask", W, H, TD, 16, SAMPLEFORMAT_VOID);

    nyxus_ut_expect_grid_unsigned (fixture.path, false /*permit_fp: the mask configuration*/, W, H);
}

// Negative: a complex format is not untyped, it is a type no loader implements, so it stays as
// declared and is refused. COMPLEXINT (5) and COMPLEXIEEEFP (6) are both >= 3, so the mask gate on
// real-valued pixels refuses them there, and with real-valued pixels permitted the value reaches
// the format dispatch, which has no typed reader for it. On the prescan side the dispatch runs in the
// constructor, alongside the other header checks, so the refusal message is asserted to name the
// declared format. On the feature side the throw type separates the two: the mask gate throws a
// std::string, the dispatch a std::runtime_error.
static inline void nyxus_ut_complex_format_refused (const char* stem, int sf, int bps)
{
    const size_t W = 32, H = 32, TD = 16;
    NyxusUtTiledTiff fixture (stem, W, H, TD, bps, sf);

    // prescan side: the typed getter is resolved in the constructor, so that is where it refuses,
    // with a message naming the declared format
    const std::string declared = "sample format " + std::to_string (sf) + " with";
    nyxus_ut_expect_refusal ([&] { RawTiffTileLoader rldr (fixture.path); },
        "RawTiffTileLoader", declared.c_str());

    // the mask scenario
    NyxusGrayscaleTiffTileLoader<uint32_t> mldr (1 /*n_threads*/, fixture.path,
        false /*permit_fp: a mask may not carry real-valued pixels*/, 0.0, 0.0, 0.0);
    const size_t tw = mldr.tileWidth(0),
        th = mldr.tileHeight(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th);

    EXPECT_THROW (mldr.loadTileFromFile (tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0), std::string);

    // and the intensity scenario, where the same format has no typed reader
    NyxusGrayscaleTiffTileLoader<uint32_t> ildr (1 /*n_threads*/, fixture.path,
        true /*permit_fp*/, 0.0, 0.0, 0.0);
    auto tile2 = std::make_shared<std::vector<uint32_t>>(tw * th);

    EXPECT_THROW (ildr.loadTileFromFile (tile2, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0), std::runtime_error);
}

void test_2d_tagless_tiled_tiff_declared_complex_format_refused_mechanics()
{
    nyxus_ut_complex_format_refused ("tiled_complexfp_fmt", SAMPLEFORMAT_COMPLEXIEEEFP, 64);
    nyxus_ut_complex_format_refused ("tiled_complexint_fmt", SAMPLEFORMAT_COMPLEXINT, 32);
}

// Agreement pin: a strip fixture that omits tag 339, and only that tag, reads back unsigned on both
// strip loaders, which is the reading the tile loaders above are held to. RawTiffStripLoader's
// buffer is released by free_tile(), so the read is paired with exactly one.
void test_2d_tagless_strip_tiff_accepted_mechanics()
{
    const size_t W = 32, H = 32;
    NyxusUtSampleFormatStripTiff fixture ("tagless_strip", W, H, -1 /*no tag 339*/);

    // feature side
    NyxusGrayscaleTiffStripLoader<uint32_t> gldr (1 /*n_threads*/, fixture.path);

    ASSERT_EQ (gldr.fullWidth(0), W);
    ASSERT_EQ (gldr.fullHeight(0), H);

    const size_t tw = gldr.tileWidth(0),
        th = gldr.tileHeight(0),
        td = gldr.tileDepth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th * td);

    ASSERT_NO_THROW (gldr.loadTileFromFile (tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    for (size_t y = 0; y < std::min (th, H); y++)
        for (size_t x = 0; x < std::min (tw, W); x++)
            ASSERT_EQ ((*tile)[y * tw + x], (uint32_t) nyxus_ut_tagless_enc (x, y))
                << "at (" << x << "," << y << ")";

    // prescan side
    RawTiffStripLoader rldr (1 /*n_threads*/, fixture.path);
    ASSERT_NO_THROW (rldr.loadTileFromFile (0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    for (size_t y = 0; y < std::min (th, H); y++)
        for (size_t x = 0; x < std::min (tw, W); x++)
            ASSERT_EQ (rldr.get_uint32_pixel (y * tw + x), (uint32_t) nyxus_ut_tagless_enc (x, y))
                << "at (" << x << "," << y << ")";
    rldr.free_tile();
}

// Agreement pin: the strip loaders read a declared SAMPLEFORMAT_VOID (4) as unsigned integer, the
// same as an absent tag, which is the reading the tile loaders above are held to.
void test_2d_declared_void_format_strip_read_as_unsigned_mechanics()
{
    const size_t W = 32, H = 32;
    NyxusUtSampleFormatStripTiff fixture ("strip_void_fmt", W, H, SAMPLEFORMAT_VOID);

    // feature side
    NyxusGrayscaleTiffStripLoader<uint32_t> gldr (1 /*n_threads*/, fixture.path);
    const size_t tw = gldr.tileWidth(0),
        th = gldr.tileHeight(0),
        td = gldr.tileDepth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(tw * th * td);

    ASSERT_NO_THROW (gldr.loadTileFromFile (tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    ASSERT_EQ ((*tile)[1 * tw + 1], (uint32_t) nyxus_ut_tagless_enc (1, 1));

    // prescan side
    RawTiffStripLoader rldr (1 /*n_threads*/, fixture.path);
    ASSERT_NO_THROW (rldr.loadTileFromFile (0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    ASSERT_EQ (rldr.get_uint32_pixel (1 * tw + 1), (uint32_t) nyxus_ut_tagless_enc (1, 1));
    rldr.free_tile();
}

// Negative, the strip twin of the tiled complex-format refusal: the strip loaders keep a declared
// COMPLEXINT (5) or COMPLEXIEEEFP (6) as declared and refuse it, so a strip file and a tiled file
// declaring a complex format are refused alike. RawTiffStripLoader resolves its typed getter in the
// constructor and refuses there; NyxusGrayscaleTiffStripLoader has no mask gate, so its typed read
// is what refuses. Both refusals name the declared format.
static inline void nyxus_ut_complex_format_strip_refused (const char* stem, int sf, int bps)
{
    const size_t W = 32, H = 32;
    NyxusUtSampleFormatStripTiff fixture (stem, W, H, sf, bps);

    const std::string declared = "sample format " + std::to_string (sf) + " with";

    // prescan side
    nyxus_ut_expect_refusal ([&] { RawTiffStripLoader rldr (1 /*n_threads*/, fixture.path); },
        "RawTiffStripLoader", declared.c_str());

    // feature side
    NyxusGrayscaleTiffStripLoader<uint32_t> gldr (1 /*n_threads*/, fixture.path);
    auto tile = std::make_shared<std::vector<uint32_t>>(gldr.tileWidth(0) * gldr.tileHeight(0) * gldr.tileDepth(0));

    nyxus_ut_expect_refusal ([&] { gldr.loadTileFromFile (tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0); },
        "NyxusGrayscaleTiffStripLoader", declared.c_str());
}

void test_2d_declared_complex_format_strip_refused_mechanics()
{
    nyxus_ut_complex_format_strip_refused ("strip_complexfp_fmt", SAMPLEFORMAT_COMPLEXIEEEFP, 64);
    nyxus_ut_complex_format_strip_refused ("strip_complexint_fmt", SAMPLEFORMAT_COMPLEXINT, 32);
}
