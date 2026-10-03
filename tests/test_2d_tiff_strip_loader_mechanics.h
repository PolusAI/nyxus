#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
    #include <process.h>    // _getpid
#else
    #include <unistd.h>     // getpid
#endif

#include "../src/nyx/raw_tiff.h"         // RawTiffStripLoader, and the <tiffio.h>, <cstdint>, <cstring>, <memory>, <string>, <vector> it includes
#include "../src/nyx/grayscale_tiff.h"   // NyxusGrayscaleTiffStripLoader
#include "../src/nyx/helpers/fsystem.h"

//
// A strip-based (non-tiled) TIFF is read by two loaders, and both are tested here over the same
// runtime-written fixtures:
//   - RawTiffStripLoader is the prescan-side reader: RawImageLoader instantiates it, slideprops.cpp
//     prescans through it, and both the CLI and the Python API prescan.
//   - NyxusGrayscaleTiffStripLoader is the featurize-side reader: ImageLoader instantiates it for
//     both the intensity and the mask image, and every feature pass reads its pixels through it.
// Both use a STRIP_TILE_WIDTH x STRIP_TILE_HEIGHT = 1024 x 1024 tile grid, so an image within 1024
// in both axes is a single tile at (0,0) and every tile index below is the one that matters: these
// fixtures are sized past 1024 in one axis and in both, which is the only way to reach a tile grid
// with more than one cell.
//
// What the RawTiffStripLoader tests pin down:
//   - tile (i,j) carries image rows [i*1024, ...) x columns [j*1024, ...), not rows 0.. of every tile;
//   - the buffer's pitch is the tile width, which is what get_uint32_pixel(r*tw + c) addresses,
//     counted in bytes of one sample at 8, 16 and 32 bits and in float32;
//   - a partial edge tile zero-fills the part of the buffer the image does not reach;
//   - a compressed file, whose codec cannot seek within a strip, reads every tile of the grid, in
//     any order;
//   - free_tile() between tiles is what RawImageLoader does per tile, and the buffer stays valid
//     for the next read;
//   - a tile index past the grid is refused by the out-of-grid guard rather than read out of bounds;
//   - a strip that does not decode makes the tile read throw this loader's own TIFFReadScanline
//     error, and the same loader still reads a tile that does not reach the damage.
//
// What the NyxusGrayscaleTiffStripLoader tests pin down:
//   - in a compressed file, every tile of the grid carries the pixels of its own image rows and
//     columns, whatever the strip height -- including a strip height that does not divide 1024, a
//     single strip spanning the whole image, and a grid that is past 1024 in height only;
//   - the same holds when the tiles are read out of order;
//   - a strip that does not decode makes the tile read throw, naming the row, rather than hand
//     back the previous contents of the scanline buffer as pixels.
//
// What both are held to together: a row-major read of the grid decodes every image row once, the
// tiles of a tile row sharing its scanlines and each tile row continuing the strip the one above
// left the decoder in.
//

// pixel value at image (col, row); row-dependent so a tile reading the wrong rows is detectable
static inline uint32_t nyxus_ut_strip_enc (size_t col, size_t row)
{
    return (uint32_t) ((col * 7 + row * 3) % 65536);
}

// The value a fixture stores at image (col, row) of directory `page`, in a sample `bitsPerSample`
// wide: nyxus_ut_strip_enc offset by the page, so that two directories carry different pixels, and
// reduced modulo 256 in an 8-bit sample, so that it survives the narrowing and stays row- and
// column-dependent. Page 0 at 16 or 32 bits is nyxus_ut_strip_enc itself. Every value is below
// 2^16, so a 32-bit float holds it exactly.
static inline uint32_t nyxus_ut_strip_value (size_t col, size_t row, uint16_t bitsPerSample = 16, size_t page = 0)
{
    const uint32_t v = (uint32_t) ((nyxus_ut_strip_enc (col, row) + page * 1000) % 65536);
    return bitsPerSample == 8 ? v % 256 : v;
}

// Write a strip-based (non-tiled), single-channel grayscale TIFF carrying nyxus_ut_strip_value in
// each of `pages` directories. The sample is uint8, uint16 (the default), uint32 or float32.
// SAMPLEFORMAT is set explicitly: the TIFF 6.0 default is SAMPLEFORMAT_UINT, but a writer that omits
// the tag leaves the loader to infer it, and the fixture is about tile addressing, not inference.
// The default is uncompressed with 16 rows per strip, so the file is genuinely strip-based.
static inline void nyxus_ut_write_strip_tiff (const std::string& path, size_t w, size_t h,
    uint16_t compression = COMPRESSION_NONE, uint32_t rowsPerStrip = 16,
    uint16_t bitsPerSample = 16, uint16_t sampleFormat = SAMPLEFORMAT_UINT, uint16_t pages = 1)
{
    const bool supported = (sampleFormat == SAMPLEFORMAT_UINT && (bitsPerSample == 8 || bitsPerSample == 16 || bitsPerSample == 32))
        || (sampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 32);
    if (!supported)
        throw std::runtime_error("nyxus_ut_write_strip_tiff: unsupported sample format " + std::to_string(sampleFormat)
            + " at " + std::to_string(bitsPerSample) + " bits");

    TIFF* t = TIFFOpen (path.c_str(), "w");
    if (t == nullptr)
        throw std::runtime_error("could not create TIFF: " + path);

    const size_t sampleBytes = bitsPerSample / 8;
    std::vector<uint8_t> row (w * sampleBytes);

    for (uint16_t page = 0; page < pages; page++)
    {
        TIFFSetField (t, TIFFTAG_IMAGEWIDTH, (uint32_t) w);
        TIFFSetField (t, TIFFTAG_IMAGELENGTH, (uint32_t) h);
        TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
        TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, sampleFormat);
        TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField (t, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField (t, TIFFTAG_COMPRESSION, compression);
        TIFFSetField (t, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

        for (size_t y = 0; y < h; y++)
        {
            for (size_t x = 0; x < w; x++)
            {
                const uint32_t v = nyxus_ut_strip_value (x, y, bitsPerSample, page);
                uint8_t* dst = row.data() + x * sampleBytes;
                if (sampleFormat == SAMPLEFORMAT_IEEEFP)
                {
                    const float s = (float) v;
                    std::memcpy (dst, &s, sizeof(s));
                }
                else if (bitsPerSample == 8)
                    *dst = (uint8_t) v;
                else if (bitsPerSample == 16)
                {
                    const uint16_t s = (uint16_t) v;
                    std::memcpy (dst, &s, sizeof(s));
                }
                else
                    std::memcpy (dst, &v, sizeof(v));
            }
            if (TIFFWriteScanline (t, row.data(), (uint32_t) y, 0) < 0)
            {
                TIFFClose (t);
                throw std::runtime_error("TIFFWriteScanline failed at row " + std::to_string(y));
            }
        }

        // TIFFClose writes the last directory
        if (page + 1 < pages && TIFFWriteDirectory (t) != 1)
        {
            TIFFClose (t);
            throw std::runtime_error("TIFFWriteDirectory failed after page " + std::to_string(page));
        }
    }
    TIFFClose (t);
}

// The fixture path carries the process id: the test binary is run more than once at a time -- the
// plain build and the sanitizer build are separate processes over the same temp directory -- and
// the destructor below removes the file unconditionally, so a name shared between two runs would
// have each of them truncating or deleting the other's fixture mid-read.
static inline std::string nyxus_ut_strip_tiff_path (const char* stem)
{
#ifdef _WIN32
    const long pid = (long) _getpid();
#else
    const long pid = (long) getpid();
#endif
    return (fs::temp_directory_path() /
        (std::string("nyxus_ut_") + stem + "_p" + std::to_string(pid) + ".tif")).string();
}

// RAII for the runtime-written fixture: the tests assert mid-flight, so the file is removed by the
// destructor rather than at the end of the test body
struct NyxusUtStripTiff
{
    std::string path;

    NyxusUtStripTiff (const char* stem, size_t w, size_t h,
        uint16_t compression = COMPRESSION_NONE, uint32_t rowsPerStrip = 16,
        uint16_t bitsPerSample = 16, uint16_t sampleFormat = SAMPLEFORMAT_UINT, uint16_t pages = 1)
    {
        path = nyxus_ut_strip_tiff_path (stem);
        nyxus_ut_write_strip_tiff (path, w, h, compression, rowsPerStrip, bitsPerSample, sampleFormat, pages);
    }

    ~NyxusUtStripTiff()
    {
        std::error_code ec;
        fs::remove (path, ec);
    }
};

// Walk the whole tile grid of a strip TIFF of the given geometry and sample type and check every
// pixel against nyxus_ut_strip_value, freeing the tile between reads exactly as RawImageLoader does.
// A float sample is read through get_dpequiv_pixel, an integer one through get_uint32_pixel.
static inline void nyxus_ut_assert_strip_grid (const char* stem, size_t W, size_t H,
    uint16_t compression = COMPRESSION_NONE, uint32_t rowsPerStrip = 16,
    uint16_t bitsPerSample = 16, uint16_t sampleFormat = SAMPLEFORMAT_UINT)
{
    NyxusUtStripTiff fixture (stem, W, H, compression, rowsPerStrip, bitsPerSample, sampleFormat);

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
            ASSERT_NO_THROW (ldr.loadTileFromFile (tr, tc, 0, 0, 0, 0)) << stem << " tile (" << tr << "," << tc << ")";

            const size_t row0 = tr * th,
                col0 = tc * tw,
                validH = (std::min) (th, H - row0),
                validW = (std::min) (tw, W - col0);

            for (size_t r = 0; r < th; r++)
                for (size_t c = 0; c < tw; c++)
                {
                    // inside the image: the pixel of global (col0+c, row0+r); outside: zero fill
                    const uint32_t expected = (r < validH && c < validW) ?
                        nyxus_ut_strip_value (col0 + c, row0 + r, bitsPerSample) : 0u;
                    if (sampleFormat == SAMPLEFORMAT_IEEEFP)
                        ASSERT_EQ (ldr.get_dpequiv_pixel (r * tw + c), (double) expected)
                            << stem << " tile (" << tr << "," << tc << ") at in-tile (" << c << "," << r << ")";
                    else
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

// The tile is pitched and offset in bytes of one sample, so every sample width addresses it
// differently. A 16-bit sample is 2 bytes, so a byte width fixed at 2 addresses it correctly; at
// uint32, float32 and uint8 a pitch or offset in the wrong byte width lands on the wrong pixels.
// Each width runs whether or not an earlier one failed, and uint8, whose tile buffer is the
// smallest, runs last.
void test_2d_raw_tiff_strip_loader_sample_width_grid_mechanics()
{
    EXPECT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x300_u32", 1100, 300, COMPRESSION_NONE, 16, 32, SAMPLEFORMAT_UINT));
    EXPECT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x300_f32", 1100, 300, COMPRESSION_NONE, 16, 32, SAMPLEFORMAT_IEEEFP));
    EXPECT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x300_u8", 1100, 300, COMPRESSION_NONE, 16, 8, SAMPLEFORMAT_UINT));
}

// A compressed striped TIFF: a codec decodes a strip only from its first row forward and cannot
// skip rows it has not decoded, so a tile row whose first row is not where the decoder stands has
// to start its strip over. The strip heights are chosen so that row 1024 -- where the second tile
// row starts -- is in the middle of a strip: 100 rows per strip does not divide 1024, and a single
// strip spanning the whole image puts every tile inside the same one. LZW and PackBits are both
// built into libtiff, and neither can seek.
void test_2d_raw_tiff_strip_loader_compressed_grid_mechanics()
{
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x1100_lzw100", 1100, 1100, COMPRESSION_LZW, 100));
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x1100_lzw1", 1100, 1100, COMPRESSION_LZW, 1100));
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid ("strip1100x1100_pb100", 1100, 1100, COMPRESSION_PACKBITS, 100));
}

// Out of order: a compressed file is read back to front and the first tile again, so every tile
// is reached from a reader positioned past it, in a later strip or later in the same strip.
void test_2d_raw_tiff_strip_loader_compressed_revisit_mechanics()
{
    const size_t W = 1100, H = 1100;
    NyxusUtStripTiff fixture ("strip1100x1100_lzw_revisit", W, H, COMPRESSION_LZW, 100);

    auto ldr = RawTiffStripLoader (1 /*n_threads*/, fixture.path);
    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);

    const size_t order[][2] = { {1,1}, {1,0}, {0,1}, {0,0}, {1,1} };
    for (const auto& t : order)
    {
        ASSERT_NO_THROW (ldr.loadTileFromFile (t[0], t[1], 0, 0, 0, 0)) << "tile (" << t[0] << "," << t[1] << ")";

        const size_t row0 = t[0] * th,
            col0 = t[1] * tw,
            validH = (std::min) (th, H - row0),
            validW = (std::min) (tw, W - col0);

        for (size_t r = 0; r < validH; r++)
            for (size_t c = 0; c < validW; c++)
                ASSERT_EQ (ldr.get_uint32_pixel (r * tw + c), nyxus_ut_strip_enc (col0 + c, row0 + r))
                    << "tile (" << t[0] << "," << t[1] << ") at in-tile (" << c << "," << r << ")";

        ldr.free_tile();
    }
}

// Expect loadTileFromFile(row, col) to be refused by the out-of-grid guard itself: a
// std::runtime_error whose message says the tile is outside the image, so that any other libtiff
// error on the way does not count as the refusal.
static inline void nyxus_ut_expect_tile_outside_image (RawTiffStripLoader& ldr, size_t row, size_t col)
{
    EXPECT_THROW (
        {
            try
            {
                ldr.loadTileFromFile (row, col, 0, 0, 0, 0);
            }
            catch (const std::runtime_error& e)
            {
                EXPECT_NE (std::string(e.what()).find("is outside the image"), std::string::npos)
                    << "tile (" << row << "," << col << ") refused for another reason: " << e.what();
                throw;
            }
        },
        std::runtime_error) << "tile (" << row << "," << col << ")";
}

// Negative: a tile index past the grid names no image rows or columns and is refused.
void test_2d_raw_tiff_strip_loader_tile_out_of_grid_refused_mechanics()
{
    NyxusUtStripTiff fixture ("strip1100x1100_oob", 1100, 1100);

    auto ldr = RawTiffStripLoader (1 /*n_threads*/, fixture.path);

    nyxus_ut_expect_tile_outside_image (ldr, 2, 0);   // tile row past the last row of the grid
    nyxus_ut_expect_tile_outside_image (ldr, 0, 2);   // tile column past the last column
    nyxus_ut_expect_tile_outside_image (ldr, 9, 9);   // both

    ldr.free_tile();
}

// Read tile (tr, tc) through the featurize-side loader and check every in-image pixel against
// nyxus_ut_strip_enc. Only the in-image part is checked: this loader writes the image's own rows
// and columns and leaves the rest of the buffer to the caller, which skips pixels past the image.
static inline void nyxus_ut_assert_grayscale_strip_tile (NyxusGrayscaleTiffStripLoader<uint32_t>& ldr,
    std::shared_ptr<std::vector<uint32_t>>& tile, const char* stem, size_t W, size_t H, size_t tr, size_t tc)
{
    ASSERT_NO_THROW (ldr.loadTileFromFile (tile, tr, tc, 0, 0, 0, 0)) << stem << " tile (" << tr << "," << tc << ")";

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0),
        row0 = tr * th,
        col0 = tc * tw,
        validH = (std::min) (th, H - row0),
        validW = (std::min) (tw, W - col0);

    for (size_t r = 0; r < validH; r++)
        for (size_t c = 0; c < validW; c++)
            ASSERT_EQ ((*tile)[r * tw + c], nyxus_ut_strip_enc (col0 + c, row0 + r))
                << stem << " tile (" << tr << "," << tc << ") at in-tile (" << c << "," << r << ")";
}

// Walk the whole tile grid of a strip TIFF of the given geometry, in row-major order -- the order
// the feature passes visit it in -- and check every pixel.
static inline void nyxus_ut_assert_grayscale_strip_grid (const char* stem, size_t W, size_t H,
    uint16_t compression, uint32_t rowsPerStrip)
{
    NyxusUtStripTiff fixture (stem, W, H, compression, rowsPerStrip);

    NyxusGrayscaleTiffStripLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path);

    ASSERT_EQ (ldr.fullWidth(0), W) << stem;
    ASSERT_EQ (ldr.fullHeight(0), H) << stem;

    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0),
        nCols = (W + tw - 1) / tw,
        nRows = (H + th - 1) / th;
    ASSERT_GT (nCols * nRows, (size_t)1) << stem << ": geometry does not produce a multi-tile grid";

    auto tile = std::make_shared<std::vector<uint32_t>> (tw * th * ldr.tileDepth(0));

    for (size_t tr = 0; tr < nRows; tr++)
        for (size_t tc = 0; tc < nCols; tc++)
            ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_grayscale_strip_tile (ldr, tile, stem, W, H, tr, tc));
}

// A compressed striped TIFF read by the featurize-side loader. The second tile row starts at image
// row 1024, inside a strip: 100 rows per strip does not divide 1024, and a single strip puts every
// tile inside the same one. The 300 x 1100 geometry is past 1024 in height only, so its tiles are
// read top to bottom and its second tile row continues the strip the first one left off in.
void test_2d_grayscale_tiff_strip_loader_compressed_grid_mechanics()
{
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_grayscale_strip_grid ("gstrip1100x1100_lzw100", 1100, 1100, COMPRESSION_LZW, 100));
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_grayscale_strip_grid ("gstrip1100x1100_lzw1", 1100, 1100, COMPRESSION_LZW, 1100));
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_grayscale_strip_grid ("gstrip1100x1100_pb100", 1100, 1100, COMPRESSION_PACKBITS, 100));
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_grayscale_strip_grid ("gstrip300x1100_lzw100", 300, 1100, COMPRESSION_LZW, 100));
}

// Out of order: a compressed file is read back to front and the first tile again, so every tile
// is reached from a reader positioned past it, in a later strip or later in the same strip.
void test_2d_grayscale_tiff_strip_loader_compressed_revisit_mechanics()
{
    const size_t W = 1100, H = 1100;
    const char* stem = "gstrip1100x1100_lzw_revisit";
    NyxusUtStripTiff fixture (stem, W, H, COMPRESSION_LZW, 100);

    NyxusGrayscaleTiffStripLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path);
    auto tile = std::make_shared<std::vector<uint32_t>> (ldr.tileWidth(0) * ldr.tileHeight(0) * ldr.tileDepth(0));

    const size_t order[][2] = { {1,1}, {1,0}, {0,1}, {0,0}, {1,1} };
    for (const auto& t : order)
        ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_grayscale_strip_tile (ldr, tile, stem, W, H, t[0], t[1]));
}

// Walk the whole tile grid row-major through both strip loaders, the order the prescan and the
// feature passes read it in, and expect each loader to have decoded every image row exactly once.
// Reading the last tile again decodes nothing more.
static inline void nyxus_ut_assert_strip_grid_decodes_each_row_once (const char* stem, size_t W, size_t H,
    uint16_t compression, uint32_t rowsPerStrip)
{
    NyxusUtStripTiff fixture (stem, W, H, compression, rowsPerStrip);

    RawTiffStripLoader raw (1 /*n_threads*/, fixture.path);
    NyxusGrayscaleTiffStripLoader<uint32_t> grey (1 /*n_threads*/, fixture.path);

    const size_t tw = raw.tileWidth(0),
        th = raw.tileHeight(0),
        nCols = (W + tw - 1) / tw,
        nRows = (H + th - 1) / th;
    ASSERT_GT (nCols, (size_t)1) << stem << ": geometry does not produce more than one tile per tile row";

    auto tile = std::make_shared<std::vector<uint32_t>> (grey.tileWidth(0) * grey.tileHeight(0) * grey.tileDepth(0));

    for (size_t tr = 0; tr < nRows; tr++)
        for (size_t tc = 0; tc < nCols; tc++)
        {
            ASSERT_NO_THROW (raw.loadTileFromFile (tr, tc, 0, 0, 0, 0)) << stem << " tile (" << tr << "," << tc << ")";
            raw.free_tile();
            ASSERT_NO_THROW (grey.loadTileFromFile (tile, tr, tc, 0, 0, 0, 0)) << stem << " tile (" << tr << "," << tc << ")";
        }

    EXPECT_EQ (raw.scanlines_decoded(), H) << stem << ": RawTiffStripLoader";
    EXPECT_EQ (grey.scanlines_decoded(), H) << stem << ": NyxusGrayscaleTiffStripLoader";

    ASSERT_NO_THROW (raw.loadTileFromFile (nRows - 1, nCols - 1, 0, 0, 0, 0));
    raw.free_tile();
    ASSERT_NO_THROW (grey.loadTileFromFile (tile, nRows - 1, nCols - 1, 0, 0, 0, 0));
    EXPECT_EQ (raw.scanlines_decoded(), H) << stem << ": RawTiffStripLoader, last tile read again";
    EXPECT_EQ (grey.scanlines_decoded(), H) << stem << ": NyxusGrayscaleTiffStripLoader, last tile read again";
}

// The decode cost of a grid read: the tiles of one tile row share its scanlines, and a top-to-bottom
// scan of a compressed file continues each strip where the tile row above left the decoder rather
// than restarting it. A single strip spanning the image is where a restart per tile costs the most
// -- each tile row would decode the image from row 0 again -- and 100 rows per strip puts row 1024,
// where the second tile row starts, in the middle of a strip. Uncompressed, the column tiles still
// share their tile row's scanlines.
void test_2d_tiff_strip_loaders_decode_each_row_once_mechanics()
{
    EXPECT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid_decodes_each_row_once ("dstrip2100x2100_lzw1", 2100, 2100, COMPRESSION_LZW, 2100));
    EXPECT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid_decodes_each_row_once ("dstrip2100x2100_lzw100", 2100, 2100, COMPRESSION_LZW, 100));
    EXPECT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid_decodes_each_row_once ("dstrip2100x2100_pb100", 2100, 2100, COMPRESSION_PACKBITS, 100));
    EXPECT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_grid_decodes_each_row_once ("dstrip2100x2100_none", 2100, 2100, COMPRESSION_NONE, 16));
}

// Overwrite 64 bytes three quarters of the way into strip `strip` of a single-directory TIFF with
// 0xFF. In an LZW strip that is a run of codes past the end of the code table, which the decoder
// refuses. A strip under 256 bytes is refused, so that the 64 bytes always end inside the strip.
// Returns false if the strip cannot be located.
static inline bool nyxus_ut_corrupt_strip (const std::string& path, uint32_t strip)
{
    uint64_t offset = 0, nbytes = 0;
    {
        TIFF* t = TIFFOpen (path.c_str(), "r");
        if (t == nullptr)
            return false;
        uint64_t* offsets = nullptr;
        uint64_t* counts = nullptr;
        const bool ok = TIFFGetField (t, TIFFTAG_STRIPOFFSETS, &offsets) == 1
            && TIFFGetField (t, TIFFTAG_STRIPBYTECOUNTS, &counts) == 1
            && strip < TIFFNumberOfStrips (t);
        if (ok)
        {
            offset = offsets[strip];
            nbytes = counts[strip];
        }
        TIFFClose (t);
        if (!ok || nbytes < 256)
            return false;
    }

    std::fstream f (path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f)
        return false;
    const std::string junk (64, '\xFF');
    f.seekp ((std::streamoff) (offset + nbytes * 3 / 4));
    f.write (junk.data(), (std::streamsize) junk.size());
    return (bool) f;
}

// Precondition of the corrupt-strip tests: libtiff itself refuses to decode strip `strip`, so the
// fixture is broken the way the test means it to be.
static inline void nyxus_ut_assert_strip_does_not_decode (const std::string& path, uint32_t strip)
{
    TIFF* t = TIFFOpen (path.c_str(), "r");
    ASSERT_NE (t, nullptr);
    std::vector<uint8_t> stripBuf (TIFFStripSize (t));
    const tmsize_t decoded = TIFFReadEncodedStrip (t, strip, stripBuf.data(), (tmsize_t) stripBuf.size());
    TIFFClose (t);
    ASSERT_EQ (decoded, (tmsize_t)-1) << "strip " << strip << " still decodes; the fixture is not corrupt";
}

// The first row in [fromRow, toRow) that libtiff cannot read when the rows are read in order from
// fromRow, the first row of a strip; toRow if every one of them reads.
static inline uint32_t nyxus_ut_first_unreadable_row (const std::string& path, uint32_t fromRow, uint32_t toRow)
{
    TIFF* t = TIFFOpen (path.c_str(), "r");
    if (t == nullptr)
        return fromRow;
    std::vector<uint8_t> line (TIFFScanlineSize (t));
    uint32_t r = fromRow;
    while (r < toRow && TIFFReadScanline (t, line.data(), r) == 1)
        r++;
    TIFFClose (t);
    return r;
}

// Negative: an LZW strip TIFF past 1024 px whose strip 10 -- image rows 1000..1099, which the
// second tile row starts inside -- does not decode. Reading a tile that needs that strip throws a
// std::runtime_error naming TIFFReadScanline and the row; it is never returned as pixels.
void test_2d_grayscale_tiff_strip_loader_corrupt_strip_refused_mechanics()
{
    const size_t W = 1100, H = 1100;
    const uint32_t rowsPerStrip = 100, badStrip = 10;
    NyxusUtStripTiff fixture ("gstrip1100x1100_lzw_corrupt", W, H, COMPRESSION_LZW, rowsPerStrip);
    ASSERT_TRUE (nyxus_ut_corrupt_strip (fixture.path, badStrip)) << "could not corrupt " << fixture.path;
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_does_not_decode (fixture.path, badStrip));

    NyxusGrayscaleTiffStripLoader<uint32_t> ldr (1 /*n_threads*/, fixture.path);
    auto tile = std::make_shared<std::vector<uint32_t>> (ldr.tileWidth(0) * ldr.tileHeight(0) * ldr.tileDepth(0));

    // tile (1,0) covers image rows 1024..1099, all inside the damaged strip
    EXPECT_THROW (
        {
            try
            {
                ldr.loadTileFromFile (tile, 1, 0, 0, 0, 0, 0);
            }
            catch (const std::runtime_error& e)
            {
                EXPECT_NE (std::string(e.what()).find("TIFFReadScanline failed at row"), std::string::npos)
                    << "refused for another reason: " << e.what();
                throw;
            }
        },
        std::runtime_error);
}

// Negative, prescan side: the same damaged file read by RawTiffStripLoader. Tile (1,0) starts at
// image row 1024, inside strip 10, so the loader decodes that strip from row 1000 and throws a
// std::runtime_error from its own TIFFReadScanline check when the decoder reaches the damage. The
// message is this loader's, not NyxusGrayscaleTiffStripLoader's. The damage sits three quarters of
// the way into the strip's bytes, below row 1023, so tile (0,0) -- rows 0..1023, which reads the
// top of strip 10 -- still loads and carries its own pixels.
void test_2d_raw_tiff_strip_loader_corrupt_strip_refused_mechanics()
{
    const size_t W = 1100, H = 1100;
    const uint32_t rowsPerStrip = 100, badStrip = 10;
    NyxusUtStripTiff fixture ("strip1100x1100_lzw_corrupt", W, H, COMPRESSION_LZW, rowsPerStrip);
    ASSERT_TRUE (nyxus_ut_corrupt_strip (fixture.path, badStrip)) << "could not corrupt " << fixture.path;
    ASSERT_NO_FATAL_FAILURE (nyxus_ut_assert_strip_does_not_decode (fixture.path, badStrip));

    // precondition: the first row libtiff cannot decode is past the last row of tile (0,0) and
    // inside strip 10
    const uint32_t stripStart = badStrip * rowsPerStrip,
        badRow = nyxus_ut_first_unreadable_row (fixture.path, stripStart, stripStart + rowsPerStrip);
    ASSERT_GT (badRow, (uint32_t)1023) << "the damage reaches tile (0,0)";
    ASSERT_LT (badRow, stripStart + rowsPerStrip) << "every row of strip " << badStrip << " decodes";

    auto ldr = RawTiffStripLoader (1 /*n_threads*/, fixture.path);

    EXPECT_THROW (
        {
            try
            {
                ldr.loadTileFromFile (1, 0, 0, 0, 0, 0);
            }
            catch (const std::runtime_error& e)
            {
                const std::string msg = e.what();
                EXPECT_NE (msg.find("calling TIFFReadScanline(row = "), std::string::npos)
                    << "refused for another reason: " << msg;
                EXPECT_EQ (msg.find("Tile Loader ERROR"), std::string::npos)
                    << "the featurize-side loader's message: " << msg;
                throw;
            }
        },
        std::runtime_error);

    // positive: the loader reads the undamaged tile after the refusal
    const size_t tw = ldr.tileWidth(0),
        th = ldr.tileHeight(0);
    ASSERT_NO_THROW (ldr.loadTileFromFile (0, 0, 0, 0, 0, 0));
    for (size_t r = 0; r < th; r++)
        for (size_t c = 0; c < tw; c++)
            ASSERT_EQ (ldr.get_uint32_pixel (r * tw + c), nyxus_ut_strip_enc (c, r)) << "tile (0,0) at (" << c << "," << r << ")";

    ldr.free_tile();
}
