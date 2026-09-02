#pragma once

#include <gtest/gtest.h>

#ifdef OMEZARR_SUPPORT

#include <memory>
#include <vector>
#include "../src/nyx/omezarr.h"
#include "../src/nyx/raw_omezarr.h"
#include "../src/nyx/helpers/fsystem.h"

// The OME-Zarr test datasets under tests/data/omezarr are generated with bfio
// (see tests/data/omezarr/README.md). They are zarr-v2 stores with the 5D
// (T, C, Z, Y, X) layout that the nyxus z5-based loaders expect.
//
//   test.ome.zarr  : 512 x 512  uint16, single 1024x1024 chunk (one tile)
//                    pixel value = (row + col) % 65536
//   multi.ome.zarr : 1500 x 1200 uint16, 1024x1024 chunks (2x2 tile grid,
//                    partial edge tiles) ; value = (row*7 + col*3) % 65536

static inline fs::path omezarr_data_path(const char* name)
{
    fs::path p(__FILE__);
    fs::path pp = p.parent_path();
    fs::path rel(std::string("/data/omezarr/") + name);
    return fs::path(pp.string() + rel.make_preferred().string());
}

// ---------------------------------------------------------------------------
// NyxusOmeZarrLoader (AbstractTileLoader) tests
// ---------------------------------------------------------------------------

// Geometry: dimensions and chunk/tile sizes are read correctly from metadata.
void test_2d_omezarr_tileloader_geometry_mechanics()
{
    fs::path ds = omezarr_data_path("test.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());

    ASSERT_EQ(ldr.fullHeight(0), 512u);
    ASSERT_EQ(ldr.fullWidth(0), 512u);
    ASSERT_EQ(ldr.fullDepth(0), 1u);
    ASSERT_EQ(ldr.tileHeight(0), 1024u);
    ASSERT_EQ(ldr.tileWidth(0), 1024u);
    ASSERT_EQ(ldr.tileDepth(0), 1u);
    ASSERT_EQ(ldr.numberPyramidLevels(), 1u);
}

// Content: read the single tile and verify exact pixel values and the checksum.
void test_2d_omezarr_tileloader_content_mechanics()
{
    fs::path ds = omezarr_data_path("test.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());

    size_t th = ldr.tileHeight(0);
    size_t tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);

    ASSERT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0));

    const std::vector<uint32_t>& buf = *tile;

    // pixel value = (row + col) % 65536, laid out row-major with stride tileWidth
    ASSERT_EQ(buf[0 * tw + 0], 0u);        // (0,0)
    ASSERT_EQ(buf[0 * tw + 511], 511u);    // (0,511)
    ASSERT_EQ(buf[100 * tw + 50], 150u);   // (100,50)
    ASSERT_EQ(buf[511 * tw + 511], 1022u); // (511,511)

    // full-image checksum over the valid 512x512 region
    unsigned long long total = 0;
    for (size_t r = 0; r < ldr.fullHeight(0); ++r)
        for (size_t c = 0; c < ldr.fullWidth(0); ++c)
            total += buf[r * tw + c];
    ASSERT_EQ(total, 133955584ull);
}

// Multi-tile: 2x2 tile grid with partial edge tiles.
void test_2d_omezarr_tileloader_multitile_mechanics()
{
    fs::path ds = omezarr_data_path("multi.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());

    const size_t H = ldr.fullHeight(0);
    const size_t W = ldr.fullWidth(0);
    const size_t th = ldr.tileHeight(0);
    const size_t tw = ldr.tileWidth(0);
    ASSERT_EQ(H, 1500u);
    ASSERT_EQ(W, 1200u);

    const size_t nRows = (H + th - 1) / th; // 2
    const size_t nCols = (W + tw - 1) / tw; // 2
    ASSERT_EQ(nRows, 2u);
    ASSERT_EQ(nCols, 2u);

    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);

    unsigned long long total = 0;
    for (size_t tr = 0; tr < nRows; ++tr)
    {
        for (size_t tc = 0; tc < nCols; ++tc)
        {
            std::fill(tile->begin(), tile->end(), 0u);
            ASSERT_NO_THROW(ldr.loadTileFromFile(tile, tr, tc, 0, 0));

            const std::vector<uint32_t>& buf = *tile;
            const size_t row0 = tr * th;
            const size_t col0 = tc * tw;
            const size_t validH = std::min(th, H - row0);
            const size_t validW = std::min(tw, W - col0);

            for (size_t r = 0; r < validH; ++r)
            {
                for (size_t c = 0; c < validW; ++c)
                {
                    size_t gr = row0 + r;
                    size_t gc = col0 + c;
                    uint32_t expected = static_cast<uint32_t>((gr * 7 + gc * 3) % 65536);
                    ASSERT_EQ(buf[r * tw + c], expected)
                        << "mismatch at global (" << gr << "," << gc << ")";
                    total += buf[r * tw + c];
                }
            }
        }
    }
    ASSERT_EQ(total, 12681000000ull);
}

// ---------------------------------------------------------------------------
// RawOmezarrLoader (RawFormatLoader) tests
// ---------------------------------------------------------------------------

void test_2d_omezarr_raw_geometry_mechanics()
{
    fs::path ds = omezarr_data_path("test.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = RawOmezarrLoader(ds.string());

    ASSERT_EQ(ldr.fullHeight(0), 512u);
    ASSERT_EQ(ldr.fullWidth(0), 512u);
    ASSERT_EQ(ldr.fullDepth(0), 1u);
    ASSERT_EQ(ldr.tileHeight(0), 1024u);
    ASSERT_EQ(ldr.tileWidth(0), 1024u);
}

void test_2d_omezarr_raw_content_mechanics()
{
    fs::path ds = omezarr_data_path("test.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = RawOmezarrLoader(ds.string());
    const size_t tw = ldr.tileWidth(0);

    ASSERT_NO_THROW(ldr.loadTileFromFile(0, 0, 0, 0));

    // get_uint32_pixel indexes into the internal tile buffer (stride tileWidth)
    ASSERT_EQ(ldr.get_uint32_pixel(0 * tw + 0), 0u);
    ASSERT_EQ(ldr.get_uint32_pixel(0 * tw + 511), 511u);
    ASSERT_EQ(ldr.get_uint32_pixel(100 * tw + 50), 150u);
    ASSERT_EQ(ldr.get_uint32_pixel(511 * tw + 511), 1022u);

    // get_dpequiv_pixel returns the same values as double
    ASSERT_DOUBLE_EQ(ldr.get_dpequiv_pixel(100 * tw + 50), 150.0);

    unsigned long long total = 0;
    for (size_t r = 0; r < ldr.fullHeight(0); ++r)
        for (size_t c = 0; c < ldr.fullWidth(0); ++c)
            total += ldr.get_uint32_pixel(r * tw + c);
    ASSERT_EQ(total, 133955584ull);
}

// Multi-tile read through the RawFormatLoader path, including partial tiles.
void test_2d_omezarr_raw_multitile_mechanics()
{
    fs::path ds = omezarr_data_path("multi.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = RawOmezarrLoader(ds.string());
    const size_t H = ldr.fullHeight(0);
    const size_t W = ldr.fullWidth(0);
    const size_t th = ldr.tileHeight(0);
    const size_t tw = ldr.tileWidth(0);

    const size_t nRows = (H + th - 1) / th;
    const size_t nCols = (W + tw - 1) / tw;

    unsigned long long total = 0;
    for (size_t tr = 0; tr < nRows; ++tr)
    {
        for (size_t tc = 0; tc < nCols; ++tc)
        {
            ASSERT_NO_THROW(ldr.loadTileFromFile(tr, tc, 0, 0));
            const size_t row0 = tr * th;
            const size_t col0 = tc * tw;
            const size_t validH = std::min(th, H - row0);
            const size_t validW = std::min(tw, W - col0);
            for (size_t r = 0; r < validH; ++r)
            {
                for (size_t c = 0; c < validW; ++c)
                {
                    size_t gr = row0 + r, gc = col0 + c;
                    uint32_t expected = static_cast<uint32_t>((gr * 7 + gc * 3) % 65536);
                    ASSERT_EQ(ldr.get_uint32_pixel(r * tw + c), expected)
                        << "mismatch at global (" << gr << "," << gc << ")";
                    total += ldr.get_uint32_pixel(r * tw + c);
                }
            }
        }
    }
    ASSERT_EQ(total, 12681000000ull);
}

// ---------------------------------------------------------------------------
// Signed and real-valued datasets: the load-time intensity map
//
// Both loaders used to narrow every sample into the unsigned pipeline type -- an
// explicit static_cast in RawOmezarrLoader, an implicit one in the std::copy of
// NyxusOmeZarrLoader -- so a signed dataset's negatives wrapped and a real-valued
// dataset's fractions were dropped. Nothing recorded that conversion, so the prescan
// measured the slide's extrema on the converted values and no offset derived from them
// could undo it. The prescan now reads the sample as the file states it, and the tile
// loader applies the same offset / quantization map every other backend takes.
//
//   signed.ome.zarr : 32x32 int16,   value = -1013 + (row + col)   [-1013 .. -951]
//   float.ome.zarr  : 32x32 float32, value = -10.5 + 0.25*(row+col) [-10.5 .. 5.0]
// ---------------------------------------------------------------------------

// The prescan sees a signed dataset's negatives, rather than the ~4.29e9 they wrapped to.
void test_2d_omezarr_raw_signed_source_domain_mechanics()
{
    fs::path ds = omezarr_data_path("signed.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = RawOmezarrLoader(ds.string());
    const size_t tw = ldr.tileWidth(0);
    ASSERT_NO_THROW(ldr.loadTileFromFile(0, 0, 0, 0));

    // The slide minimum, and what an offset would have to be derived from
    ASSERT_DOUBLE_EQ(ldr.get_dpequiv_pixel(0 * tw + 0), -1013.0);
    ASSERT_DOUBLE_EQ(ldr.get_dpequiv_pixel(0 * tw + 31), -982.0);
    ASSERT_DOUBLE_EQ(ldr.get_dpequiv_pixel(31 * tw + 31), -951.0);

    double mn = ldr.get_dpequiv_pixel(0), mx = mn, sum = 0.0;
    for (size_t r = 0; r < ldr.fullHeight(0); ++r)
        for (size_t c = 0; c < ldr.fullWidth(0); ++c)
        {
            double v = ldr.get_dpequiv_pixel(r * tw + c);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
        }
    ASSERT_DOUBLE_EQ(mn, -1013.0);
    ASSERT_DOUBLE_EQ(mx, -951.0);
    ASSERT_DOUBLE_EQ(sum, -1005568.0);
}

// The prescan sees a real-valued dataset's fractions, and flags the dataset real-valued
// so the recorder reaches its quantization branch at all.
void test_2d_omezarr_raw_float_source_domain_mechanics()
{
    fs::path ds = omezarr_data_path("float.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = RawOmezarrLoader(ds.string());
    const size_t tw = ldr.tileWidth(0);
    ASSERT_TRUE(ldr.get_fp_pixels());
    ASSERT_NO_THROW(ldr.loadTileFromFile(0, 0, 0, 0));

    ASSERT_DOUBLE_EQ(ldr.get_dpequiv_pixel(0 * tw + 0), -10.5);   // not truncated to -10, nor to 0
    ASSERT_DOUBLE_EQ(ldr.get_dpequiv_pixel(0 * tw + 1), -10.25);  // the fraction survives
    ASSERT_DOUBLE_EQ(ldr.get_dpequiv_pixel(31 * tw + 31), 5.0);
}

// The tile loader shifts a signed dataset by the recorded offset instead of wrapping it,
// so 1 grey level == 1 intensity unit and the inverse recovers the source value.
void test_2d_omezarr_tileloader_signed_offset_map_mechanics()
{
    fs::path ds = omezarr_data_path("signed.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    // The offset ImageLoader::open hands over for this slide: floor(min) == -1013
    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string(), -1013.0);

    const size_t th = ldr.tileHeight(0), tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);
    ASSERT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0));
    const std::vector<uint32_t>& buf = *tile;

    ASSERT_EQ(buf[0 * tw + 0], 0u);      // the minimum lands on grey level 0, not on 4294966283
    ASSERT_EQ(buf[0 * tw + 1], 1u);      // 1 HU == 1 grey level
    ASSERT_EQ(buf[31 * tw + 31], 62u);   // the maximum spans the range
}

// A real-valued dataset takes the same min-max quantization a real-valued TIFF takes.
void test_2d_omezarr_tileloader_float_quantized_map_mechanics()
{
    fs::path ds = omezarr_data_path("float.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    // [fpmin, fpmax] = [-10.5, 5.0] onto [0, 10000], as the recorder derives them
    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string(), -10.5, 5.0, 1e4, true);

    const size_t th = ldr.tileHeight(0), tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);
    ASSERT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0));
    const std::vector<uint32_t>& buf = *tile;

    ASSERT_EQ(buf[0 * tw + 0], 0u);         // min endpoint -> 0
    ASSERT_EQ(buf[31 * tw + 31], 10000u);   // max endpoint -> the target dynamic range
    // (row+col) == 31 is the exact midpoint of the 62-step ramp -> DR/2
    ASSERT_EQ(buf[0 * tw + 31], 5000u);
}

// An unsigned dataset is opened with no offset and no quantization, so its grey levels
// are still its own values -- the case the blanket exemption did get right.
void test_2d_omezarr_tileloader_unsigned_unaffected_mechanics()
{
    fs::path ds = omezarr_data_path("test.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());

    const size_t th = ldr.tileHeight(0), tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);
    ASSERT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0));
    const std::vector<uint32_t>& buf = *tile;

    ASSERT_EQ(buf[0 * tw + 0], 0u);
    ASSERT_EQ(buf[100 * tw + 50], 150u);
    ASSERT_EQ(buf[511 * tw + 511], 1022u);
}

#endif // OMEZARR_SUPPORT
