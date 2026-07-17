#pragma once

#include <gtest/gtest.h>

#ifdef OMEZARR_SUPPORT

#include <memory>
#include <vector>
#include "../src/nyx/omezarr.h"
#include "../src/nyx/raw_omezarr.h"
#include "../src/nyx/image_loader.h"
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
void test_omezarr_tileloader_geometry()
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
void test_omezarr_tileloader_content()
{
    fs::path ds = omezarr_data_path("test.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());

    size_t th = ldr.tileHeight(0);
    size_t tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);

    ASSERT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));

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
void test_omezarr_tileloader_multitile()
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
            ASSERT_NO_THROW(ldr.loadTileFromFile(tile, tr, tc, 0, 0/*channel*/, 0/*timeframe*/, 0));

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

void test_raw_omezarr_geometry()
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

void test_raw_omezarr_content()
{
    fs::path ds = omezarr_data_path("test.ome.zarr");
    ASSERT_TRUE(fs::exists(ds));

    auto ldr = RawOmezarrLoader(ds.string());
    const size_t tw = ldr.tileWidth(0);

    ASSERT_NO_THROW(ldr.loadTileFromFile(0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));

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
void test_raw_omezarr_multitile()
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
            ASSERT_NO_THROW(ldr.loadTileFromFile(tr, tc, 0, 0/*channel*/, 0/*timeframe*/, 0));
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
// 5D (T,C,Z,Y,X) channel/timeframe addressability.
//
// dim5.ome.zarr (see gen_dim5.py) encodes every voxel as
//   value(x,y,z,c,t) = 1 + ((((t*C + c)*Z + z)*Y + y)*X + x),  C=3,Z=4,Y=6,X=8
// chunked one z/c/t-plane per chunk. Reading plane (z,c,t) must return exactly
// that plane's values; a loader that ignored C/T (offset pinned to {0,0,...},
// the pre-fix behavior) would return the c=0/t=0 plane for every (c,t).
// ---------------------------------------------------------------------------

static inline uint32_t dim5_enc(int x, int y, int z, int c, int t)
{
    const int C = 3, Z = 4, Y = 6, X = 8;
    return static_cast<uint32_t>(1 + ((((t * C + c) * Z + z) * Y + y) * X + x));
}

// AbstractTileLoader stack. (T,C,Z) are the store's extents (X=8,Y=6 fixed). The
// encoded value is axis-order- AND rank-invariant, so one body covers the default
// TCZYX, the non-default CTZYX, the lower-rank 3D/2D stores, and the no-axes
// (legacy fallback) store — all of which must return the same encoded values.
void test_omezarr_addressing(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = omezarr_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());
    ASSERT_EQ(ldr.fullWidth(0), (size_t)X);
    ASSERT_EQ(ldr.fullHeight(0), (size_t)Y);
    ASSERT_EQ(ldr.fullDepth(0), (size_t)Z);
    const size_t tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(ldr.tileHeight(0) * tw, 0u);

    for (int t = 0; t < T; ++t)
      for (int c = 0; c < C; ++c)
        for (int z = 0; z < Z; ++z)
        {
            std::fill(tile->begin(), tile->end(), 0u);
            ASSERT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, z, c, t, 0));
            const std::vector<uint32_t>& buf = *tile;
            for (int y = 0; y < Y; ++y)
              for (int x = 0; x < X; ++x)
                ASSERT_EQ(buf[y * tw + x], dim5_enc(x, y, z, c, t))
                    << store << " plane (z" << z << " c" << c << " t" << t << ") at (" << x << "," << y << ")";
        }
}

// RawFormatLoader stack (same coverage as above).
void test_raw_omezarr_addressing(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = omezarr_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = RawOmezarrLoader(ds.string());
    ASSERT_EQ(ldr.fullDepth(0), (size_t)Z);
    const size_t tw = ldr.tileWidth(0);

    for (int t = 0; t < T; ++t)
      for (int c = 0; c < C; ++c)
        for (int z = 0; z < Z; ++z)
        {
            ASSERT_NO_THROW(ldr.loadTileFromFile(0, 0, z, c, t, 0));
            for (int y = 0; y < Y; ++y)
              for (int x = 0; x < X; ++x)
                ASSERT_EQ(ldr.get_uint32_pixel(y * tw + x), dim5_enc(x, y, z, c, t))
                    << store << " plane (z" << z << " c" << c << " t" << t << ") at (" << x << "," << y << ")";
        }

    if (C > 1 && T > 1)   // distinct (c,t) planes must differ -> the offset really uses C/T
    {
        ldr.loadTileFromFile(0, 0, 0, 0, 0, 0); uint32_t p000 = ldr.get_uint32_pixel(0);
        ldr.loadTileFromFile(0, 0, 0, 1, 0, 0); uint32_t p010 = ldr.get_uint32_pixel(0);
        ldr.loadTileFromFile(0, 0, 0, 0, 1, 0); uint32_t p001 = ldr.get_uint32_pixel(0);
        ASSERT_NE(p000, p010) << "channel index ignored";
        ASSERT_NE(p000, p001) << "timeframe index ignored";
    }
}

// Whole-volume assembly through the ImageLoader facade: load_volume() must stack
// all Z-planes (per (channel,timeframe)) into one X*Y*Z buffer. This is the
// foundation that lets the volumetric pipeline consume plane-by-plane OME-Zarr.
void test_omezarr_facade_volume(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = omezarr_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";               // whole-slide: intensity only
    FpImageOptions fp;
    ImageLoader il;
    ASSERT_TRUE(il.open(p, fp)) << ds.string();
    ASSERT_EQ(il.get_full_width(), (size_t)X);
    ASSERT_EQ(il.get_full_height(), (size_t)Y);
    ASSERT_EQ(il.get_full_depth(), (size_t)Z);

    for (int t = 0; t < T; ++t)
      for (int c = 0; c < C; ++c)
      {
          ASSERT_TRUE(il.load_volume(c, t));
          const std::vector<uint32_t>& vol = il.get_int_volume_buffer();
          ASSERT_EQ(vol.size(), (size_t)X * Y * Z);
          for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
              for (int x = 0; x < X; ++x)
                ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], dim5_enc(x, y, z, c, t))
                    << store << " vol (x" << x << " y" << y << " z" << z << " c" << c << " t" << t << ")";
      }
    il.close();
}

// Every one of the 6 legal orderings of {t,c,z} before y,x must read correctly
// (proves the axis-role resolution, not just the default TCZYX).
void test_omezarr_all_5d_permutations()
{
    for (const char* s : { "dim5.ome.zarr", "dim5_tzcyx.ome.zarr", "dim5_ctzyx.ome.zarr",
                           "dim5_cztyx.ome.zarr", "dim5_ztcyx.ome.zarr", "dim5_zctyx.ome.zarr" })
    {
        test_omezarr_addressing(s, 2, 3, 4);
        test_raw_omezarr_addressing(s, 2, 3, 4);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// Negative: requesting a Z/C/T plane beyond the array extent must throw, not read
// out-of-bounds / wrong data. dim5.ome.zarr has T=2, C=3, Z=4.
void test_omezarr_out_of_range_throws()
{
    fs::path ds = omezarr_data_path("dim5.ome.zarr");
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());
    auto tile = std::make_shared<std::vector<uint32_t>>(ldr.tileHeight(0) * ldr.tileWidth(0), 0u);
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 99, 0, 0));   // channel out of range
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0, 99, 0));   // timeframe out of range
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 99, 0, 0, 0));   // z out of range

    auto raw = RawOmezarrLoader(ds.string());
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 0, 99, 0, 0));
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 0, 0, 99, 0));
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 99, 0, 0, 0));
}

// Illegal / adversarial: self-inconsistent metadata must be rejected cleanly
// (throw), not crash. bad_axes_count declares 5 axes for a 3D array (indexing the
// shape by axis role would read OOB); bad_no_xy has axes but none labeled x/y.
void test_omezarr_malformed_throws()
{
    for (const char* s : { "bad_axes_count.ome.zarr", "bad_no_xy.ome.zarr" })
    {
        fs::path ds = omezarr_data_path(s);
        ASSERT_TRUE(fs::exists(ds)) << ds.string();
        EXPECT_ANY_THROW(NyxusOmeZarrLoader<uint32_t>(1, ds.string())) << s;
        EXPECT_ANY_THROW(RawOmezarrLoader(ds.string())) << s;
    }
    // a store path that does not exist at all
    EXPECT_ANY_THROW(RawOmezarrLoader(omezarr_data_path("does_not_exist.ome.zarr").string()));
}

#endif // OMEZARR_SUPPORT
