#pragma once

// OME-TIFF native read: the multi-page loaders map (z,c,t) to the correct IFD via
// the OME-XML DimensionOrder, instead of assuming every page is a Z-slice.
//
// Fixtures (tests/data/ometiff, see gen_ome_tiff.py) encode every voxel as
//   value(x,y,z,c,t) = 1 + ((((t*C + c)*Z + z)*Y + y)*X + x),  C=3,Z=4,Y=6,X=8
// so reading plane (z,c,t) must return that plane's values; a loader that mapped
// (z,c,t) to the wrong page returns provably wrong data. Unlike OME-Zarr this needs
// no USE_Z5 -- TIFF is a core dependency, so these run in every build.

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "../src/nyx/grayscale_tiff.h"
#include "../src/nyx/raw_tiff.h"
#include "../src/nyx/helpers/fsystem.h"

static inline fs::path ometiff_data_path(const char* name)
{
    fs::path p(__FILE__);
    fs::path rel(std::string("/data/ometiff/") + name);
    return fs::path(p.parent_path().string() + rel.make_preferred().string());
}

static inline uint32_t ometiff_enc(int x, int y, int z, int c, int t)
{
    const int C = 3, Z = 4, Y = 6, X = 8;
    return static_cast<uint32_t>(1 + ((((t * C + c) * Z + z) * Y + y) * X + x));
}

// AbstractTileLoader stack (NyxusGrayscaleTiffStripLoader). (T,C,Z) are the store's
// extents (X=8,Y=6 fixed). One body covers TCZYX, non-default CTZYX, 3D, 2D, and
// the plain non-OME multi-page store -- all must return the same encoded values.
void test_ometiff_addressing(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = ometiff_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = NyxusGrayscaleTiffStripLoader<uint32_t>(1, ds.string());
    ASSERT_EQ(ldr.fullWidth(0), (size_t)X);
    ASSERT_EQ(ldr.fullHeight(0), (size_t)Y);
    ASSERT_EQ(ldr.fullDepth(0), (size_t)Z);   // 5D OME-TIFF must report SizeZ, not the total page count
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
                ASSERT_EQ(buf[y * tw + x], ometiff_enc(x, y, z, c, t))
                    << store << " plane (z" << z << " c" << c << " t" << t << ") at (" << x << "," << y << ")";
        }
}

// RawFormatLoader stack (RawTiffStripLoader).
void test_raw_ometiff_addressing(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = ometiff_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = RawTiffStripLoader(1, ds.string());
    ASSERT_EQ(ldr.fullDepth(0), (size_t)Z);
    const size_t w = ldr.fullWidth(0);

    for (int t = 0; t < T; ++t)
      for (int c = 0; c < C; ++c)
        for (int z = 0; z < Z; ++z)
        {
            ASSERT_NO_THROW(ldr.loadTileFromFile(0, 0, z, c, t, 0));
            for (int y = 0; y < Y; ++y)
              for (int x = 0; x < X; ++x)
                ASSERT_EQ(ldr.get_uint32_pixel(y * w + x), ometiff_enc(x, y, z, c, t))
                    << store << " plane (z" << z << " c" << c << " t" << t << ") at (" << x << "," << y << ")";
        }
    ldr.free_tile();
}

// Every one of the 6 legal OME DimensionOrder values (all orderings of {T,C,Z}
// before Y,X) must read correctly -- proves ifdForPlane honors DimensionOrder.
void test_ometiff_all_5d_permutations()
{
    for (const char* s : { "dim5.ome.tif", "dim5_tzcyx.ome.tif", "dim5_ctzyx.ome.tif",
                           "dim5_cztyx.ome.tif", "dim5_ztcyx.ome.tif", "dim5_zctyx.ome.tif" })
    {
        test_ometiff_addressing(s, 2, 3, 4);
        test_raw_ometiff_addressing(s, 2, 3, 4);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// Negative: an out-of-range Z/C/T plane maps to a non-existent IFD and must throw.
// dim5.ome.tif has T=2, C=3, Z=4 (24 IFDs).
void test_ometiff_out_of_range_throws()
{
    fs::path ds = ometiff_data_path("dim5.ome.tif");
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = NyxusGrayscaleTiffStripLoader<uint32_t>(1, ds.string());
    auto tile = std::make_shared<std::vector<uint32_t>>(ldr.tileHeight(0) * ldr.tileWidth(0), 0u);
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 99, 0, 0));   // channel out of range
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0, 99, 0));   // timeframe out of range
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 99, 0, 0, 0));   // z out of range

    auto raw = RawTiffStripLoader(1, ds.string());
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 0, 99, 0, 0));
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 0, 0, 99, 0));
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 99, 0, 0, 0));
}

// Illegal / adversarial: a non-grayscale (RGB) OME-TIFF and a corrupt/non-TIFF file
// must be rejected cleanly (throw), not crash.
void test_ometiff_malformed_throws()
{
    for (const char* s : { "bad_rgb.ome.tif", "bad_corrupt.tif" })
    {
        fs::path ds = ometiff_data_path(s);
        ASSERT_TRUE(fs::exists(ds)) << ds.string();
        EXPECT_ANY_THROW(NyxusGrayscaleTiffStripLoader<uint32_t>(1, ds.string())) << s;
        EXPECT_ANY_THROW(RawTiffStripLoader(1, ds.string())) << s;
    }
    // a path that does not exist
    EXPECT_ANY_THROW(RawTiffStripLoader(1, ometiff_data_path("does_not_exist.tif").string()));
}
