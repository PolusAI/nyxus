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
#include "../src/nyx/image_loader.h"
#include "../src/nyx/globals.h"           // scan_trivial_wholevolume, LR
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

// End-to-end through the WIRED volumetric consumer: scan_trivial_wholevolume must
// feed the whole X*Y*Z volume (all Z planes) into the ROI's voxel cloud, with the
// correct encoded intensity. Before the wiring it read only plane z=0, so this
// asserts the consumer fix, not just the facade.
void test_ometiff_wholevolume_consumer(const char* store, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = ometiff_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
    FpImageOptions fp;
    ImageLoader ilo;
    ASSERT_TRUE(ilo.open(p, fp)) << ds.string();

    LR vroi;
    ASSERT_TRUE(Nyxus::scan_trivial_wholevolume(vroi, ds.string(), ilo, 0/*channel*/, 0/*timeframe*/));

    // every voxel of the X*Y*Z volume must be captured (not just plane z=0 -> 48)
    ASSERT_EQ(vroi.raw_pixels_3D.size(), (size_t)X * Y * Z);
    for (const Pixel3& px : vroi.raw_pixels_3D)
        ASSERT_EQ((uint32_t)px.inten, ometiff_enc((int)px.x, (int)px.y, (int)px.z, 0, 0))
            << "voxel (" << px.x << "," << px.y << "," << px.z << ")";
    ilo.close();
}

// End-to-end through the wired volumetric consumer for EVERY (channel,timeframe):
// scan_trivial_wholevolume(vroi, .., c, t) must fill the voxel cloud with THAT c/t
// plane's encoded intensity (before the wiring it always read c=0,t=0).
void test_ometiff_wholevolume_consumer_ct(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = ometiff_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
    FpImageOptions fp;
    ImageLoader ilo;
    ASSERT_TRUE(ilo.open(p, fp)) << ds.string();

    for (int t = 0; t < T; ++t)
      for (int c = 0; c < C; ++c)
      {
          LR vroi;
          ASSERT_TRUE(Nyxus::scan_trivial_wholevolume(vroi, ds.string(), ilo, c, t)) << store;
          ASSERT_EQ(vroi.raw_pixels_3D.size(), (size_t)X * Y * Z) << store;
          for (const Pixel3& px : vroi.raw_pixels_3D)
              ASSERT_EQ((uint32_t)px.inten, ometiff_enc((int)px.x, (int)px.y, (int)px.z, c, t))
                  << store << " (c" << c << " t" << t << ") voxel (" << px.x << "," << px.y << "," << px.z << ")";
      }
    ilo.close();
}

// Whole-volume assembly through the ImageLoader facade: load_volume() stacks all
// Z-planes (per (channel,timeframe)) into one X*Y*Z buffer -- the foundation that
// lets the volumetric pipeline consume a multi-page OME-TIFF.
void test_ometiff_facade_volume(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = ometiff_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
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
                ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], ometiff_enc(x, y, z, c, t))
                    << store << " vol (x" << x << " y" << y << " z" << z << " c" << c << " t" << t << ")";
      }
    il.close();
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

// TILED multi-plane OME-TIFF: the TILE loaders (not just the strip loaders) must map
// (z,c,t) -> IFD. dim5_tiled.ome.tif is 5D TCZYX with one 16x16 tile per 6x8 plane.
void test_ometiff_tiled_addressing()
{
    const int T = 2, C = 3, Z = 4, Y = 6, X = 8;
    fs::path ds = ometiff_data_path("dim5_tiled.ome.tif");
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    // AbstractTileLoader stack
    auto ldr = NyxusGrayscaleTiffTileLoader<uint32_t>(1, ds.string(), true, 0.0f, 1.0f, 1e4f, false);
    ASSERT_EQ(ldr.fullWidth(0), (size_t)X);
    ASSERT_EQ(ldr.fullHeight(0), (size_t)Y);
    ASSERT_EQ(ldr.fullDepth(0), (size_t)Z);        // SizeZ, not the 24 IFDs
    ASSERT_EQ(ldr.numberChannels(), (size_t)C);
    ASSERT_EQ(ldr.fullTimestamps(0), (size_t)T);
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
                    << "tile plane (z" << z << " c" << c << " t" << t << ") at (" << x << "," << y << ")";
        }
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 99, 0, 0));   // channel out of range
    EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, 99, 0, 0, 0));   // z out of range

    // RawFormatLoader stack
    auto raw = RawTiffTileLoader(ds.string());
    ASSERT_EQ(raw.fullDepth(0), (size_t)Z);
    ASSERT_EQ(raw.numberChannels(), (size_t)C);
    ASSERT_EQ(raw.fullTimestamps(0), (size_t)T);
    const size_t rw = raw.tileWidth(0);
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < C; ++c)
        for (int z = 0; z < Z; ++z)
        {
            ASSERT_NO_THROW(raw.loadTileFromFile(0, 0, z, c, t, 0));
            for (int y = 0; y < Y; ++y)
              for (int x = 0; x < X; ++x)
                ASSERT_EQ(raw.get_uint32_pixel(y * rw + x), ometiff_enc(x, y, z, c, t))
                    << "raw tile plane (z" << z << " c" << c << " t" << t << ") at (" << x << "," << y << ")";
            raw.free_tile();
        }
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 0, 0, 99, 0));         // timeframe out of range
    EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, 99, 0, 0, 0));         // z out of range
}

// Regression: a single-channel mask paired with a multi-channel intensity. The mask is
// channel-agnostic and must be REUSED for every intensity channel (not indexed at the
// intensity's channel, which would read it out of range and drop the ROI for c>0).
// End-to-end this bug produced only the c=0 rows; here we assert at the facade.
void test_ometiff_multichannel_mask_pairing()
{
    fs::path ipath = ometiff_data_path("dim5.ome.tif");        // intensity C=3, T=2, Z=4
    fs::path mpath = ometiff_data_path("dim3_mask.ome.tif");   // single-channel ZYX label mask
    ASSERT_TRUE(fs::exists(ipath)) << ipath.string();
    ASSERT_TRUE(fs::exists(mpath)) << mpath.string();

    SlideProps p;
    p.fname_int = ipath.string();
    p.fname_seg = mpath.string();
    FpImageOptions fp;
    ImageLoader il;
    ASSERT_TRUE(il.open(p, fp)) << ipath.string();

    ASSERT_TRUE(il.load_volume(0, 0));
    const std::vector<uint32_t> seg0 = il.get_seg_volume_buffer();
    const std::vector<uint32_t> int0 = il.get_int_volume_buffer();
    size_t nz = 0; for (auto v : seg0) if (v) ++nz;
    ASSERT_GT(nz, 0u) << "mask has no ROI voxels";

    for (int c = 0; c < 3; ++c)
    {
        ASSERT_TRUE(il.load_volume(c, 0)) << "channel " << c;                       // must not throw/fail for c>0
        ASSERT_EQ(il.get_seg_volume_buffer(), seg0) << "mask changed for channel " << c;   // reused identically
        if (c > 0)
            ASSERT_NE(il.get_int_volume_buffer(), int0) << "intensity channel " << c << " read c=0 data";
    }
    il.close();
}

// Phase 5 negative: the whole-volume facade read must propagate an out-of-range channel
// or timeframe as a throw (the (z,c,t)->IFD map range-guards). dim5 has C=3, T=2.
void test_ometiff_load_volume_out_of_range()
{
    fs::path ds = ometiff_data_path("dim5.ome.tif");
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
    FpImageOptions fp;
    ImageLoader il;
    ASSERT_TRUE(il.open(p, fp)) << ds.string();

    EXPECT_ANY_THROW(il.load_volume(99, 0));   // channel out of range (C=3)
    EXPECT_ANY_THROW(il.load_volume(0, 99));   // timeframe out of range (T=2)
    // in-range still works
    EXPECT_TRUE(il.load_volume(2, 1));
    il.close();
}

// The strip loaders must ADVERTISE the OME C/T extents via numberChannels() /
// fullTimestamps() -- what the volumetric pipeline keys off to iterate channels
// and timeframes. A plain (non-OME) multi-page TIFF has no OME-XML, so it must keep
// the single-plane default of 1 for both (its pages are all Z-slices).
void test_ometiff_ct_counts(const char* store, int T, int C, int Z)
{
    fs::path ds = ometiff_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = NyxusGrayscaleTiffStripLoader<uint32_t>(1, ds.string());
    ASSERT_EQ(ldr.numberChannels(), (size_t)C) << store;
    ASSERT_EQ(ldr.fullTimestamps(0), (size_t)T) << store;
    ASSERT_EQ(ldr.fullDepth(0), (size_t)Z) << store;

    auto raw = RawTiffStripLoader(1, ds.string());
    ASSERT_EQ(raw.numberChannels(), (size_t)C) << store;
    ASSERT_EQ(raw.fullTimestamps(0), (size_t)T) << store;
    ASSERT_EQ(raw.fullDepth(0), (size_t)Z) << store;
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
