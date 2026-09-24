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
#include <limits>
#include <memory>
#include <vector>
#include "../src/nyx/grayscale_tiff.h"
#include "../src/nyx/raw_tiff.h"
#include "../src/nyx/image_loader.h"
#include "../src/nyx/globals.h"           // scan_trivial_wholevolume, LR, Environment, scan_slide_props
#include "../src/nyx/helpers/fsystem.h"
#include "test_main_nyxus.h"	// ometiff_data_path

static inline uint32_t ometiff_enc(int x, int y, int z, int c, int t)
{
    const int C = 3, Z = 4, Y = 6, X = 8;
    return static_cast<uint32_t>(1 + ((((t * C + c) * Z + z) * Y + y) * X + x));
}

// AbstractTileLoader stack (NyxusGrayscaleTiffStripLoader). (T,C,Z) are the store's
// extents (X=8,Y=6 fixed). One body covers TCZYX, non-default CTZYX, 3D, 2D, and
// the plain non-OME multi-page store -- all must return the same encoded values.
void assert_ometiff_addressing(const char* store, int T, int C, int Z)
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
void assert_raw_ometiff_addressing(const char* store, int T, int C, int Z)
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
void assert_ometiff_wholevolume_consumer(const char* store, int Z)
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
void assert_ometiff_wholevolume_consumer_ct(const char* store, int T, int C, int Z)
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

// Whole-volume assembly through the ImageLoader facade: the streamed planes stack all
// Z-planes (per (channel,timeframe)) into one X*Y*Z buffer -- the foundation that
// lets the volumetric pipeline consume a multi-page OME-TIFF.
void assert_ometiff_facade_volume(const char* store, int T, int C, int Z)
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
          std::vector<uint32_t> vol_i1, vol_s1;
          ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, c, t, vol_i1, vol_s1));
          const std::vector<uint32_t>& vol = vol_i1;
          ASSERT_EQ(vol.size(), (size_t)X * Y * Z);
          for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
              for (int x = 0; x < X; ++x)
                ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], ometiff_enc(x, y, z, c, t))
                    << store << " vol (x" << x << " y" << y << " z" << z << " c" << c << " t" << t << ")";
      }
    il.close();
}

// The multitile fixtures carry their own (larger Y/X) shape because a TIFF tile size must be
// a multiple of 16: only a plane bigger than 16x16 yields a real tile grid. Their coordinate
// encoding is encoded_tczyx with each fixture's own C/Z/Y/X (see gen_ome_tiff.write_multitile).
inline uint32_t ometiff_enc_dims(int x, int y, int z, int c, int t, int C, int Z, int Y, int X)
{
    return (uint32_t)(1 + ((((t * C + c) * Z + z) * Y + y) * X + x));
}

// Multi-TILE plane: each plane spans a grid of 16x16 tiles, unlike every other fixture (one
// tile/chunk per plane), so assembling a volume from tile (0,0) alone silently returns wrong
// data outside the first 16x16 corner -- and over-reads that tile's buffer. With Y or X not a
// multiple of 16 the last tile of the row/column is PARTIAL, exercising the validH/validW seam
// clamp. Asserting the exact value at every voxel IS the seam check.
void assert_ometiff_multitile_facade_volume(const char* store, int T, int C, int Z, int Y, int X)
{
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
          std::vector<uint32_t> vol_i2, vol_s2;
          ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, c, t, vol_i2, vol_s2));
          const std::vector<uint32_t>& vol = vol_i2;
          ASSERT_EQ(vol.size(), (size_t)X * Y * Z);
          for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
              for (int x = 0; x < X; ++x)
                ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], ometiff_enc_dims(x, y, z, c, t, C, Z, Y, X))
                    << store << " vol (x" << x << " y" << y << " z" << z << " c" << c << " t" << t << ")";
      }
    il.close();
}

// Every one of the 6 legal OME DimensionOrder values (all orderings of {T,C,Z}
// before Y,X) must read correctly -- proves ifdForPlane honors DimensionOrder.
void test_ometiff_all_5d_permutations_mechanics()
{
    for (const char* s : { "dim5.ome.tif", "dim5_tzcyx.ome.tif", "dim5_ctzyx.ome.tif",
                           "dim5_cztyx.ome.tif", "dim5_ztcyx.ome.tif", "dim5_zctyx.ome.tif" })
    {
        assert_ometiff_addressing(s, 2, 3, 4);
        assert_raw_ometiff_addressing(s, 2, 3, 4);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// TILED multi-plane OME-TIFF: the TILE loaders (not just the strip loaders) must map
// (z,c,t) -> IFD. dim5_tiled.ome.tif is 5D TCZYX with one 16x16 tile per 6x8 plane.
void test_ometiff_tiled_addressing_mechanics()
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
void test_ometiff_multichannel_mask_pairing_mechanics()
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

    std::vector<uint32_t> vol_i3, vol_s3;
    ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, 0, vol_i3, vol_s3));
    const std::vector<uint32_t> seg0 = vol_s3;
    const std::vector<uint32_t> int0 = vol_i3;
    size_t nz = 0; for (auto v : seg0) if (v) ++nz;
    ASSERT_GT(nz, 0u) << "mask has no ROI voxels";

    for (int c = 0; c < 3; ++c)
    {
        std::vector<uint32_t> vol_i4, vol_s4;
        ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, c, 0, vol_i4, vol_s4)) << "channel " << c;                       // must not throw/fail for c>0
        ASSERT_EQ(vol_s4, seg0) << "mask changed for channel " << c;   // reused identically
        if (c > 0)
            ASSERT_NE(vol_i4, int0) << "intensity channel " << c << " read c=0 data";
    }
    il.close();
}

// Negative: the whole-volume facade read must propagate an out-of-range channel
// or timeframe as a throw (the (z,c,t)->IFD map range-guards). dim5 has C=3, T=2.
void test_ometiff_stream_volume_out_of_range_mechanics()
{
    fs::path ds = ometiff_data_path("dim5.ome.tif");
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
    FpImageOptions fp;
    ImageLoader il;
    ASSERT_TRUE(il.open(p, fp)) << ds.string();

    EXPECT_ANY_THROW(il.stream_volume_planes(99, 0, [](size_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&) {}));   // channel out of range (C=3)
    EXPECT_ANY_THROW(il.stream_volume_planes(0, 99, [](size_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&) {}));   // timeframe out of range (T=2)
    // in-range still works
    std::vector<uint32_t> vol_i5, vol_s5;
    EXPECT_NO_THROW(Nyxus::assemble_streamed_volume(il, 2, 1, vol_i5, vol_s5));
    il.close();
}

// The strip loaders must ADVERTISE the OME C/T extents via numberChannels() /
// fullTimestamps() -- what the volumetric pipeline keys off to iterate channels
// and timeframes. A plain (non-OME) multi-page TIFF has no OME-XML, so it must keep
// the single-plane default of 1 for both (its pages are all Z-slices).
void assert_ometiff_ct_counts(const char* store, int T, int C, int Z)
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
void test_ometiff_out_of_range_throws_mechanics()
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

// dim3_oddtile.ome.tif: a single-channel tiled Z-stack (Z=3) whose 40x24 planes span a 3x2
// grid of 16x16 tiles, the last row-tile 8 tall and the last col-tile 8 wide.
// value(x,y,z) = 1 + (z*Y + y)*X + x, so the volume holds 1..2880.
static inline uint32_t oddtile_enc(int x, int y, int z)
{
    const int Y = 40, X = 24;
    return static_cast<uint32_t>(1 + (z * Y + y) * X + x);
}

// The facade's volume must hold every tile of every plane at its own place. What this
// discriminates: an assembly that reads only tile (0,0) leaves the other five tiles of each
// plane wrong, and one that copies whole tile rows past the 8-wide seam over-reads the tile.
void test_ometiff_oddtile_zstack_facade_volume_mechanics()
{
    const int Z = 3, Y = 40, X = 24;
    fs::path ds = ometiff_data_path("dim3_oddtile.ome.tif");
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
    FpImageOptions fp;
    ImageLoader il;
    ASSERT_TRUE(il.open(p, fp)) << ds.string();
    ASSERT_EQ(il.get_full_depth(), (size_t)Z);
    ASSERT_EQ(il.get_num_tiles_vert(), 3u);
    ASSERT_EQ(il.get_num_tiles_hor(), 2u);

    std::vector<uint32_t> vol_i6, vol_s6;
    ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, 0, vol_i6, vol_s6));
    const std::vector<uint32_t>& vol = vol_i6;
    ASSERT_EQ(vol.size(), (size_t)X * Y * Z);
    for (int z = 0; z < Z; ++z)
      for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
          ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], oddtile_enc(x, y, z))
              << "oddtile vol (x" << x << " y" << y << " z" << z << ")";
    il.close();
}

// The 3D prescan (RawImageLoader::for_each_voxel over the raw tile loader) must cover the
// whole volume. What this discriminates: a prescan that reads one tile, or one plane, reports
// a maximum below 2880 and an ROI smaller than the volume.
void test_ometiff_oddtile_zstack_prescan_mechanics()
{
    fs::path ip = ometiff_data_path("dim3_oddtile.ome.tif");
    ASSERT_TRUE(fs::exists(ip)) << ip.string();

    Environment e;
    SlideProps p (ip.string(), "");		// whole-slide: no mask
    ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

    EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
    EXPECT_DOUBLE_EQ(p.max_preroi_inten, 2880.0);
    EXPECT_EQ(p.max_roi_area, (size_t)(24 * 40 * 3));
}

// The strip-loader counterpart: dim3_zyx.ome.tif is a Z=4 stack of 6x8 planes holding 1..192.
// A prescan that reads only plane 0 reports a maximum of 48.
void test_ometiff_zstack_prescan_mechanics()
{
    fs::path ip = ometiff_data_path("dim3_zyx.ome.tif");
    ASSERT_TRUE(fs::exists(ip)) << ip.string();

    Environment e;
    SlideProps p (ip.string(), "");		// whole-slide: no mask
    ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

    EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
    EXPECT_DOUBLE_EQ(p.max_preroi_inten, (double)ometiff_enc(7, 5, 3, 0, 0));
    EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// An OME-TIFF with more than one channel or timepoint is refused by the 2D prescan rather than
// featurized at C=0, T=0 alone; the 3D prescan accepts it, since 3D featurizes every channel
// and timepoint. A single-channel, single-timepoint Z-stack passes in both (see
// test_ometiff_zstack_prescan_mechanics).
void test_ometiff_multichannel_timepoint_refused_mechanics()
{
    Environment e;
    for (const char* s : { "dim5.ome.tif", "dim4_czyx.ome.tif", "dim4_tzyx.ome.tif", "dim5_tiled.ome.tif" })
      for (int dim : { 2, 3 })
      {
          fs::path ip = ometiff_data_path(s);
          ASSERT_TRUE(fs::exists(ip)) << ip.string();
          SlideProps p (ip.string(), "");
          EXPECT_EQ(Nyxus::scan_slide_props(p, dim, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()), dim == 3)
              << s << " dim=" << dim;
      }
}

// The one rule every volumetric consumer pairs frames by: the stream of (c, t) reads the mask at
// frame t when the mask has as many frames as the intensity (N:N), and at frame 0 when one mask
// serves every intensity frame (1:N). dim4_tzyx (T=2, Z=4) is the intensity; as a "mask" its
// values are all non-zero, so the mask volume can be checked voxel by voxel.
void test_ometiff_mask_frame_pairing_mechanics()
{
    const int Z = 4, Y = 6, X = 8;

    struct Case { const char* mask; int mask_frames; };
    for (Case k : { Case{ "dim3_zyx.ome.tif", 1 }, Case{ "dim4_tzyx.ome.tif", 2 } })
    {
        SlideProps p;
        p.fname_int = ometiff_data_path("dim4_tzyx.ome.tif").string();
        p.fname_seg = ometiff_data_path(k.mask).string();
        FpImageOptions fp;
        ImageLoader il;
        ASSERT_TRUE(il.open(p, fp)) << k.mask;
        ASSERT_EQ(il.get_inten_time(), 2u);
        ASSERT_EQ(il.get_mask_time(), (size_t)k.mask_frames);

        for (int t = 0; t < 2; ++t)
        {
            std::vector<uint32_t> vol_i7, vol_s7;
            ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, t, vol_i7, vol_s7)) << k.mask << " t=" << t;
            const std::vector<uint32_t>& vi = vol_i7;
            const std::vector<uint32_t>& vm = vol_s7;
            const int mask_t = (k.mask_frames == 2) ? t : 0;
            for (int z = 0; z < Z; ++z)
              for (int y = 0; y < Y; ++y)
                for (int x = 0; x < X; ++x)
                {
                    const size_t i = (size_t)z * X * Y + (size_t)y * X + x;
                    ASSERT_EQ(vi[i], ometiff_enc(x, y, z, 0, t)) << k.mask << " intensity t=" << t;
                    ASSERT_EQ(vm[i], ometiff_enc(x, y, z, 0, mask_t)) << k.mask << " mask for intensity t=" << t;
                }
        }
        il.close();
    }
}

// A multi-page TIFF with more directories than a 16-bit index can name. tdir_t is 32 bits wide
// since libtiff 4.5, and both strip loaders must read page 65536 itself. What this
// discriminates: a loader that narrows the page number to 16 bits reads page 0 (value 1) in its
// place. The file is written at run time (65537 1x1 uint32 pages, value = page + 1) into the
// temp directory, since it would be ~14 MB in the repository.
void test_ometiff_directory_beyond_16bit_mechanics()
{
    if ((std::numeric_limits<tdir_t>::max)() <= 65535u)
        GTEST_SKIP() << "this libtiff addresses at most 65536 directories";

    const uint32_t n_pages = 65537;
    fs::path path = fs::temp_directory_path() / "nyxus_tiff_65537_pages.tif";
    {
        TIFF* t = TIFFOpen(path.string().c_str(), "w");
        ASSERT_NE(t, nullptr) << path.string();
        for (uint32_t page = 0; page < n_pages; ++page)
        {
            TIFFSetField(t, TIFFTAG_IMAGEWIDTH, 1u);
            TIFFSetField(t, TIFFTAG_IMAGELENGTH, 1u);
            TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 32);
            TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1u);
            uint32_t v = page + 1;
            ASSERT_EQ(TIFFWriteScanline(t, &v, 0, 0), 1) << "page " << page;
            ASSERT_EQ(TIFFWriteDirectory(t), 1) << "page " << page;
        }
        TIFFClose(t);
    }

    {
        auto ldr = NyxusGrayscaleTiffStripLoader<uint32_t>(1, path.string());
        ASSERT_EQ(ldr.fullDepth(0), (size_t)n_pages);
        auto tile = std::make_shared<std::vector<uint32_t>>(ldr.tileHeight(0) * ldr.tileWidth(0), 0u);
        for (uint32_t page : { 0u, 65535u, 65536u })
        {
            ASSERT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, page, 0, 0, 0));
            EXPECT_EQ((*tile)[0], page + 1) << "page " << page;
        }
        EXPECT_ANY_THROW(ldr.loadTileFromFile(tile, 0, 0, n_pages, 0, 0, 0));
    }
    {
        auto raw = RawTiffStripLoader(1, path.string());
        ASSERT_EQ(raw.fullDepth(0), (size_t)n_pages);
        for (uint32_t page : { 0u, 65535u, 65536u })
        {
            ASSERT_NO_THROW(raw.loadTileFromFile(0, 0, page, 0, 0, 0));
            EXPECT_EQ(raw.get_uint32_pixel(0), page + 1) << "page " << page;
            raw.free_tile();
        }
        EXPECT_ANY_THROW(raw.loadTileFromFile(0, 0, n_pages, 0, 0, 0));
    }

    std::error_code ec;
    fs::remove(path, ec);
}

// Segmented 3D featurizes every (channel, timeframe), so the prescan measures every plane.
// dim5 (C=3, T=2) is the intensity and dim3_zyx the mask, whose 192 non-zero voxels are 192
// one-voxel ROIs shared by every plane. What this discriminates: a prescan that measures only
// plane (0,0) reports a maximum of 192 where plane (2,1) reaches 1152, and one that adds up an
// ROI's voxels over the planes reports an area of 6 for a one-voxel ROI.
// A mask with a frame per timepoint (N:N) may label different ROIs in each frame: dim4_tzyx as its
// own mask carries labels 1..192 in frame 0 and 577..768 in frame 1, so a prescan that takes ROI
// geometry from the first frame alone finds half of the 384 ROIs. 2D still refuses a time series.
void test_ometiff_segmented_timeseries_prescan_mechanics()
{
    Environment e;
    const std::string mask = ometiff_data_path("dim3_zyx.ome.tif").string();

    SlideProps p (ometiff_data_path("dim5.ome.tif").string(), mask);
    ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
    EXPECT_EQ(p.inten_channels, 3u);
    EXPECT_EQ(p.inten_time, 2u);
    EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
    EXPECT_DOUBLE_EQ(p.max_preroi_inten, (double)ometiff_enc(7, 5, 3, 2, 1));
    EXPECT_EQ(p.n_rois, (size_t)(8 * 6 * 4));
    EXPECT_EQ(p.max_roi_area, 1u);

    const std::string tzyx = ometiff_data_path("dim4_tzyx.ome.tif").string();
    SlideProps pn (tzyx, tzyx);
    ASSERT_TRUE(Nyxus::scan_slide_props(pn, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
    EXPECT_EQ(pn.n_rois, (size_t)(2 * 8 * 6 * 4));
    EXPECT_EQ(pn.max_roi_area, 1u);
    EXPECT_DOUBLE_EQ(pn.max_preroi_inten, (double)ometiff_enc(7, 5, 3, 0, 1));

    SlideProps p2 (tzyx, mask);
    EXPECT_FALSE(Nyxus::scan_slide_props(p2, 2, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
}
// open() starts a pair at plane (0,0): a plane stream_volume_planes() selected on an earlier pair must not
// carry over. What this discriminates: load_tile() on the single-timepoint dim3_zyx reading at
// the timeframe 1 left behind by dim4_tzyx is out of range, and throws.
void test_ometiff_reopen_resets_plane_mechanics()
{
    FpImageOptions fp;
    ImageLoader il;

    SlideProps p4 (ometiff_data_path("dim4_tzyx.ome.tif").string(), "");
    ASSERT_TRUE(il.open(p4, fp));
    std::vector<uint32_t> vol_i8, vol_s8;
    ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, 1, vol_i8, vol_s8));
    il.close();

    SlideProps p3 (ometiff_data_path("dim3_zyx.ome.tif").string(), "");
    ASSERT_TRUE(il.open(p3, fp));
    ASSERT_NO_THROW(il.load_tile(0, 0));
    EXPECT_EQ(il.get_int_tile_buffer()[0], ometiff_enc(0, 0, 0, 0, 0));
    il.close();
}

// A plain (non-OME) uint16 TIFF written at run time: one directory per page, each page's value
// at (x,y) is plain_tiff_enc(x, y, page). 'tile' > 0 writes tile x tile tiles (zero-padded at
// the edges), 0 writes one-row strips. 'subfile' is the page's NEWSUBFILETYPE.
struct PlainTiffPage { uint32_t w, h, subfile; };

static inline uint32_t plain_tiff_enc(uint32_t x, uint32_t y, uint32_t page)
{
    return 1 + x + 7 * y + 11 * page;
}

static void write_plain_tiff (const fs::path& path, const std::vector<PlainTiffPage>& pages, uint32_t tile)
{
    TIFF* t = TIFFOpen(path.string().c_str(), "w");
    ASSERT_NE(t, nullptr) << path.string();
    for (uint32_t page = 0; page < pages.size(); ++page)
    {
        const PlainTiffPage& pg = pages[page];
        TIFFSetField(t, TIFFTAG_SUBFILETYPE, pg.subfile);
        TIFFSetField(t, TIFFTAG_IMAGEWIDTH, pg.w);
        TIFFSetField(t, TIFFTAG_IMAGELENGTH, pg.h);
        TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 16);
        TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        if (tile > 0)
        {
            TIFFSetField(t, TIFFTAG_TILEWIDTH, tile);
            TIFFSetField(t, TIFFTAG_TILELENGTH, tile);
            std::vector<uint16_t> buf(tile * tile);
            for (uint32_t y0 = 0; y0 < pg.h; y0 += tile)
                for (uint32_t x0 = 0; x0 < pg.w; x0 += tile)
                {
                    for (uint32_t r = 0; r < tile; ++r)
                        for (uint32_t c = 0; c < tile; ++c)
                            buf[r * tile + c] = (y0 + r < pg.h && x0 + c < pg.w) ? (uint16_t)plain_tiff_enc(x0 + c, y0 + r, page) : 0;
                    ASSERT_GE(TIFFWriteTile(t, buf.data(), x0, y0, 0, 0), 0) << "page " << page;
                }
        }
        else
        {
            TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1u);
            std::vector<uint16_t> row(pg.w);
            for (uint32_t y = 0; y < pg.h; ++y)
            {
                for (uint32_t x = 0; x < pg.w; ++x)
                    row[x] = (uint16_t)plain_tiff_enc(x, y, page);
                ASSERT_EQ(TIFFWriteScanline(t, row.data(), y, 0), 1) << "page " << page << " row " << y;
            }
        }
        ASSERT_EQ(TIFFWriteDirectory(t), 1) << "page " << page;
    }
    TIFFClose(t);
}

// A strip TIFF larger than one 1024x1024 tile in both directions, two pages deep. The raw strip
// loader must return the tile it is asked for, so the prescan measures the voxels phases 1 and 2
// load. What this discriminates: a loader that reads rows 0-1023 at the full scanline width
// whatever the tile returns row 0's values for tile (1,1), and the 2D and 3D prescans then
// report a maximum below the image's (the largest value sits in the far corner).
void test_ometiff_plain_strip_beyond_one_tile_mechanics()
{
    const uint32_t W = 1100, H = 1030;
    fs::path path = fs::temp_directory_path() / "nyxus_tiff_strip_1100x1030x2.tif";
    ASSERT_NO_FATAL_FAILURE(write_plain_tiff(path, { { W, H, 0 }, { W, H, 0 } }, 0));

    {
        auto raw = RawTiffStripLoader(1, path.string());
        ASSERT_EQ(raw.fullDepth(0), 2u);
        ASSERT_EQ(raw.tileWidth(0), 1024u);
        ASSERT_EQ(raw.tileHeight(0), 1024u);
        const size_t tw = raw.tileWidth(0);
        for (uint32_t tr : { 0u, 1u })
            for (uint32_t tc : { 0u, 1u })
            {
                ASSERT_NO_THROW(raw.loadTileFromFile(tr, tc, 1, 0, 0, 0));
                const uint32_t y0 = tr * 1024, x0 = tc * 1024;
                for (uint32_t y : { y0, (std::min)(y0 + 1023, H - 1) })
                    for (uint32_t x : { x0, (std::min)(x0 + 1023, W - 1) })
                        ASSERT_EQ(raw.get_uint32_pixel((y - y0) * tw + (x - x0)), plain_tiff_enc(x, y, 1))
                            << "tile (" << tr << "," << tc << ") x=" << x << " y=" << y;
                raw.free_tile();
            }
    }

    Environment e;
    SlideProps p3 (path.string(), "");
    ASSERT_TRUE(Nyxus::scan_slide_props(p3, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
    EXPECT_DOUBLE_EQ(p3.min_preroi_inten, 1.0);
    EXPECT_DOUBLE_EQ(p3.max_preroi_inten, (double)plain_tiff_enc(W - 1, H - 1, 1));
    EXPECT_EQ(p3.max_roi_area, (size_t)W * H * 2);

    SlideProps p2 (path.string(), "");
    ASSERT_TRUE(Nyxus::scan_slide_props(p2, 2, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
    EXPECT_DOUBLE_EQ(p2.min_preroi_inten, 1.0);
    EXPECT_DOUBLE_EQ(p2.max_preroi_inten, (double)plain_tiff_enc(W - 1, H - 1, 0));
    EXPECT_EQ(p2.max_roi_area, (size_t)W * H);

    std::error_code ec;
    fs::remove(path, ec);
}

// A plain TIFF's depth is its run of full-size directories, tiled or not: every TIFF loader counts
// a tiled Z-stack's planes as the strip loaders count a strip one's, and stops at a smaller
// (pyramid) level or at a page flagged reduced-resolution. What this discriminates: a tiled loader
// that takes any plain TIFF as one plane reads a 3-plane stack as depth 1, and a loader that
// counts every directory takes the pyramid level or the reduced page for a Z-plane.
void test_ometiff_plain_depth_mechanics()
{
    const uint32_t W = 40, H = 24;
    const std::vector<PlainTiffPage> pyramid = { { W, H, 0 }, { W, H, 0 }, { W, H, 0 }, { W / 2, H / 2, FILETYPE_REDUCEDIMAGE } },
        reduced = { { W, H, 0 }, { W, H, 0 }, { W, H, FILETYPE_REDUCEDIMAGE } };

    for (uint32_t tile : { 16u, 0u })
    {
        fs::path pp = fs::temp_directory_path() / (std::string("nyxus_tiff_plain_pyramid_") + std::to_string(tile) + ".tif"),
            pr = fs::temp_directory_path() / (std::string("nyxus_tiff_plain_reduced_") + std::to_string(tile) + ".tif");
        ASSERT_NO_FATAL_FAILURE(write_plain_tiff(pp, pyramid, tile));
        ASSERT_NO_FATAL_FAILURE(write_plain_tiff(pr, reduced, tile));

        if (tile > 0)
        {
            EXPECT_EQ(RawTiffTileLoader(pp.string()).fullDepth(0), 3u);
            EXPECT_EQ(RawTiffTileLoader(pr.string()).fullDepth(0), 2u);
        }
        else
        {
            EXPECT_EQ(RawTiffStripLoader(1, pp.string()).fullDepth(0), 3u);
            EXPECT_EQ(RawTiffStripLoader(1, pr.string()).fullDepth(0), 2u);
        }

        // the facade reads every plane of the stack into its volume
        FpImageOptions fp;
        ImageLoader il;
        SlideProps p (pp.string(), "");
        ASSERT_TRUE(il.open(p, fp)) << "tile " << tile;
        ASSERT_EQ(il.get_full_depth(), 3u) << "tile " << tile;
        std::vector<uint32_t> vol_i9, vol_s9;
        ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, 0, vol_i9, vol_s9));
        const std::vector<uint32_t>& vol = vol_i9;
        for (uint32_t z = 0; z < 3; ++z)
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x)
                    ASSERT_EQ(vol[((size_t)z * H + y) * W + x], plain_tiff_enc(x, y, z))
                        << "tile " << tile << " x=" << x << " y=" << y << " z=" << z;
        il.close();

        // and the prescan measures the same volume
        Environment e;
        SlideProps ps (pp.string(), "");
        ASSERT_TRUE(Nyxus::scan_slide_props(ps, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
        EXPECT_DOUBLE_EQ(ps.max_preroi_inten, (double)plain_tiff_enc(W - 1, H - 1, 2)) << "tile " << tile;
        EXPECT_EQ(ps.max_roi_area, (size_t)W * H * 3) << "tile " << tile;

        std::error_code ec;
        fs::remove(pp, ec);
        fs::remove(pr, ec);
    }
}

// IEEE half-precision bits of a value a half represents exactly (a normal number or zero).
static uint16_t ometiff_half_bits (double v)
{
    if (v == 0.0)
        return 0;
    uint16_t sign = v < 0 ? 0x8000 : 0;
    int e;
    double m = std::frexp (std::fabs (v), &e);    // |v| = m * 2^e, m in [0.5, 1)
    return (uint16_t) (sign | ((e + 14) << 10) | ((uint16_t) std::lround ((m * 2.0 - 1.0) * 1024.0) & 0x3ff));
}

// A 12x6 half-float (SampleFormat 3, 16 bits) TIFF whose value at (x,y) is (x - 4) + 0.25*y,
// written in strips and in 16x16 tiles. Both raw loaders must read each sample at its own
// 2-byte width, so the prescan's range is [-4, 8.25]. What this discriminates: a loader that
// reads a half-float sample through a 4-byte float reads pairs of samples as one garbage value
// (and a strip loader doing so reads past its scanline).
void test_ometiff_plain_half_float_mechanics()
{
    const uint32_t W = 12, H = 6;
    auto value = [](uint32_t x, uint32_t y) { return (double) x - 4.0 + 0.25 * y; };

    for (uint32_t tile : { 0u, 16u })
    {
        fs::path path = fs::temp_directory_path() / (std::string("nyxus_tiff_half_") + std::to_string(tile) + ".tif");
        {
            TIFF* t = TIFFOpen(path.string().c_str(), "w");
            ASSERT_NE(t, nullptr) << path.string();
            TIFFSetField(t, TIFFTAG_IMAGEWIDTH, W);
            TIFFSetField(t, TIFFTAG_IMAGELENGTH, H);
            TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
            TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            if (tile > 0)
            {
                TIFFSetField(t, TIFFTAG_TILEWIDTH, tile);
                TIFFSetField(t, TIFFTAG_TILELENGTH, tile);
                std::vector<uint16_t> buf(tile * tile, 0);
                for (uint32_t y = 0; y < H; ++y)
                    for (uint32_t x = 0; x < W; ++x)
                        buf[y * tile + x] = ometiff_half_bits(value(x, y));
                ASSERT_GE(TIFFWriteTile(t, buf.data(), 0, 0, 0, 0), 0);
            }
            else
            {
                TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1u);
                std::vector<uint16_t> row(W);
                for (uint32_t y = 0; y < H; ++y)
                {
                    for (uint32_t x = 0; x < W; ++x)
                        row[x] = ometiff_half_bits(value(x, y));
                    ASSERT_EQ(TIFFWriteScanline(t, row.data(), y, 0), 1);
                }
            }
            TIFFClose(t);
        }

        std::unique_ptr<RawFormatLoader> raw;
        if (tile > 0)
            raw.reset(new RawTiffTileLoader(path.string()));
        else
            raw.reset(new RawTiffStripLoader(1, path.string()));
        EXPECT_TRUE(raw->get_fp_pixels()) << "tile " << tile;
        const size_t tw = raw->tileWidth(0);
        ASSERT_NO_THROW(raw->loadTileFromFile(0, 0, 0, 0, 0, 0));
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x)
                ASSERT_DOUBLE_EQ(raw->get_dpequiv_pixel(y * tw + x), value(x, y)) << "tile " << tile << " x=" << x << " y=" << y;
        raw->free_tile();
        raw.reset();

        Environment e;
        SlideProps p (path.string(), "");
        ASSERT_TRUE(Nyxus::scan_slide_props(p, 2, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
        EXPECT_DOUBLE_EQ(p.min_preroi_inten, -4.0) << "tile " << tile;
        EXPECT_DOUBLE_EQ(p.max_preroi_inten, value(W - 1, H - 1)) << "tile " << tile;

        std::error_code ec;
        fs::remove(path, ec);
    }
}

// A multi-file OME-TIFF (a <TiffData> naming a companion file's UUID) is refused by all four
// TIFF loaders when they are constructed. What this discriminates: a loader that notes the
// layout and reads on maps the companion file's planes onto whatever local directory their
// ordinals land on. The same file whose blocks name its own UUID is read.
void test_ometiff_multi_file_refused_mechanics()
{
    auto write = [](const fs::path& path, const std::string& planeUuid)
    {
        const std::string xml =
            "<?xml version=\"1.0\"?><OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\" UUID=\"urn:uuid:self\">"
            "<Image ID=\"Image:0\"><Pixels ID=\"Pixels:0\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
            "SizeX=\"4\" SizeY=\"3\" SizeZ=\"2\" SizeC=\"1\" SizeT=\"1\">"
            "<TiffData FirstZ=\"0\" IFD=\"0\" PlaneCount=\"1\"><UUID>urn:uuid:self</UUID></TiffData>"
            "<TiffData FirstZ=\"1\" IFD=\"1\" PlaneCount=\"1\"><UUID>" + planeUuid + "</UUID></TiffData>"
            "</Pixels></Image></OME>";
        TIFF* t = TIFFOpen(path.string().c_str(), "w");
        ASSERT_NE(t, nullptr) << path.string();
        for (uint32_t page = 0; page < 2; ++page)
        {
            TIFFSetField(t, TIFFTAG_IMAGEWIDTH, 4u);
            TIFFSetField(t, TIFFTAG_IMAGELENGTH, 3u);
            TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1u);
            if (page == 0)
                TIFFSetField(t, TIFFTAG_IMAGEDESCRIPTION, xml.c_str());
            uint16_t row[4] = { 1, 2, 3, 4 };
            for (uint32_t y = 0; y < 3; ++y)
                ASSERT_EQ(TIFFWriteScanline(t, row, y, 0), 1);
            ASSERT_EQ(TIFFWriteDirectory(t), 1);
        }
        TIFFClose(t);
    };

    fs::path multi = fs::temp_directory_path() / "nyxus_ometiff_multi_file.ome.tif",
        single = fs::temp_directory_path() / "nyxus_ometiff_own_uuid.ome.tif";
    ASSERT_NO_FATAL_FAILURE(write(multi, "urn:uuid:companion"));
    ASSERT_NO_FATAL_FAILURE(write(single, "urn:uuid:self"));

    EXPECT_ANY_THROW(RawTiffStripLoader(1, multi.string()));
    EXPECT_ANY_THROW(NyxusGrayscaleTiffStripLoader<uint32_t>(1, multi.string()));

    EXPECT_EQ(RawTiffStripLoader(1, single.string()).fullDepth(0), 2u);
    EXPECT_EQ(NyxusGrayscaleTiffStripLoader<uint32_t>(1, single.string()).fullDepth(0), 2u);

    std::error_code ec;
    fs::remove(multi, ec);
    fs::remove(single, ec);
}

// Illegal / adversarial: a non-grayscale (RGB) OME-TIFF and a corrupt/non-TIFF file
// must be rejected cleanly (throw), not crash.
void test_ometiff_malformed_throws_mechanics()
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
