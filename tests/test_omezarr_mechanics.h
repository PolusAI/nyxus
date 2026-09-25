#pragma once

// The 5D (T,C,Z,Y,X) channel/timeframe surface of the OME-Zarr readers. The 2D tile/raw
// loader assertions live in test_2d_omezarr_mechanics.h, which this includes for the shared
// OME-Zarr headers and omezarr_data_path(); both files are the omezarr family.
#include "test_2d_omezarr_mechanics.h"
#include <fstream>

#ifdef OMEZARR_SUPPORT

// ---------------------------------------------------------------------------
// 5D (T,C,Z,Y,X) channel/timeframe addressability.
//
// dim5.ome.zarr (see gen_dim5.py) encodes every voxel as
//   value(x,y,z,c,t) = 1 + ((((t*C + c)*Z + z)*Y + y)*X + x),  C=3,Z=4,Y=6,X=8
// chunked one z/c/t-plane per chunk. Reading plane (z,c,t) must return exactly
// that plane's values. What this discriminates: a loader that ignores C/T -- pinning the
// offset to {0,0,...} -- returns the c=0/t=0 plane for every (c,t).
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
void assert_omezarr_addressing(const char* store, int T, int C, int Z)
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
void assert_raw_omezarr_addressing(const char* store, int T, int C, int Z)
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

// End-to-end through the WIRED volumetric consumer: scan_trivial_wholevolume must
// feed the whole X*Y*Z volume (all Z planes) into the ROI's voxel cloud with the
// correct encoded intensity -- before the wiring it read only plane z=0.
void assert_omezarr_wholevolume_consumer(const char* store, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = omezarr_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
    FpImageOptions fp;
    ImageLoader ilo;
    ASSERT_TRUE(ilo.open(p, fp)) << ds.string();

    LR vroi;
    ASSERT_TRUE(Nyxus::scan_trivial_wholevolume(vroi, ds.string(), ilo, 0/*channel*/, 0/*timeframe*/));

    ASSERT_EQ(vroi.raw_pixels_3D.size(), (size_t)X * Y * Z);   // all voxels, not just plane 0 (48)
    for (const Pixel3& px : vroi.raw_pixels_3D)
        ASSERT_EQ((uint32_t)px.inten, dim5_enc((int)px.x, (int)px.y, (int)px.z, 0, 0))
            << "voxel (" << px.x << "," << px.y << "," << px.z << ")";
    ilo.close();
}

// End-to-end through the wired volumetric consumer, but for EVERY (channel,timeframe):
// scan_trivial_wholevolume(vroi, .., c, t) must fill the ROI's voxel cloud with the
// encoded intensity of THAT c/t plane. Before the channel/timeframe wiring the consumer
// always read (c=0,t=0), so any c>0 / t>0 plane would carry the wrong values.
void assert_omezarr_wholevolume_consumer_ct(const char* store, int T, int C, int Z)
{
    const int Y = 6, X = 8;
    fs::path ds = omezarr_data_path(store);
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
              ASSERT_EQ((uint32_t)px.inten, dim5_enc((int)px.x, (int)px.y, (int)px.z, c, t))
                  << store << " (c" << c << " t" << t << ") voxel (" << px.x << "," << px.y << "," << px.z << ")";
      }
    ilo.close();
}

// Whole-volume assembly through the ImageLoader facade: the streamed planes must stack
// all Z-planes (per (channel,timeframe)) into one X*Y*Z buffer. This is the
// foundation that lets the volumetric pipeline consume plane-by-plane OME-Zarr.
void assert_omezarr_facade_volume(const char* store, int T, int C, int Z)
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
          std::vector<uint32_t> vol_i1, vol_s1;
          ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, c, t, vol_i1, vol_s1));
          const std::vector<uint32_t>& vol = vol_i1;
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
void test_omezarr_all_5d_permutations_mechanics()
{
    for (const char* s : { "dim5.ome.zarr", "dim5_tzcyx.ome.zarr", "dim5_ctzyx.ome.zarr",
                           "dim5_cztyx.ome.zarr", "dim5_ztcyx.ome.zarr", "dim5_zctyx.ome.zarr" })
    {
        assert_omezarr_addressing(s, 2, 3, 4);
        assert_raw_omezarr_addressing(s, 2, 3, 4);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// Negative: requesting a Z/C/T plane beyond the array extent must throw, not read
// out-of-bounds / wrong data. dim5.ome.zarr has T=2, C=3, Z=4.
void test_omezarr_out_of_range_throws_mechanics()
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

// The loaders must ADVERTISE the real C/T extents via numberChannels() /
// fullTimestamps(). This is what the volumetric pipeline keys off to iterate
// channels and timeframes; before this both fell back to the base-class default
// of 1, which pinned the pipeline to plane (c=0, t=0) regardless of the store.
// Covers the parsed-axes path and (via dim5_noaxes) the positional fallback.
void assert_omezarr_ct_counts(const char* store, int T, int C, int Z)
{
    fs::path ds = omezarr_data_path(store);
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());
    ASSERT_EQ(ldr.numberChannels(), (size_t)C) << store;
    ASSERT_EQ(ldr.fullTimestamps(0), (size_t)T) << store;
    ASSERT_EQ(ldr.fullDepth(0), (size_t)Z) << store;

    auto raw = RawOmezarrLoader(ds.string());
    ASSERT_EQ(raw.numberChannels(), (size_t)C) << store;
    ASSERT_EQ(raw.fullTimestamps(0), (size_t)T) << store;
    ASSERT_EQ(raw.fullDepth(0), (size_t)Z) << store;
}

// Negative: the whole-volume facade read must propagate an out-of-range channel
// or timeframe as a throw (not silently read plane 0 or OOB memory). dim5 has C=3, T=2.
void test_omezarr_stream_volume_out_of_range_mechanics()
{
    fs::path ds = omezarr_data_path("dim5.ome.zarr");
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
    std::vector<uint32_t> vol_i2, vol_s2;
    EXPECT_NO_THROW(Nyxus::assemble_streamed_volume(il, 2, 1, vol_i2, vol_s2));
    il.close();
}

// Physical calibration: the loaders must surface the OME-Zarr coordinateTransformations
// 'scale' as physicalSizeX/Y/Z and the space-axis 'unit'. dim5_calibrated encodes
// scale (t,c,z,y,x)=(1,1,2,0.5,0.5) micrometer -> physX=physY=0.5, physZ=2.0. An
// uncalibrated store (scale all 1) must report 1.0 and no unit.
void test_omezarr_physical_calibration_mechanics()
{
    fs::path cal = omezarr_data_path("dim5_calibrated.ome.zarr");
    ASSERT_TRUE(fs::exists(cal)) << cal.string();

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, cal.string());
    ASSERT_DOUBLE_EQ(ldr.physicalSizeX(), 0.5);
    ASSERT_DOUBLE_EQ(ldr.physicalSizeY(), 0.5);
    ASSERT_DOUBLE_EQ(ldr.physicalSizeZ(), 2.0);
    ASSERT_EQ(ldr.physicalSizeUnit(), "micrometer");

    auto raw = RawOmezarrLoader(cal.string());
    ASSERT_DOUBLE_EQ(raw.physicalSizeX(), 0.5);
    ASSERT_DOUBLE_EQ(raw.physicalSizeY(), 0.5);
    ASSERT_DOUBLE_EQ(raw.physicalSizeZ(), 2.0);
    ASSERT_EQ(raw.physicalSizeUnit(), "micrometer");

    // uncalibrated store: scale all 1.0 -> physical sizes default to 1.0
    fs::path plain = omezarr_data_path("dim5.ome.zarr");
    auto ldr2 = NyxusOmeZarrLoader<uint32_t>(1, plain.string());
    ASSERT_DOUBLE_EQ(ldr2.physicalSizeX(), 1.0);
    ASSERT_DOUBLE_EQ(ldr2.physicalSizeZ(), 1.0);
}

// Unit canonicalization: dim5_calibrated_nm declares the SAME physical spacing as
// dim5_calibrated above, but in nanometer (2000/500/500 nm == 2.0/0.5/0.5 um). The loader
// must report the SAME canonicalized values and unit as the micrometer fixture -- proving
// actual conversion happens, not just passthrough of whatever unit string the file declares.
void test_omezarr_unit_canonicalization_mechanics()
{
    fs::path cal_nm = omezarr_data_path("dim5_calibrated_nm.ome.zarr");
    ASSERT_TRUE(fs::exists(cal_nm)) << cal_nm.string();

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, cal_nm.string());
    ASSERT_DOUBLE_EQ(ldr.physicalSizeX(), 0.5);
    ASSERT_DOUBLE_EQ(ldr.physicalSizeY(), 0.5);
    ASSERT_DOUBLE_EQ(ldr.physicalSizeZ(), 2.0);
    ASSERT_EQ(ldr.physicalSizeUnit(), "micrometer");

    auto raw = RawOmezarrLoader(cal_nm.string());
    ASSERT_DOUBLE_EQ(raw.physicalSizeX(), 0.5);
    ASSERT_DOUBLE_EQ(raw.physicalSizeY(), 0.5);
    ASSERT_DOUBLE_EQ(raw.physicalSizeZ(), 2.0);
    ASSERT_EQ(raw.physicalSizeUnit(), "micrometer");
}

// Illegal / adversarial: self-inconsistent metadata must be rejected cleanly
// (throw), not crash. bad_axes_count declares 5 axes for a 3D array (indexing the
// shape by axis role would read OOB); bad_no_xy has axes but none labeled x/y.
void test_omezarr_malformed_throws_mechanics()
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

    // A Zarr v2 group that names no level-0 dataset: no multiscales, an empty one, or one without
    // datasets. Written at run time, since nothing past the group attributes is read. What this
    // discriminates: looking the dataset path up without checking it exists is undefined behavior
    // on the const JSON, not an exception.
    int k = 0;
    for (const char* zattrs : { "{}", "{\"multiscales\": []}", "{\"multiscales\": [{\"axes\": []}]}", "{\"ome\": {\"multiscales\": [{}]}}" })
    {
        fs::path g = fs::temp_directory_path() / ("nyxus_zarr_no_dataset_" + std::to_string(k++) + ".ome.zarr");
        std::error_code ec;
        fs::remove_all(g, ec);
        fs::create_directories(g);
        std::ofstream(g / ".zgroup") << "{\"zarr_format\": 2}";
        std::ofstream(g / ".zattrs") << zattrs;
        EXPECT_ANY_THROW(NyxusOmeZarrLoader<uint32_t>(1, g.string())) << zattrs;
        EXPECT_ANY_THROW(RawOmezarrLoader(g.string())) << zattrs;
        fs::remove_all(g, ec);
    }
}

// dim3_chunked.ome.zarr: the 4x6x8 ZYX volume of dim3_zyx (values 1..192) chunked (3,4,5), so
// its chunk grid is uneven along every axis -- Z-chunk depths 3,1, row-chunks 4,2, col-chunks 5,3.

// The facade's volume must hold every plane of every chunk at its own place. What this
// discriminates: a reader that fills only the first Z-plane of a chunk leaves planes 1 and 2
// zero, and an assembly that reads only chunk (0,0) of a plane leaves the rest of it wrong.
void test_omezarr_chunked_facade_volume_mechanics()
{
    ASSERT_NO_THROW (assert_omezarr_facade_volume("dim3_chunked.ome.zarr", 1, 1, 4));

    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, omezarr_data_path("dim3_chunked.ome.zarr").string());
    EXPECT_EQ(ldr.tileDepth(0), 3u);
    EXPECT_EQ(ldr.tileHeight(0), 4u);
    EXPECT_EQ(ldr.tileWidth(0), 5u);
}

// The 3D prescan (RawImageLoader::for_each_voxel over the raw Zarr loader) must cover the whole
// volume: a prescan that misses any chunk, or any plane of one, reports a maximum below 192 or
// an ROI smaller than the volume.
void test_omezarr_chunked_prescan_mechanics()
{
    fs::path ip = omezarr_data_path("dim3_chunked.ome.zarr");
    ASSERT_TRUE(fs::exists(ip)) << ip.string();

    Environment e;
    SlideProps p (ip.string(), "");		// whole-slide: no mask
    ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

    EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
    EXPECT_DOUBLE_EQ(p.max_preroi_inten, (double)dim5_enc(7, 5, 3, 0, 0));
    EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// Under --use-physical-spacing the prescan records the ROI geometry of the RESAMPLED grid, the
// one every scan of the volume builds: dim5_calibrated declares z=2, y=x=0.5 um, so the smallest
// axis normalizes to 1 and Z is resampled 4x. What this discriminates: a prescan that applies
// only explicit --aniso* reports the physical depth, and the memory estimate built from it
// budgets a quarter of the voxels the scans will cache.
void test_omezarr_physical_spacing_prescan_mechanics()
{
    fs::path ip = omezarr_data_path("dim5_calibrated.ome.zarr");
    ASSERT_TRUE(fs::exists(ip)) << ip.string();

    Environment e;
    SlideProps phys (ip.string(), "");
    ASSERT_TRUE(Nyxus::scan_slide_props(phys, 3, e.anisoOptions, true, e.fpimageOptions, e.resultOptions.need_annotation()));
    EXPECT_DOUBLE_EQ(phys.phys_z, 2.0);
    EXPECT_DOUBLE_EQ(phys.phys_x, 0.5);
    EXPECT_EQ(phys.max_roi_w, 8u);
    EXPECT_EQ(phys.max_roi_h, 6u);
    // the resampled extent AABB::apply_anisotropy gives the Z range 0..3 at 4x, which phase 1
    // records for the same volume (the scans then recompute the exact extent from the cloud)
    EXPECT_EQ(phys.max_roi_d, 14u);

    SlideProps off (ip.string(), "");
    ASSERT_TRUE(Nyxus::scan_slide_props(off, 3, e.anisoOptions, false, e.fpimageOptions, e.resultOptions.need_annotation()));
    EXPECT_EQ(off.max_roi_d, 4u);
}

// An OME-Zarr store with more than one channel or timepoint is refused by the 2D prescan rather
// than featurized at C=0, T=0 alone; the 3D prescan accepts it, since 3D featurizes every
// channel and timepoint. The single-channel, single-timepoint dim3_chunked passes in both (see
// test_omezarr_chunked_prescan_mechanics).
void test_omezarr_multichannel_timepoint_refused_mechanics()
{
    Environment e;
    for (const char* s : { "dim5.ome.zarr", "dim4_czyx.ome.zarr", "dim4_tzyx.ome.zarr", "dim5_noaxes.ome.zarr" })
      for (int dim : { 2, 3 })
      {
          fs::path ip = omezarr_data_path(s);
          ASSERT_TRUE(fs::exists(ip)) << ip.string();
          SlideProps p (ip.string(), "");
          EXPECT_EQ(Nyxus::scan_slide_props(p, dim, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()), dim == 3)
              << s << " dim=" << dim;
      }
}

// A store whose Z chunk spans the whole volume: one read of it is every Z-plane of a
// frame, so an out-of-core pass has nothing smaller to stream and must decline it rather
// than allocate the volume it exists to avoid. What this discriminates: dim5_zwholechunk
// carries T=2 and reports it through fullTimestamps, while its tile holds one frame and
// leaves tileTimestamps at the base class's 1 -- a boundedness rule that reads those two
// extents as one scale factor passes the store as streamable and hands the out-of-core
// path the whole cube.
void test_omezarr_whole_z_chunk_is_unstreamable_mechanics()
{
    fs::path ip = omezarr_data_path("dim5_zwholechunk.ome.zarr");
    ASSERT_TRUE(fs::exists(ip)) << ip.string();

    // the shape the discrimination rests on
    {
        auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ip.string());
        ASSERT_EQ(ldr.fullDepth(0), 4u);
        ASSERT_EQ(ldr.tileDepth(0), 4u) << "the fixture's Z chunk must span the whole volume";
        ASSERT_EQ(ldr.fullTimestamps(0), 2u) << "and it must carry more than one time frame";
        ASSERT_EQ(ldr.tileTimestamps(0), 1u) << "the loader leaves the base-class default in place";
    }

    FpImageOptions fp;
    SlideProps p;
    p.fname_int = ip.string();
    p.fname_seg = "";
    ImageLoader il;
    ASSERT_TRUE(il.open(p, fp)) << ip.string();
    EXPECT_FALSE(il.streams_bounded())
        << "one read of this store delivers all 4 Z-planes of the frame, which is the whole volume";

    // the refusal reports the planes that read delivers -- the frame's depth, NOT the depth
    // times the store's two time frames, which a read of this loader does not carry
    size_t planes = 0;
    bool of_mask = true;
    ASSERT_TRUE(il.unstreamable_read (planes, of_mask));
    EXPECT_EQ(planes, 4u) << "a read of this store is one frame's 4 Z-planes";
    EXPECT_FALSE(of_mask);
    il.close();

    // the same store family chunked one Z-plane per chunk still streams, so the refusal above is
    // the chunking and not the format
    fs::path bounded = omezarr_data_path("dim5.ome.zarr");
    ASSERT_TRUE(fs::exists(bounded)) << bounded.string();
    SlideProps q;
    q.fname_int = bounded.string();
    q.fname_seg = "";
    ImageLoader il2;
    ASSERT_TRUE(il2.open(q, fp)) << bounded.string();
    EXPECT_TRUE(il2.streams_bounded()) << "a one-plane Z chunk is a bounded read at any T";
    il2.close();
}

// Nested v2 chunk keys: dim3_nested.ome.zarr is dim3_zyx's volume, values and chunking with
// `dimension_separator: "/"` and a blosc codec, so chunk (1,0,0) is the file 0/1/0/0 rather
// than 0/1.0.0. Nesting is what bioformats2raw writes by default -- 0.4 mandates it, and
// asking for flat keys downgrades the store's declared NGFF version to 0.1 -- yet every other
// 3D/5D fixture here is flat-separator, so the layout of real converter output went untested.
// What this discriminates: a reader that composes chunk keys itself instead of letting the
// store's separator do it finds no chunk where it looks and returns fill_value, i.e. zeros,
// for the whole volume.
void test_omezarr_nested_chunk_keys_mechanics()
{
    assert_omezarr_addressing("dim3_nested.ome.zarr", 1, 1, 4);
    if (::testing::Test::HasFatalFailure()) return;
    assert_raw_omezarr_addressing("dim3_nested.ome.zarr", 1, 1, 4);
    if (::testing::Test::HasFatalFailure()) return;
    ASSERT_NO_THROW (assert_omezarr_facade_volume("dim3_nested.ome.zarr", 1, 1, 4));
}

// The two refusals a store from the common converter runs into must say what is wrong, not
// only that something is. Both stores below hold perfectly good data.
//
//  * bigendian.ome.zarr declares `dtype: ">u2"`. z5's zarrToDtype() map holds only the '<' and
//    '|' spellings, so openDataset throws "Unsupported zarr dtype: >u2" -- accurate, and no
//    help at all: it names neither big-endianness nor the way out. bioformats2raw through
//    0.9.x writes big-endian with no switch to change it, so every 16-bit store that converter
//    produced lands here.
//  * b2r_layout.ome.zarr (and its v3 twin) is in bioformats2raw layout: the root carries only
//    {"bioformats2raw.layout": 3} and the image is the child group `0`. The path is one level
//    too high; nothing is damaged.
//
// What this discriminates: the pre-fix code answered the first with z5's raw dtype complaint
// and the second with "the group declares no multiscales" and no path, so both read as a
// corrupt file. The assertions are on the message, because the throw itself already happened
// before the fix.
void test_omezarr_diagnosed_refusals_mechanics()
{
    struct Case { const char* store; const char* phrase; const char* why; };
    const Case cases[] = {
        { "bigendian.ome.zarr",     "big-endian",       "names the byte order, not just the dtype code" },
        { "b2r_layout.ome.zarr",    "bioformats2raw",   "names the layout it recognized" },
        { "b2r_layout_v3.ome.zarr", "bioformats2raw",   "and does so with the v3 'ome'-nested attributes too" },
    };

    for (const Case& c : cases)
    {
        fs::path ds = omezarr_data_path(c.store);
        ASSERT_TRUE(fs::exists(ds)) << ds.string();

        // both loader stacks resolve the layout through the same open_zarr_level0
        for (int raw = 0; raw < 2; ++raw)
        {
            try
            {
                if (raw) RawOmezarrLoader dead (ds.string());
                else     NyxusOmeZarrLoader<uint32_t> dead (1, ds.string());
                FAIL() << c.store << ": expected a throw (raw=" << raw << ")";
            }
            catch (const std::runtime_error& e)
            {
                const std::string msg = e.what();
                EXPECT_NE(msg.find (c.phrase), std::string::npos)
                    << c.store << " (raw=" << raw << "): " << c.why << ": " << msg;
                EXPECT_NE(msg.find (ds.string()), std::string::npos)
                    << c.store << " (raw=" << raw << "): the message names the store: " << msg;
            }
        }
    }

    // The positive counterpart of the layout case: the SAME store read one level down, at the
    // series group the refusal points at, holds the dim3_zyx volume and reads correctly. This
    // is what makes the refusal a path problem rather than a data problem.
    const std::string series = (omezarr_data_path("b2r_layout.ome.zarr") / "0").string();
    ASSERT_TRUE(fs::exists(series)) << series;
    EXPECT_NO_THROW(NyxusOmeZarrLoader<uint32_t>(1, series));
    auto ldr = NyxusOmeZarrLoader<uint32_t>(1, series);
    EXPECT_EQ(ldr.fullWidth(0), 8u);
    EXPECT_EQ(ldr.fullHeight(0), 6u);
    EXPECT_EQ(ldr.fullDepth(0), 4u);
}


#endif // OMEZARR_SUPPORT
