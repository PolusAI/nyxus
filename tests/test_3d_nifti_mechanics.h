#include <gtest/gtest.h>
#include <set>
#include "../src/nyx/raw_nifti.h"
#include "../src/nyx/image_loader.h"
#include "../src/nyx/helpers/fsystem.h"

void test_3d_nifti_loader_mechanics () 
{
    fs::path p(__FILE__);
    fs::path pp = p.parent_path();

    fs::path f1("/data/nifti/signal/signal1.nii");
    fs::path f2("/data/nifti/signal/signal2.nii.gz");

    fs::path data1_p = (pp.string() + f1.make_preferred().string());
    fs::path data2_p = (pp.string() + f2.make_preferred().string());

    ASSERT_TRUE (fs::exists(data1_p));
    ASSERT_TRUE (fs::exists(data2_p));

    // uncompressed
    auto ldr1 = NiftiLoader<uint32_t> (data1_p.string());
    size_t h=0, w=0, d=0;
    ASSERT_NO_THROW(d = ldr1.fullDepth(0));
    ASSERT_NO_THROW(w = ldr1.fullWidth(0));
    ASSERT_NO_THROW(h = ldr1.fullHeight(0));
    ASSERT_TRUE(d == 20);
    ASSERT_TRUE(w == 512);
    ASSERT_TRUE(h == 512);

    // compressed
    if (nifti_is_gzfile(data2_p.string().c_str()))  // bypass .gz checks if zlib option is disabled
    {
        auto ldr2 = NiftiLoader<uint32_t>(data2_p.string());
        h = w = d = 0;
        ASSERT_NO_THROW(d = ldr2.fullDepth(0));
        ASSERT_NO_THROW(w = ldr2.fullWidth(0));
        ASSERT_NO_THROW(h = ldr2.fullHeight(0));
        ASSERT_TRUE(d == 20);
        ASSERT_TRUE(w == 512);
        ASSERT_TRUE(h == 512);
    }
}

void test_3d_nifti_data_access_consistency_mechanics()
{
    fs::path p(__FILE__);
    fs::path pp = p.parent_path();
    fs::path f("/data/nifti/phantoms/2torus.nii");
    fs::path data1_p = (pp.string() + f.make_preferred().string());
    ASSERT_TRUE(fs::exists(data1_p));

    // The volume spans -1024..2000, so the offset the scan records for it -- and hands the
    // loader -- is -1024. The offset is always the scan's: one a loader discovered by
    // itself would be one nothing downstream could undo (SlideProps::inten_offset).
    auto ldr1 = NiftiLoader<uint32_t>(data1_p.string(), -1024.0);

    size_t h = 0, w = 0, d = 0;
    ASSERT_NO_THROW(d = ldr1.fullDepth(0));
    ASSERT_NO_THROW(w = ldr1.fullWidth(0));
    ASSERT_NO_THROW(h = ldr1.fullHeight(0));

    auto t = std::make_shared<std::vector<uint32_t>> (d*w*h);
    ASSERT_NO_THROW (ldr1.loadTileFromFile (t, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));

    // stats
    std::vector<uint32_t>& databuf = *t;
    double tot = 0;
    for (auto x : databuf)
        tot += x;
    ASSERT_TRUE(tot == 544286216);

    // Without that offset the same volume clamps its sub-zero voxels instead of shifting them,
    // which is what keeps a stray negative from wrapping when no slide props are available.
    auto ldr0 = NiftiLoader<uint32_t>(data1_p.string());
    auto t0 = std::make_shared<std::vector<uint32_t>> (d*w*h);
    ASSERT_NO_THROW (ldr0.loadTileFromFile (t0, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    double tot0 = 0;
    for (auto x : *t0)
        tot0 += x;
    ASSERT_TRUE(tot0 < tot);
    ASSERT_TRUE(tot0 > 0);
}

// Safety net for the volumetric consumers' streamed read: for a NIfTI
// (whole-4D loader), the streamed volume of (0,t) must equal the t-th timeframe slab of the
// current whole-volume read (load_tile(0,0) + get_int_tile_buffer). If this holds,
// swapping the consumer's read source cannot regress NIfTI.
void test_3d_nifti_facade_stream_equivalence_mechanics()
{
    fs::path p(__FILE__);
    fs::path f("/data/nifti/signal/signal1.nii");
    fs::path ds = (p.parent_path().string() + f.make_preferred().string());
    ASSERT_TRUE(fs::exists(ds)) << ds.string();

    SlideProps sp;
    sp.fname_int = ds.string();
    sp.fname_seg = "";
    FpImageOptions fp;
    ImageLoader il;
    ASSERT_TRUE(il.open(sp, fp)) << ds.string();

    const size_t timeFrameSize = il.get_full_width() * il.get_full_height() * il.get_full_depth();
    const size_t nt = il.get_inten_time();
    ASSERT_GE(nt, 1u);

    ASSERT_TRUE(il.load_tile((size_t)0, (size_t)0));       // the current NIfTI read: whole x*y*z*t blob
    const std::vector<uint32_t> whole = il.get_int_tile_buffer();  // copy (the stream reuses ptrI)
    ASSERT_GE(whole.size(), timeFrameSize * nt);

    for (size_t t = 0; t < nt; ++t)
    {
        std::vector<uint32_t> vol_i1, vol_s1;
        ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, t, vol_i1, vol_s1));
        const std::vector<uint32_t>& vol = vol_i1;
        ASSERT_EQ(vol.size(), timeFrameSize);
        for (size_t i = 0; i < timeFrameSize; ++i)
            ASSERT_EQ(vol[i], whole[t * timeFrameSize + i]) << "t=" << t << " i=" << i;
    }
    il.close();
}

// ---------------------------------------------------------------------------
// Real-valued NIfTI volumes (FLOAT32 / FLOAT64) and the load-time map.
//
// A volume holding fractional values takes the quantization a real-valued TIFF or OME-Zarr takes:
// the header rescale first, then a hard clamp to [fpmin, fpmax] spread over the target dynamic
// range, with the inverse recorded so the intensity families report the volume's own values. A
// float volume of whole values keeps the offset map instead, 1 grey level per unit. Every grey level
// reads back to within one quantization step, (fpmax - fpmin) / DR, below the value stored. The
// fixtures are 8x8x8 with mask3d.nii (all ones) as the ROI; see tests/data/hounsfield/README.md.
//
//   ct3d_unit.nii        : float32, no rescale, v(idx) = idx/511 over [0, 1]
//   ct3d_float_slope.nii : float32, scl_slope 2, scl_inter -1024, stored idx + 0.25, so the
//                          physical value is 2*idx - 1023.5 over [-1023.5, -1.5]
//
// Narrowed to whole units instead, ct3d_unit.nii reads back as 0 everywhere but its last voxel,
// and every voxel of ct3d_float_slope.nii lands half a unit off.
// ---------------------------------------------------------------------------

static fs::path nifti_hu_path (const char* name)
{
    fs::path p(__FILE__);
    return p.parent_path() / "data" / "hounsfield" / name;
}

// Scans one fixture volume against mask3d.nii, then reads it on the map the scan recorded.
static std::vector<uint32_t> nifti_scan_and_load (SlideProps & p, const FpImageOptions & fpo)
{
    AnisotropyOptions aniso;
    EXPECT_TRUE(Nyxus::scan_slide_props (p, 3, aniso, false, fpo, false));
    ImageLoader ilo;
    EXPECT_TRUE(ilo.open (p, fpo));
    EXPECT_TRUE(ilo.load_tile ((size_t)0, (size_t)0));
    std::vector<uint32_t> buf = ilo.get_int_tile_buffer();
    ilo.close();
    return buf;
}

// The raw loader is what tells the scan a volume is real-valued, and a FLOAT32/FLOAT64 volume says
// so only when a finite voxel's header-rescaled value carries a fraction. One of whole values keeps
// 1 grey level == 1 intensity unit, and an integer datatype never says so, whatever its rescale.
void test_3d_nifti_float_reports_real_valued_voxels_mechanics()
{
    EXPECT_TRUE(RawNiftiLoader (nifti_hu_path("ct3d_unit.nii").string()).get_fp_pixels());		// float32 over [0, 1]
    EXPECT_TRUE(RawNiftiLoader (nifti_hu_path("ct3d_float_slope.nii").string()).get_fp_pixels());	// float32, x.5 after the rescale

    // float32 of whole values, NaN and both infinities among them
    EXPECT_FALSE(RawNiftiLoader (nifti_hu_path("ct3d_nan.nii").string()).get_fp_pixels());

    fs::path p(__FILE__);
    fs::path f64 = p.parent_path() / "data" / "nifti" / "signal" / "signal1.nii";
    ASSERT_TRUE(fs::exists(f64)) << f64.string();
    EXPECT_FALSE(RawNiftiLoader (f64.string()).get_fp_pixels());		// float64 of whole values

    EXPECT_FALSE(RawNiftiLoader (nifti_hu_path("ct3d_frac.nii").string()).get_fp_pixels());	// int16, scl_slope 0.5
    EXPECT_FALSE(RawNiftiLoader (nifti_hu_path("mask3d.nii").string()).get_fp_pixels());	// uint8
}

// A volume normalised to [0, 1] keeps its fraction: 512 distinct values become 512 distinct grey
// levels over [0, 10000], and each reads back to within 1e-4 of the value stored.
void test_3d_nifti_float_unit_range_quantized_mechanics()
{
    SlideProps p (nifti_hu_path("ct3d_unit.nii").string(), nifti_hu_path("mask3d.nii").string());
    FpImageOptions fpo;
    std::vector<uint32_t> buf = nifti_scan_and_load (p, fpo);
    ASSERT_GE(buf.size(), 512u);

    EXPECT_TRUE(p.fp_phys_pivoxels);
    ASSERT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0 / 10000.0);

    const double step = p.inten_scale;
    for (size_t idx = 0; idx < 512; idx++)
    {
        double v = (double)(float)((double)idx / 511.0);	// the float32 the file stores
        EXPECT_NEAR(p.to_source_intensity (buf[idx]), v, step) << "idx " << idx;
    }
    EXPECT_EQ(buf[0], 0u);
    EXPECT_EQ(buf[511], 10000u);
    EXPECT_EQ(std::set<uint32_t>(buf.begin(), buf.begin() + 512).size(), 512u);

    // the forward map lands on the grey level the loader stored
    EXPECT_EQ(p.to_grey_level (p.max_preroi_inten), buf[511]);
}

// The header rescale applies before the quantization, so the window is in physical units: the
// scan measures [-1023.5, -1.5] on the rescaled values, and the loader has to rescale each voxel
// the same way before clamping it into that window.
void test_3d_nifti_float_slope_quantized_mechanics()
{
    SlideProps p (nifti_hu_path("ct3d_float_slope.nii").string(), nifti_hu_path("mask3d.nii").string());
    FpImageOptions fpo;
    std::vector<uint32_t> buf = nifti_scan_and_load (p, fpo);
    ASSERT_GE(buf.size(), 512u);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_DOUBLE_EQ(p.inten_offset, -1023.5);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1022.0 / 10000.0);

    const double step = p.inten_scale;
    for (size_t idx = 0; idx < 512; idx++)
    {
        double physical = 2.0 * (double) idx - 1023.5;
        EXPECT_NEAR(p.to_source_intensity (buf[idx]), physical, step) << "idx " << idx;
    }
    EXPECT_EQ(buf[0], 0u);
    EXPECT_EQ(buf[511], 10000u);
}

// --fpimgmin / --fpimgmax / --fpimgdr (min_intensity / max_intensity / dynamic_range in Python)
// name the window instead of the scanned range. [0, 2] over 100 grey levels puts the volume's
// [0, 1] on levels 0..50, each 0.02 wide.
void test_3d_nifti_float_fpimage_window_mechanics()
{
    SlideProps p (nifti_hu_path("ct3d_unit.nii").string(), nifti_hu_path("mask3d.nii").string());
    FpImageOptions fpo;
    fpo.raw_min_intensity = "0";
    fpo.raw_max_intensity = "2";
    fpo.raw_target_dyn_range = "100";
    ASSERT_TRUE(fpo.parse_input());
    std::vector<uint32_t> buf = nifti_scan_and_load (p, fpo);
    ASSERT_GE(buf.size(), 512u);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);
    EXPECT_DOUBLE_EQ(p.inten_scale, 2.0 / 100.0);

    EXPECT_EQ(buf[0], 0u);
    EXPECT_EQ(buf[255], 24u);		// 100 * (255/511) / 2 = 24.95
    EXPECT_EQ(buf[511], 50u);
    for (size_t idx = 0; idx < 512; idx++)
    {
        double v = (double)(float)((double)idx / 511.0);
        EXPECT_NEAR(p.to_source_intensity (buf[idx]), v, 0.02) << "idx " << idx;
    }
}

// The readings of a NIfTI that must not take the quantized map. Under --preserve-hu a volume with
// fractional values takes the offset map, rounding to whole units; a float volume of whole values
// takes the offset map and reads back exactly; an integer volume with a header rescale takes the
// stored map; and a mask volume, read with the rescale off and no quantization, keeps its voxels
// as labels even when they are stored as floats.
void test_3d_nifti_float_not_quantized_mechanics()
{
    {
        SlideProps p (nifti_hu_path("ct3d_nan.nii").string(), nifti_hu_path("mask3d.nii").string());
        FpImageOptions fpo;
        std::vector<uint32_t> buf = nifti_scan_and_load (p, fpo);
        ASSERT_GE(buf.size(), 512u);
        EXPECT_FALSE(p.fp_phys_pivoxels);
        ASSERT_EQ((int)p.inten_map, (int)IntenMap::offset);
        EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
        EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);
        EXPECT_EQ(buf[100], 100u);		// v(idx) = idx, stored as it is
        EXPECT_EQ(buf[511], 511u);
        EXPECT_EQ(buf[3], 0u);			// NaN
    }
    {
        SlideProps p (nifti_hu_path("ct3d_unit.nii").string(), nifti_hu_path("mask3d.nii").string());
        p.preserve_hu = true;
        FpImageOptions fpo;
        std::vector<uint32_t> buf = nifti_scan_and_load (p, fpo);
        ASSERT_GE(buf.size(), 512u);
        ASSERT_EQ((int)p.inten_map, (int)IntenMap::offset);
        EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
        EXPECT_EQ(buf[255], 0u);		// 0.499 rounds down
        EXPECT_EQ(buf[256], 1u);		// 0.501 rounds up
        EXPECT_EQ(buf[511], 1u);
    }
    {
        SlideProps p (nifti_hu_path("ct3d_frac.nii").string(), nifti_hu_path("mask3d.nii").string());
        FpImageOptions fpo;
        std::vector<uint32_t> buf = nifti_scan_and_load (p, fpo);
        ASSERT_GE(buf.size(), 512u);
        EXPECT_FALSE(p.fp_phys_pivoxels);
        ASSERT_EQ((int)p.inten_map, (int)IntenMap::stored);
        EXPECT_EQ(buf[511], 511u);		// the stored integer itself
    }
    {
        NiftiLoader<uint32_t> ldr (nifti_hu_path("ct3d_unit.nii").string(), 0.0, false);	// as ImageLoader opens a mask
        auto t = std::make_shared<std::vector<uint32_t>> (512);
        ASSERT_NO_THROW(ldr.loadTileFromFile (t, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
        EXPECT_EQ((*t)[510], 0u);
        EXPECT_EQ((*t)[511], 1u);
    }
}
