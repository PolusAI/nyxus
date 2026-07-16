#include <gtest/gtest.h>
#include "../src/nyx/raw_nifti.h"
#include "../src/nyx/image_loader.h"
#include "../src/nyx/helpers/fsystem.h"

void test_3d_nifti_loader () 
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

void test_3d_nifti_data_access_consistency()
{
    fs::path p(__FILE__);
    fs::path pp = p.parent_path();
    fs::path f("/data/nifti/phantoms/2torus.nii");
    fs::path data1_p = (pp.string() + f.make_preferred().string());
    ASSERT_TRUE(fs::exists(data1_p));

    auto ldr1 = NiftiLoader<uint32_t>(data1_p.string());

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
}

// Safety net for wiring the volumetric consumer onto load_volume(): for a NIfTI
// (whole-4D loader), load_volume(0,t) must equal the t-th timeframe slab of the
// current whole-volume read (load_tile(0,0) + get_int_tile_buffer). If this holds,
// swapping the consumer's read source cannot regress NIfTI.
void test_facade_nifti_load_volume_equivalence()
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
    const std::vector<uint32_t> whole = il.get_int_tile_buffer();  // copy (load_volume reuses ptrI)
    ASSERT_GE(whole.size(), timeFrameSize * nt);

    for (size_t t = 0; t < nt; ++t)
    {
        ASSERT_TRUE(il.load_volume(0, t));
        const std::vector<uint32_t>& vol = il.get_int_volume_buffer();
        ASSERT_EQ(vol.size(), timeFrameSize);
        for (size_t i = 0; i < timeFrameSize; ++i)
            ASSERT_EQ(vol[i], whole[t * timeFrameSize + i]) << "t=" << t << " i=" << i;
    }
    il.close();
}
