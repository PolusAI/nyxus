#include <gtest/gtest.h>
#include "../src/nyx/raw_nifti.h"
#include "../src/nyx/helpers/fsystem.h"

void test_nifti_loader () 
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

void test_nifti_data_access_consistency()
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
    ASSERT_NO_THROW (ldr1.loadTileFromFile (t, 0, 0, 0, 0));

    // stats
    std::vector<uint32_t>& databuf = *t;
    double tot = 0;
    for (auto x : databuf)
        tot += x;
    ASSERT_TRUE(tot == 544286216);
}
