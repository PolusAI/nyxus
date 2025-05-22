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
    ASSERT_NO_THROW (auto d = ldr1.fullDepth(0), d==20);
    ASSERT_NO_THROW (auto w = ldr1.fullWidth(0), w==512);
    ASSERT_NO_THROW (auto f = ldr1.fullHeight(0), h==512);

    // compressed
    auto ldr2 = NiftiLoader<uint32_t>(data2_p.string());
    ASSERT_NO_THROW(auto d = ldr2.fullDepth(0), d == 20);
    ASSERT_NO_THROW(auto w = ldr2.fullWidth(0), w == 512);
    ASSERT_NO_THROW(auto f = ldr2.fullHeight(0), h == 512);
}