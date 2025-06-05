#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_gldm.h"

// Feature values calculated on intensity ut_inten.nii and mask ut_inten.nii, label 57:
// (100 grey levels, offset 1, and asymmetric cooc matrix)
static std::unordered_map<std::string, float> d3gldm_GT
{
    {"3GLDM_SDE", 0.26},
    {"3GLDM_LDE", 34.77},
    {"3GLDM_LGLE", 0.26},
    {"3GLDM_HGLE", 1957.2},
    {"3GLDM_SDLGLE", 0.00014},
    {"3GLDM_SDHGLE", 617.0},
    {"3GLDM_LDLGLE", 0.044},
    {"3GLDM_LDHGLE", 41214.0},
    {"3GLDM_GLN", 6481.0},
    {"3GLDM_DN", 32498.0},
    {"3GLDM_DNN", 0.118},
    {"3GLDM_GLV", 153.1},
    {"3GLDM_DV", 13.6},
    {"3GLDM_DE", 8.4}
};

std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3gldm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    clear_slide_rois();
    ASSERT_TRUE(gatherRoisMetrics_3D(ipath, mpath));
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(batch, ipath, mpath));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch));

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // set feature's state
    Environment::ibsi_compliance = false;

    // extract the feature
    LR& r = Nyxus::roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLDM_feature f;
    ASSERT_NO_THROW(f.calculate(r));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3gldm_GT[fname], 10.));
}

void test_3gldm_sde()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE");
}

void test_3gldm_lde()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_LDE, "3GLDM_LDE");
}

void test_3gldm_lgle()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE");
}

void test_3gldm_hgle()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_HGLE, "3GLDM_HGLE");
}

void test_3gldm_sdlgle()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_SDLGLE, "3GLDM_SDLGLE");
}

void test_3gldm_sdhgle()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_SDHGLE, "3GLDM_SDHGLE");
}

void test_3gldm_ldlgle()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_LDLGLE, "3GLDM_LDLGLE");
}

void test_3gldm_ldhgle()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_LDHGLE, "3GLDM_LDHGLE");
}

void test_3gldm_gln()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_GLN, "3GLDM_GLN");
}

void test_3gldm_dn()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_DN, "3GLDM_DN");
}

void test_3gldm_dnn()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_DNN, "3GLDM_DNN");
}

void test_3gldm_glv()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_GLV, "3GLDM_GLV");
}

void test_3gldm_dv()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_DV, "3GLDM_DV");
}

void test_3gldm_de()
{
    test_3gldm_feature(Nyxus::Feature3D::GLDM_DE, "3GLDM_DE");
}

