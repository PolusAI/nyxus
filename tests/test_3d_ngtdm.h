#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_ngtdm.h"

// dig. phantom values for intensity based features
// Calculated at grey scalefactlr 100
static std::unordered_map<std::string, float> d3ngtdm_GT {
    {"3NGTDM_COARSENESS",   0.00004},
    {"3NGTDM_CONTRAST",     0.66},
    {"3NGTDM_BUSYNESS",     46.0},
    {"3NGTDM_COMPLEXITY",   2936.0},
    {"3NGTDM_STRENGTH",     0.024}
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3ngtdm_feature(const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
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
    D3_NGTDM_feature f;
    ASSERT_NO_THROW(f.calculate(r));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3ngtdm_GT[fname], 10.));
}

void test_3ngtdm_coarseness()
{
    test_3ngtdm_feature(Nyxus::Feature3D::NGTDM_COARSENESS, "3NGTDM_COARSENESS");
}

void test_3ngtdm_contrast()
{
    test_3ngtdm_feature(Nyxus::Feature3D::NGTDM_CONTRAST, "3NGTDM_CONTRAST");
}

void test_3ngtdm_busyness()
{
    test_3ngtdm_feature(Nyxus::Feature3D::NGTDM_BUSYNESS, "3NGTDM_BUSYNESS");
}

void test_3ngtdm_complexity()
{
    test_3ngtdm_feature(Nyxus::Feature3D::NGTDM_COMPLEXITY, "3NGTDM_COMPLEXITY");
}

void test_3ngtdm_strength()
{
    test_3ngtdm_feature(Nyxus::Feature3D::NGTDM_STRENGTH, "3NGTDM_STRENGTH");
}

