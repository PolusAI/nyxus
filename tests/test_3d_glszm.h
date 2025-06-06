#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glszm.h"

// dig. phantom values for intensity based features
// Calculated at 100 grey levels
static std::unordered_map<std::string, float> d3glszm_GT{
    {"3GLSZM_SAE",  0.6},
    {"3GLSZM_LAE",  1377.1},
    {"3GLSZM_LGLZE",    0.0005},
    {"3GLSZM_HGLZE",    2485.9},
    {"3GLSZM_SALGLE",   0.0003},
    {"3GLSZM_SAHGLE",   1592.0},
    {"3GLSZM_LALGLE",   1.9},
    {"3GLSZM_LAHGLE",   1.24578e+06},
    {"3GLSZM_GLN",  2037.0},
    {"3GLSZM_GLNN", 0.03},
    {"3GLSZM_SZN",  24582.1},
    {"3GLSZM_SZNN", 0.36},
    {"3GLSZM_ZP",   0.275},
    {"3GLSZM_GLV",  106.5},
    {"3GLSZM_ZV",   1362.3},
    {"3GLSZM_ZE",   7.44}
};

std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3glszm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
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
    D3_GLSZM_feature f;
    ASSERT_NO_THROW(f.calculate(r));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3glszm_GT[fname], 10.));
}

void test_3glszm_sae()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_SAE, "3GLSZM_SAE");
}

void test_3glszm_lae()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_LAE, "3GLSZM_LAE");
}

void test_3glszm_lglze()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_LGLZE, "3GLSZM_LGLZE");
}

void test_3glszm_hglze()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_HGLZE, "3GLSZM_HGLZE");
}

void test_3glszm_salgle()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_SALGLE, "3GLSZM_SALGLE");
}

void test_3glszm_sahgle()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_SAHGLE, "3GLSZM_SAHGLE");
}

void test_3glszm_lalgle()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_LALGLE, "3GLSZM_LALGLE");
}

void test_3glszm_lahgle()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_LAHGLE, "3GLSZM_LAHGLE");
}

void test_3glszm_gln()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_GLN, "3GLSZM_GLN");
}

void test_3glszm_glnn()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_GLNN, "3GLSZM_GLNN");
}

void test_3glszm_szn()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_SZN, "3GLSZM_SZN");
}

void test_3glszm_sznn()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_SZNN, "3GLSZM_SZNN");
}

void test_3glszm_zp()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_ZP, "3GLSZM_ZP");
}

void test_3glszm_glv()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_GLV, "3GLSZM_GLV");
}

void test_3glszm_zv()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_ZV, "3GLSZM_ZV");
}

void test_3glszm_ze()
{
    test_3glszm_feature(Nyxus::Feature3D::GLSZM_ZE, "3GLSZM_ZE");
}

