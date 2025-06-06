#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glrlm.h"

// dig. phantom values for intensity based features
// Calculated at 100 grey levels
static std::unordered_map<std::string, float> d3glrlm_GT{
    {"3GLRLM_SRE", 0.84},
    {"3GLRLM_LRE", 40.8},
    {"3GLRLM_LGLRE", 0.072},
    {"3GLRLM_HGLRE", 1922.0},
    {"3GLRLM_SRLGLE", 0.007},
    {"3GLRLM_SRHGLE", 1771.8},
    {"3GLRLM_LRLGLE", 37.4},
    {"3GLRLM_LRHGLE", 5678.4},
    {"3GLRLM_GLN", 5811.0},
    {"3GLRLM_GLNN", 0.026},
    {"3GLRLM_RLN", 154513.0},
    {"3GLRLM_RLNN", 0.68},
    {"3GLRLM_RP", 0.83},
    {"3GLRLM_GLV", 254.9},
    {"3GLRLM_RV", 34.9},
    {"3GLRLM_RE", 6.4}
};

std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3glrlm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
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
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3glrlm_GT[fname], 10.));
}

void test_3glrlm_sre()
{
    test_3glrlm_feature (Nyxus::Feature3D::GLRLM_SRE, "3GLRLM_SRE");
}

void test_3glrlm_lre()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_LRE, "3GLRLM_LRE");
}

void test_3glrlm_lglre()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_LGLRE, "3GLRLM_LGLRE");
}

void test_3glrlm_hglre()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_HGLRE, "3GLRLM_HGLRE");
}

void test_3glrlm_srlgle()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_SRLGLE, "3GLRLM_SRLGLE");
}

void test_3glrlm_srhgle()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_SRHGLE, "3GLRLM_SRHGLE");
}

void test_3glrlm_lrlgle()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_LRLGLE, "3GLRLM_LRLGLE");
}

void test_3glrlm_lrhgle()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_LRHGLE, "3GLRLM_LRHGLE");
}

void test_3glrlm_gln()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_GLN, "3GLRLM_GLN");
}

void test_3glrlm_glnn()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_GLNN, "3GLRLM_GLNN");
}

void test_3glrlm_rln()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_RLN, "3GLRLM_RLN");
}

void test_3glrlm_rlnn()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_RLNN, "3GLRLM_RLNN");
}

void test_3glrlm_rp()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_RP, "3GLRLM_RP");
}

void test_3glrlm_glv()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_GLV, "3GLRLM_GLV");
}

void test_3glrlm_rv()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_RV, "3GLRLM_RV");
}

void test_3glrlm_re()
{
    test_3glrlm_feature(Nyxus::Feature3D::GLRLM_RE, "3GLRLM_RE");
}