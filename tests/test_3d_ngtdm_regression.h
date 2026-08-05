#pragma once

// TAXONOMY: regression snapshot (SPEC §2). Not currently #included from test_all.cc
// (orphaned legacy GT table; live 3D NGTDM oracle is test_3d_ngtdm_pyradiomics.h).

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
#if 0
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    Environment e;
    clear_slide_rois (e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // set feature's state
    Environment::ibsi_compliance = false;

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_NGTDM_feature f;
    Fsettings s;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3ngtdm_GT[fname], 10.));
#endif

    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    Environment e;
    // (1) slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    // (2) properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
    // (3) voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
    // (4) buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (5) feature settings
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 64;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    //

    // (6) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_NGTDM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) saving values

    f.save_value(r.fvals);

    // we have just 1 value, no need to aggregate subfeatures
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


// --- folded from former test_3d_*_coverage.h (Wave-9 parameterized suites) ---
#include "test_3d_regression_common.h"

// Per-family slice of the 3D regression sweep (Wave 9). The shared harness, the two parameterized
// fixtures, their TEST_P bodies, and the global count-guard live in test_3d_regression_common.h; this
// file only re-instantiates the two suites for the "ngtdm" family. Every public 3D feature is
// classified into exactly one family (first-match on the calculator featuresets), so the per-family
// instantiations together reproduce the original 94-embedded + 119-unvetted split with no drift.

INSTANTIATE_TEST_SUITE_P(
	NGTDM_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("ngtdm", true)),
	sanitize_3d_feature_test_name);

INSTANTIATE_TEST_SUITE_P(
	NGTDM_UNVETTED_LOCAL_REGRESSION,
	Test3DFeature_UNVETTED_LOCAL_REGRESSION,
	testing::ValuesIn(feature_3d_cases_for_family("ngtdm", false)),
	sanitize_3d_feature_test_name);
