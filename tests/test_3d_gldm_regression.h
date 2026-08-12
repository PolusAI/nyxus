#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_gldm.h"
#include "test_ref_vals.h"

// Feature values calculated on intensity ut_inten.nii and mask ut_inten.nii, label 57:
// (100 grey levels, offset 1, and asymmetric cooc matrix)
static ref_vals_map<double> gldm_3d_regression_ref_vals
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

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void assert_3d_gldm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
#if 0
    Fsettings s;

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
    D3_GLDM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, gldm_3d_regression_ref_vals[fname], 10.));
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
    D3_GLDM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) saving values

    f.save_value(r.fvals);

    // we have just 1 value, no need to aggregate subfeatures
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, gldm_3d_regression_ref_vals[fname], 10.));

}

void test_3d_gldm_sde_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE");
}

void test_3d_gldm_lde_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_LDE, "3GLDM_LDE");
}

void test_3d_gldm_lgle_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE");
}

void test_3d_gldm_hgle_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_HGLE, "3GLDM_HGLE");
}

void test_3d_gldm_sdlgle_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_SDLGLE, "3GLDM_SDLGLE");
}

void test_3d_gldm_sdhgle_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_SDHGLE, "3GLDM_SDHGLE");
}

void test_3d_gldm_ldlgle_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_LDLGLE, "3GLDM_LDLGLE");
}

void test_3d_gldm_ldhgle_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_LDHGLE, "3GLDM_LDHGLE");
}

void test_3d_gldm_gln_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_GLN, "3GLDM_GLN");
}

void test_3d_gldm_dn_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DN, "3GLDM_DN");
}

void test_3d_gldm_dnn_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DNN, "3GLDM_DNN");
}

void test_3d_gldm_glv_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_GLV, "3GLDM_GLV");
}

void test_3d_gldm_dv_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DV, "3GLDM_DV");
}

void test_3d_gldm_de_regression()
{
    assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DE, "3GLDM_DE");
}

