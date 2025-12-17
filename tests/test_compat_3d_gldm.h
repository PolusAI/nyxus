#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/raw_nifti.h"

#include "../src/nyx/helpers/fsystem.h"

// Feature values calculated on intensity ut_inten.nii and mask ut_inten.nii, label 57:
// (100 grey levels, offset 1, and asymmetric cooc matrix)
//
// Getting Pyradiomics ground truth values:
//      pyradiomics mri.nii.gz liver.nii.gz --param settings1.yaml
// 
// where file "settings1.yaml" is:
// 
//  setting:
//  #disabled - binWidth: 25
//  binCount : 20
//  label : 1
//  interpolator : 'sitkBSpline'
//  resampledPixelSpacing :
//  weightingNorm: 
//
//  imageType :
//        Original : {} 
//  featureClass :
//      gldm:
//

static std::unordered_map<std::string, float> compat_3gldm_GT
{
    {"3GLDM_DE", 6.60487318745419}, // Case - 1_original_gldm_DependenceEntropy
    {"3GLDM_DN", 620.2816666666666}, // Case - 1_original_gldm_DependenceNonUniformity
    {"3GLDM_DNN", 0.12922534722222223}, // Case - 1_original_gldm_DependenceNonUniformityNormalized
    {"3GLDM_DV", 5.425478993055556}, // Case - 1_original_gldm_DependenceVariance
    {"3GLDM_GLN", 481.78125}, // Case - 1_original_gldm_GrayLevelNonUniformity
    {"3GLDM_GLV", 8.728494401041667}, // Case - 1_original_gldm_GrayLevelVariance
    {"3GLDM_HGLE", 129.87979166666668}, // Case - 1_original_gldm_HighGrayLevelEmphasis
    {"3GLDM_LDE", 24.279166666666665}, // Case - 1_original_gldm_LargeDependenceEmphasis
    {"3GLDM_LDHGLE", 3061.1764583333334}, // Case - 1_original_gldm_LargeDependenceHighGrayLevelEmphasis
    {"3GLDM_LDLGLE", 0.252649584876794}, // Case - 1_original_gldm_LargeDependenceLowGrayLevelEmphasis
    {"3GLDM_LGLE", 0.012371308742463947}, // Case - 1_original_gldm_LowGrayLevelEmphasis
    {"3GLDM_SDE", 0.1635035514256671}, // Case - 1_original_gldm_SmallDependenceEmphasis
    {"3GLDM_SDHGLE", 21.9586484612667}, // Case - 1_original_gldm_SmallDependenceHighGrayLevelEmphasis
    {"3GLDM_SDLGLE", 0.0024445083605478196}  // Case - 1_original_gldm_SmallDependenceLowGrayLevelEmphasis
};

void test_compat_3gldm_feature (const Nyxus::Feature3D & expecting_fcode, const std::string & fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = compat_3gldm_GT.find (fname);
    ASSERT_TRUE (iter != compat_3gldm_GT.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // (2) mock the 3D workflow

    Environment e;

    // slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back (ipath, mpath);
    ASSERT_TRUE (scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    // properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois (e.uniqueLabels, e.roiData);
    ASSERT_TRUE (gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

    // voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE (scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

    // buffers
    ASSERT_NO_THROW (allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (3) common feature extraction settings

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    // (4) GLCM-specific feature settings mocking default pyRadiomics settings

    s[(int)NyxSetting::GLDM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE (e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE ((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW (r.initialize_fvals());
    D3_GLDM_feature f;
    ASSERT_NO_THROW (f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // aggregate angled subfeatures (13 angles for 3D)
    double atot = f.calc_ave(r.fvals[fcode]);

    // (7) verdict
    ASSERT_TRUE(agrees_gt(atot, compat_3gldm_GT[fname], 10.));
}

void test_compat_3GLDM_DE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_DE, "3GLDM_DE"); 
}

void test_compat_3GLDM_DN() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_DN, "3GLDM_DN"); 
}

void test_compat_3GLDM_DNN() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_DNN, "3GLDM_DNN"); 
}

void test_compat_3GLDM_DV() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_DV, "3GLDM_DV"); 
}

void test_compat_3GLDM_GLN() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_GLN, "3GLDM_GLN"); 
}

void test_compat_3GLDM_GLV() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_GLV, "3GLDM_GLV"); 
}

void test_compat_3GLDM_HGLE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_HGLE, "3GLDM_HGLE"); 
}

void test_compat_3GLDM_LDE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_LDE, "3GLDM_LDE"); 
}

void test_compat_3GLDM_LDHGLE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_LDHGLE, "3GLDM_LDHGLE"); 
}

void test_compat_3GLDM_LDLGLE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_LDLGLE, "3GLDM_LDLGLE"); 
}

void test_compat_3GLDM_LGLE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_LGLE, "3GLDM_LGLE");
}

void test_compat_3GLDM_SDE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE"); 
}

void test_compat_3GLDM_SDHGLE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_SDHGLE, "3GLDM_SDHGLE"); 
}

void test_compat_3GLDM_SDLGLE() { 
    test_compat_3gldm_feature(Nyxus::Feature3D::GLDM_SDLGLE, "3GLDM_SDLGLE"); 
}



