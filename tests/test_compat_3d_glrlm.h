#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glrlm.h"
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
//      glrlm:
//

static std::unordered_map<std::string, float> compat_3glrlm_GT
{
    {"3GLRLM_GLN", 406.68709120394277}, // Case-1_original_glrlm_GrayLevelNonUniformity
    {"3GLRLM_GLNN", 0.09722976558135092}, // Case-1_original_glrlm_GrayLevelNonUniformityNormalized
    {"3GLRLM_GLV", 9.100102904831404}, // Case-1_original_glrlm_GrayLevelVariance
    {"3GLRLM_HGLRE", 130.25347348795043}, // Case-1_original_glrlm_HighGrayLevelRunEmphasis
    {"3GLRLM_LRE", 1.5538285862328314}, // Case-1_original_glrlm_LongRunEmphasis
    {"3GLRLM_LRHGLE", 200.98033929654184}, // Case-1_original_glrlm_LongRunHighGrayLevelEmphasis
    {"3GLRLM_LRLGLE", 0.01863138831176311}, // Case-1_original_glrlm_LongRunLowGrayLevelEmphasis
    {"3GLRLM_LGLRE", 0.012578735424633676}, // Case-1_original_glrlm_LowGrayLevelRunEmphasis
    {"3GLRLM_RE", 4.228290966541947}, // Case-1_original_glrlm_RunEntropy
    {"3GLRLM_RLN", 3309.7814564084974}, // Case-1_original_glrlm_RunLengthNonUniformity
    {"3GLRLM_RLNN", 0.7807974007564221}, // Case-1_original_glrlm_RunLengthNonUniformityNormalized
    {"3GLRLM_RP", 0.8714583333333334}, // Case-1_original_glrlm_RunPercentage
    {"3GLRLM_RV", 0.19950155996777244}, // Case-1_original_glrlm_RunVariance
    {"3GLRLM_SRE", 0.9003824440228139}, // Case-1_original_glrlm_ShortRunEmphasis
    {"3GLRLM_SRHGLE", 117.56903884692184}, // Case-1_original_glrlm_ShortRunHighGrayLevelEmphasis
    {"3GLRLM_SRLGLE", 0.011465297979291003} // Case-1_original_glrlm_ShortRunLowGrayLevelEmphasis
};

void test_compat_3glrlm_feature(const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = compat_3glrlm_GT.find(fname);
    ASSERT_TRUE(iter != compat_3glrlm_GT.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // (2) mock the 3D workflow

    Environment e;

    // slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    // properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

    // voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

    // buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

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

    // (4) NGTDM-specific feature settings mocking default pyRadiomics settings

    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // (7) aggregate angled subfeatures
    double atot = f.calc_ave (r.fvals[fcode]);

    // (8) verdict
    ASSERT_TRUE(agrees_gt(atot, compat_3glrlm_GT[fname], 10.));
}

void test_compat_3GLRLM_GLN() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_GLN, "3GLRLM_GLN");
}

void test_compat_3GLRLM_GLNN() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_GLNN, "3GLRLM_GLNN");
}

void test_compat_3GLRLM_GLV() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_GLV, "3GLRLM_GLV");
}

void test_compat_3GLRLM_HGLRE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_HGLRE, "3GLRLM_HGLRE");
}

void test_compat_3GLRLM_LRE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_LRE, "3GLRLM_LRE");
}

void test_compat_3GLRLM_LRHGLE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_LRHGLE, "3GLRLM_LRHGLE");
}

void test_compat_3GLRLM_LRLGLE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_LRLGLE, "3GLRLM_LRLGLE");
}

void test_compat_3GLRLM_LGLRE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_LGLRE, "3GLRLM_LGLRE");
}

void test_compat_3GLRLM_RE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_RE, "3GLRLM_RE");
}

void test_compat_3GLRLM_RLN() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_RLN, "3GLRLM_RLN");
}

void test_compat_3GLRLM_RLNN() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_RLNN, "3GLRLM_RLNN");
}

void test_compat_3GLRLM_RP() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_RP, "3GLRLM_RP");
}

void test_compat_3GLRLM_RV() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_RV, "3GLRLM_RV");
}

void test_compat_3GLRLM_SRE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_SRE, "3GLRLM_SRE");
}

void test_compat_3GLRLM_SRHGLE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_SRHGLE, "3GLRLM_SRHGLE");
}

void test_compat_3GLRLM_SRLGLE() {
    test_compat_3glrlm_feature (Nyxus::Feature3D::GLRLM_SRLGLE, "3GLRLM_SRLGLE");
}








