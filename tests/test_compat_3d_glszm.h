#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glszm.h"
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
//      glszm:
//

static std::unordered_map<std::string, float> compat_3glszm_GT
{
    {"3GLSZM_GLN", 61.77441860465116},  // Case-1_original_glszm_GrayLevelNonUniformity
    {"3GLSZM_GLNN", 0.07183071930773391},  // Case-1_original_glszm_GrayLevelNonUniformityNormalized
    {"3GLSZM_GLV", 14.965087885343427},  // Case-1_original_glszm_GrayLevelVariance
    {"3GLSZM_HGLZE", 134.6639534883721},  // Case-1_original_glszm_HighGrayLevelZoneEmphasis
    {"3GLSZM_LAE", 723.7093023255813},  // Case-1_original_glszm_LargeAreaEmphasis
    {"3GLSZM_LAHGLE", 87509.9523255814},  // Case-1_original_glszm_LargeAreaHighGrayLevelEmphasis
    {"3GLSZM_LALGLE", 6.280653691016313},  // Case-1_original_glszm_LargeAreaLowGrayLevelEmphasis
    {"3GLSZM_LGLZE", 0.016482439101794737},  // Case-1_original_glszm_LowGrayLevelZoneEmphasis
    {"3GLSZM_SZN", 231.4279069767442},  // Case-1_original_glszm_SizeZoneNonUniformity
    {"3GLSZM_SZNN", 0.2691022174148188},  // Case-1_original_glszm_SizeZoneNonUniformityNormalized
    {"3GLSZM_SAE", 0.5306840085503507},  // Case-1_original_glszm_SmallAreaEmphasis
    {"3GLSZM_SAHGLE", 72.65640040229414},  // Case-1_original_glszm_SmallAreaHighGrayLevelEmphasis
    {"3GLSZM_SALGLE", 0.008788101239865679},  // Case-1_original_glszm_SmallAreaLowGrayLevelEmphasis
    {"3GLSZM_ZE", 6.426417026786065},  // Case-1_original_glszm_ZoneEntropy
    {"3GLSZM_ZP", 0.17916666666666667},  // Case-1_original_glszm_ZonePercentage
    {"3GLSZM_ZV", 692.5573282855598}  // Case-1_original_glszm_ZoneVariance
};

void test_compat_3glszm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = compat_3glszm_GT.find(fname);
    ASSERT_TRUE(iter != compat_3glszm_GT.end());

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

    // (4) GLCM-specific feature settings mocking default pyRadiomics settings

    s[(int)NyxSetting::GLSZM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLSZM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // aggregate angled subfeatures (13 angles for 3D)
    double atot = f.calc_ave(r.fvals[fcode]);

    // (7) verdict
    ASSERT_TRUE (agrees_gt(atot, compat_3glszm_GT[fname], 10.));
}

void test_compat_3glszm_sae()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_SAE, "3GLSZM_SAE");
}

void test_compat_3glszm_lae()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_LAE, "3GLSZM_LAE");
}

void test_compat_3glszm_lglze()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_LGLZE, "3GLSZM_LGLZE");
}

void test_compat_3glszm_hglze()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_HGLZE, "3GLSZM_HGLZE");
}

void test_compat_3glszm_salgle()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_SALGLE, "3GLSZM_SALGLE");
}

void test_compat_3glszm_sahgle()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_SAHGLE, "3GLSZM_SAHGLE");
}

void test_compat_3glszm_lalgle()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_LALGLE, "3GLSZM_LALGLE");
}

void test_compat_3glszm_lahgle()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_LAHGLE, "3GLSZM_LAHGLE");
}

void test_compat_3glszm_gln()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_GLN, "3GLSZM_GLN");
}

void test_compat_3glszm_glnn()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_GLNN, "3GLSZM_GLNN");
}

void test_compat_3glszm_szn()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_SZN, "3GLSZM_SZN");
}

void test_compat_3glszm_sznn()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_SZNN, "3GLSZM_SZNN");
}

void test_compat_3glszm_zp()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_ZP, "3GLSZM_ZP");
}

void test_compat_3glszm_glv()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_GLV, "3GLSZM_GLV");
}

void test_compat_3glszm_zv()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_ZV, "3GLSZM_ZV");
}

void test_compat_3glszm_ze()
{
    test_compat_3glszm_feature (Nyxus::Feature3D::GLSZM_ZE, "3GLSZM_ZE");
}



