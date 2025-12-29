#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_ngtdm.h"
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
//  binWidth: 1
//  ### binCount : 20
//  label : 57
//  interpolator : 'sitkBSpline'
//  resampledPixelSpacing :
//  weightingNorm: 
//
//  imageType :
//        Original : {} 
//  featureClass :
//      ngtdm:
//

static std::tuple<std::string, std::string, int> get_3d_compat_ngtdm_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/compat_int/compat_int_ngtdm_3d.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/compat_seg/compat_seg_ngtdm_3d.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    return { ipath, mpath, 57 };
}

static std::unordered_map<std::string, double> compat_3ngtdm_GT
{
    {"3NGTDM_BUSYNESS", 4.553401556426767},         // Case-1_original_ngtdm_Busyness
    {"3NGTDM_COARSENESS", 0.030118770647251797},    // Case-1_original_ngtdm_Coarseness
    {"3NGTDM_COMPLEXITY", 32.13037220400344},       // Case-1_original_ngtdm_Complexity
    {"3NGTDM_CONTRAST", 0.23138014315250832},       // Case-1_original_ngtdm_Contrast
    {"3NGTDM_STRENGTH", 1.245800596888454}          // Case-1_original_ngtdm_Strength
};

void test_compat_3ngtdm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = compat_3ngtdm_GT.find(fname);
    ASSERT_TRUE(iter != compat_3ngtdm_GT.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_compat_ngtdm_phantom();
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

    s[(int)NyxSetting::NGTDM_GREYDEPTH].ival = 0/*no binning*/; //xxxxxxxxxx -20;  // intentionally negative to activate radiomics binCount-based grey-binning
    s[(int)NyxSetting::NGTDM_RADIUS].ival = 1;

    // (5) feature extraction

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

    // (6) get values

    f.save_value(r.fvals);

    // (7) verdict
    auto x1 = r.fvals[fcode];
    auto x2 = compat_3ngtdm_GT[fname];
    ASSERT_TRUE (agrees_gt(x1[0], x2, 10.));
}

void test_ngtd_matrix_correctness()
{
    // data (data and gt source: pyradiomics web page)

    std::vector<PixIntens> rawVolume =
    {
        1, 2, 5, 2,
        3, 5, 1, 3,
        1, 3, 5, 5,
        3, 1, 1, 1
    };

    SimpleCube <PixIntens> D (rawVolume, 4/*width*/, 4/*height*/, 1/*depth*/);
    PixIntens zeroI = 0;
    // --- unique intensities
    std::unordered_set<PixIntens> U (rawVolume.begin(), rawVolume.end());
    U.erase (0);
    // --- sorted non-zero (i.e. non-mask) intensities
    std::vector<PixIntens> I (U.begin(), U.end());
    std::sort (I.begin(), I.end());

    // zones

    std::vector <std::pair<PixIntens, double>> Zones;
    D3_NGTDM_feature::gather_zones (Zones, D, 1 /*radius*/, zeroI);

    // matrix

    std::vector <int> N;
    std::vector <double> P, S;
    double Nvp = D3_NGTDM_feature::calc_NGTDM (N, P, S, Zones, I);

    //
    // Expecting the following NGTDM:
    // 
    // I       N       P       S
    // -------------------------------
    // 1       6       0.375   13.35
    // 2       2       0.125   2.00
    // 3       4       0.25    3.03
    // 4       0       0       0
    // 5       4       0.25    10.075
    //

    ASSERT_TRUE(N[0] == 6);         ASSERT_TRUE(N[1] == 2);         ASSERT_TRUE(N[2] == 4);         ASSERT_TRUE(N[3] == 4);

    ASSERT_TRUE(agrees_gt(P[0], 0.375, 1));     ASSERT_TRUE(agrees_gt(P[1], 0.125, 1));     ASSERT_TRUE(agrees_gt(P[2], 0.25, 1));      ASSERT_TRUE(agrees_gt(P[3], 0.25, 1));

    ASSERT_TRUE(agrees_gt(S[0], 13.35, 1));     ASSERT_TRUE(agrees_gt(S[1], 2.0, 1));       ASSERT_TRUE(agrees_gt(S[2], 3.03, 1));      ASSERT_TRUE(agrees_gt(S[3], 10.075, 1));
}

void test_compat_3NGTDM_BUSYNESS() {
    test_compat_3ngtdm_feature (Nyxus::Feature3D::NGTDM_BUSYNESS, "3NGTDM_BUSYNESS");
}

void test_compat_3NGTDM_COARSENESS() {
    test_compat_3ngtdm_feature(Nyxus::Feature3D::NGTDM_COARSENESS, "3NGTDM_COARSENESS");
}

void test_compat_3NGTDM_COMPLEXITY() {
    test_compat_3ngtdm_feature(Nyxus::Feature3D::NGTDM_COMPLEXITY, "3NGTDM_COMPLEXITY");
}

void test_compat_3NGTDM_CONTRAST() {
    test_compat_3ngtdm_feature(Nyxus::Feature3D::NGTDM_CONTRAST, "3NGTDM_CONTRAST");
}

void test_compat_3NGTDM_STRENGTH() {
    test_compat_3ngtdm_feature(Nyxus::Feature3D::NGTDM_STRENGTH, "3NGTDM_STRENGTH");
}

