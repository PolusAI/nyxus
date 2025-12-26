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
//      glcm:
//        - 'Autocorrelation'
//        - 'JointAverage'
//        - 'ClusterProminence'
//        - 'ClusterShade'
//        - 'ClusterTendency'
//        - 'Contrast'
//        - 'Correlation'
//        - 'DifferenceAverage'
//        - 'DifferenceEntropy'
//        - 'DifferenceVariance'
//        - 'JointEnergy'
//        - 'JointEntropy'
//        - 'Imc1'
//        - 'Imc2'
//        - 'Idm'
//        - 'Idmn'
//        - 'Id'
//        - 'Idn'
//        - 'InverseVariance'
//        - 'MaximumProbability'
//        - 'SumAverage'
//        - 'SumEntropy'
//        - 'SumSquares'
//

static std::unordered_map<std::string, double> compat_d3glcm_GT
{
    {"3GLCM_ACOR", 122.14708306342365},         // Case-1_original_glcm_Autocorrelation
    {"3GLCM_ASM", 0.0143339715631298},          // Case-1_original_glcm_JointEnergy
    {"3GLCM_CLUPROM", 1870.7687419551776},      // Case-1_original_glcm_ClusterProminence
    {"3GLCM_CLUSHADE", 8.755242780815239},      // Case-1_original_glcm_ClusterShade
    {"3GLCM_CLUTEND", 23.113911920055934},      // Case-1_original_glcm_ClusterTendency
    {"3GLCM_CONTRAST", 8.76143159022662},       // Case-1_original_glcm_Contrast
    {"3GLCM_CORRELATION", 0.4305477709920443},  // Case-1_original_glcm_Correlation
    {"3GLCM_DIFAVE", 2.2143984613019545},       // Case-1_original_glcm_DifferenceAverage
    {"3GLCM_DIFENTRO", 2.645537347146111},      // Case-1_original_glcm_DifferenceEntropy
    {"3GLCM_DIFVAR", 3.4395235149928194},       // Case-1_original_glcm_DifferenceVariance
    //  {"3GLCM_DIS", 27.5},    // deprecated in pyRadimics as equivalent to 'glcm_DifferenceAverage'
    {"3GLCM_ID", 0.47211428859469606},          // Case-1_original_glcm_Id
    {"3GLCM_IDN", 0.9822362042997563},          // Case-1_original_glcm_Idn
    {"3GLCM_IDM", 0.4040020605537021},          // Case-1_original_glcm_Idm
    {"3GLCM_IDMN", 0.9822362042997563},         // Case-1_original_glcm_Idmn
    {"3GLCM_INFOMEAS1", -0.09924883901268647},  // Case-1_original_glcm_Imc1
    {"3GLCM_INFOMEAS2", 0.5781205730305887},    // Case-1_original_glcm_Imc2
    {"3GLCM_IV", 0.36184532347527026},          // Case-1_original_glcm_InverseVariance
    {"3GLCM_JAVE", 10.888107083238083},         // Case-1_original_glcm_JointAverage
    {"3GLCM_JE", 6.701464036118752},            // Case-1_original_glcm_JointEntropy
    // only in pyRadiomics: Case-1_original_glcm_MCC
    {"3GLCM_JMAX", 0.036309525310650057},       // 1_original_glcm_MaximumProbability
    {"3GLCM_JVAR", 7.968835877570637},          // Case-1_original_glcm_SumSquares
    {"3GLCM_SUMAVERAGE", 21.776214166476173},   // Case-1_original_glcm_SumAverage
    {"3GLCM_SUMENTROPY", 4.27263829307018}      // Case-1_original_glcm_SumEntropy
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/phantoms/ut_inten.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/phantoms/ut_mask57.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    // ROI sitting in the mask phantom
    return { ipath, mpath, 57 };
}

static std::tuple<std::string, std::string, int> get_3d_compat_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/compat_int/compat_int_mri.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/compat_seg/compat_seg_liver.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    // ROI sitting in the mask phantom
    return { ipath, mpath, 1 };
}

void test_compat_3glcm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare
    
    // check that requested feature exists
    auto iter = compat_d3glcm_GT.find(fname);
    ASSERT_TRUE(iter != compat_d3glcm_GT.end());

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

    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // (7) aggregate angled subfeatures
    double atot = f.calc_ave (r.fvals[fcode]);

    // (8) verdict
    ASSERT_TRUE(agrees_gt(atot, compat_d3glcm_GT[fname], 10.));
}

void test_compat_3glcm_ACOR()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_ACOR, "3GLCM_ACOR");
}

void test_compat_3glcm_angular_2d_moment()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_ASM, "3GLCM_ASM");
}

void test_compat_3glcm_CLUPROM()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CLUPROM, "3GLCM_CLUPROM");
}

void test_compat_3glcm_CLUSHADE()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CLUSHADE, "3GLCM_CLUSHADE");
}

void test_compat_3glcm_CLUTEND()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CLUTEND, "3GLCM_CLUTEND");
}

void test_compat_3glcm_contrast()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CONTRAST, "3GLCM_CONTRAST");
}

void test_compat_3glcm_correlation()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CORRELATION, "3GLCM_CORRELATION");
}

void test_compat_3glcm_difference_average()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_DIFAVE, "3GLCM_DIFAVE");
}

void test_compat_3glcm_difference_entropy()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_DIFENTRO, "3GLCM_DIFENTRO");
}

void test_compat_3glcm_difference_variance()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_DIFVAR, "3GLCM_DIFVAR");
}

void test_compat_3glcm_ID()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_ID, "3GLCM_ID");
}

void test_compat_3glcm_IDN()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IDN, "3GLCM_IDN");
}

void test_compat_3glcm_IDM()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IDM, "3GLCM_IDM");
}

void test_compat_3glcm_IDMN()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IDMN, "3GLCM_IDMN");
}

void test_compat_3glcm_infomeas1()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_INFOMEAS1, "3GLCM_INFOMEAS1");
}

void test_compat_3glcm_infomeas2()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_INFOMEAS2, "3GLCM_INFOMEAS2");
}

void test_compat_3glcm_IV()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IV, "3GLCM_IV");
}

void test_compat_3glcm_JAVE()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JAVE, "3GLCM_JAVE");
}

void test_compat_3glcm_JE()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JE, "3GLCM_JE");
}

void test_compat_3glcm_JMAX()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JMAX, "3GLCM_JMAX");
}

void test_compat_3glcm_JVAR()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JVAR, "3GLCM_JVAR");
}

void test_compat_3glcm_sum_average()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_SUMAVERAGE, "3GLCM_SUMAVERAGE");
}

void test_compat_3glcm_sum_entropy()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_SUMENTROPY, "3GLCM_SUMENTROPY");
}


