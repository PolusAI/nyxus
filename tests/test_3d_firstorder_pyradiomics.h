#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_intensity.h"
#include "test_ref_vals.h"

// Provenance (SPEC 6.4):
//   tool      = PyRadiomics 3.0.1
//   fixture   = tests/data/nifti/compat_int/compat_int_mri.nii and
//               tests/data/nifti/compat_seg/compat_seg_liver.nii, label 1
//   config    = binCount 20, no resampling, no weighting
//   recipe    = firstorder3d.pyradiomics_bincount20
//   generator = tests/vetting/oracles/gen_firstorder3d_pyradiomics.py
//
// PyRadiomics computes the statistics from the original intensities. binCount affects only
// Entropy and Uniformity; Nyxus selects the matching bin-count mode with GREYDEPTH = -20.

static const ref_vals_map<double> firstorder_3d_pyradiomics_ref_vals
{
    {"3P10", 362.0}, // Case-1_original_firstorder_10Percentile
    {"3P90", 527.0}, // Case-1_original_firstorder_90Percentile
    {"3ENERGY", 965351311.0}, // Case-1_original_firstorder_Energy
    {"3ENTROPY", 3.593829136968073}, // Case-1_original_firstorder_Entropy
    {"3INTERQUARTILE_RANGE", 79.0}, // Case-1_original_firstorder_InterquartileRange
    {"3KURTOSIS", 3.2668612130047703}, // Case-1_original_firstorder_Kurtosis
    {"3MAX", 653.0}, // Case-1_original_firstorder_Maximum
    {"3MEAN_ABSOLUTE_DEVIATION", 50.10342447916667}, // Case-1_original_firstorder_MeanAbsoluteDeviation
    {"3MEAN", 443.754375}, // Case-1_original_firstorder_Mean
    {"3MEDIAN", 442.0}, // Case-1_original_firstorder_Median
    {"3MIN", 212.0}, // Case-1_original_firstorder_Minimum
    {"3RANGE", 441.0}, // Case-1_original_firstorder_Range
    {"3ROBUST_MEAN_ABSOLUTE_DEVIATION", 33.49484999156687}, // Case-1_original_firstorder_RobustMeanAbsoluteDeviation
    {"3ROOT_MEAN_SQUARED", 448.45831072501414}, // Case-1_original_firstorder_RootMeanSquared
    {"3SKEWNESS", 0.035921672542198836}, // Case-1_original_firstorder_Skewness
    // Case-1_original_firstorder_TotalEnergy: 2124273520.80537
    {"3UNIFORMITY", 0.10037109374999999}, // Case-1_original_firstorder_Uniformity
    {"3VARIANCE", 4196.911126692708}, // Case-1_original_firstorder_Variance
};

// agrees_gt accepts a divisor: 100 means rel=1e-2, 1000 means rel=1e-3, and 1e9 means rel=1e-9.
// The bands are based on the residuals recorded in firstorder_3d_pyradiomics_vetting_report.md.
static double firstorder_3d_pyradiomics_divisor(Nyxus::Feature3D feature)
{
    switch (feature)
    {
    // Nyxus uses a 100-bin interpolated CDF; PyRadiomics uses sample percentiles.
    // The worst measured residual is 4.99e-3 on robust mean absolute deviation.
    case Nyxus::Feature3D::P10:
    case Nyxus::Feature3D::P90:
    case Nyxus::Feature3D::INTERQUARTILE_RANGE:
    case Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION:
        return 100.0;
    // PyRadiomics uses population variance (N); Nyxus 3VARIANCE uses sample variance (N-1).
    // Their measured residual on this ROI is 2.08e-4.
    case Nyxus::Feature3D::VARIANCE:
        return 1000.0;
    default:
        // The remaining twelve features agree to better than rel=1e-12.
        return 1.0e9;
    }
}

static std::tuple<std::string, std::string, int> get_3d_compat_fo_phantom()
{
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/compat_int/compat_int_mri.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/compat_seg/compat_seg_liver.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    return { ipath, mpath, 1 };
}

void assert_3d_firstorder_feature_pyradiomics (const Nyxus::Feature3D &expected_fcode, const std::string &fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = firstorder_3d_pyradiomics_ref_vals.find(fname);
    ASSERT_TRUE (iter != firstorder_3d_pyradiomics_ref_vals.end());

    // check availability of GT for the requested feature
    auto [ipath, mpath, label] = get_3d_compat_fo_phantom();
    ASSERT_TRUE (fs::exists(ipath));
    ASSERT_TRUE (fs::exists(mpath));

    // (2) mock the 3D workflow

    Environment e;

    // slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE (scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    // properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
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
    s[(int)NyxSetting::GREYDEPTH].ival = -20;   // intentionally negative to activate radiomics binCount-based grey-binning
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    // (4) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE (e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE ((int) expected_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW (r.initialize_fvals());
    D3_VoxelIntensityFeatures f;
    ASSERT_NO_THROW (f.calculate(r, s, e.dataset));

    // (6) get values

    f.save_value (r.fvals);

    // (7) verdict
    ASSERT_TRUE(agrees_gt(
        r.fvals[fcode][0],
        firstorder_3d_pyradiomics_ref_vals.at(fname),
        firstorder_3d_pyradiomics_divisor(expected_fcode))) << fname;
}

