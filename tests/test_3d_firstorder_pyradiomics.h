#pragma once

#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "../src/nyx/environment.h"				// Environment
#include "../src/nyx/feature_settings.h"			// Fsettings, NyxSetting
#include "../src/nyx/featureset.h"				// Nyxus::Feature3D
#include "../src/nyx/helpers/fsystem.h"			// fs::exists
#include "../src/nyx/slideprops.h"				// SlideProps, scan_slide_props
#include "../src/nyx/features/3d_intensity.h"	// D3_VoxelIntensityFeatures
#include "test_main_nyxus.h"					// agrees_gt; also supplies LR (roi_cache.h) and the 3D workflow (globals.h)
#include "test_ref_vals.h"						// ref_vals_map

// PROVENANCE (SPEC 6.4)
//   oracle  : PyRadiomics 3.0.1
//   fixture : tests/data/nifti/compat_int/compat_int_mri.nii +
//             tests/data/nifti/compat_seg/compat_seg_liver.nii, label 1 (get_3d_compat_fo_phantom)
//   recipe  : firstorder3d.pyradiomics_bincount20
//
//      pyradiomics compat_int_mri.nii compat_seg_liver.nii --param settings1.yaml
//
//   where settings1.yaml is:
//
//      setting:
//        binCount: 20
//        label: 1
//        interpolator: 'sitkBSpline'
//        resampledPixelSpacing:
//        weightingNorm:
//      imageType:
//        Original: {}
//      featureClass:
//        firstorder:
//
// PyRadiomics computes first-order features on the original intensities; only Entropy and
// Uniformity read the discretized histogram, which is what binCount 20 sets and what the
// GREYDEPTH = -20 setting below matches on the Nyxus side.

// Three bands, each set from the measured Nyxus-vs-PyRadiomics residual (audit report
// firstorder_3d_pyradiomics_vetting_report.md). agrees_gt's third argument is a DIVISOR
// (tolerance = golden / frac), so a larger number is a tighter band.
//
//   _EXACT (rel 1e-9)    twelve of the seventeen, which agree to better than 1e-12.
//   _BINNED (rel 1e-2)   3P10, 3P90, 3INTERQUARTILE_RANGE, 3ROBUST_MEAN_ABSOLUTE_DEVIATION. Nyxus
//                        estimates these from the 100-bin interpolated histogram in
//                        TrivialHistogram while PyRadiomics uses numpy percentiles, so the residual
//                        is the estimator's error. Worst measured 5.0e-3, on the robust MAD.
//   _VARIANCE (rel 1e-3) 3VARIANCE alone. PyRadiomics Variance is the biased (N) estimator and
//                        Nyxus 3VARIANCE is the unbiased (N-1) one, so the two differ by exactly
//                        n/(n-1); on this ROI that is 2.1e-4. 3VARIANCE_BIASED is the feature that
//                        matches PyRadiomics term for term.
static constexpr double FO3D_PYRAD_EXACT = 1.e9;
static constexpr double FO3D_PYRAD_BINNED = 100.;
static constexpr double FO3D_PYRAD_VARIANCE = 1000.;

static double firstorder_3d_pyradiomics_band (const std::string& fname)
{
    if (fname == "3P10" || fname == "3P90" || fname == "3INTERQUARTILE_RANGE"
        || fname == "3ROBUST_MEAN_ABSOLUTE_DEVIATION")
        return FO3D_PYRAD_BINNED;
    if (fname == "3VARIANCE")
        return FO3D_PYRAD_VARIANCE;
    return FO3D_PYRAD_EXACT;
}

static ref_vals_map<double> firstorder_3d_pyradiomics_ref_vals
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
    ASSERT_TRUE (agrees_gt(r.fvals[fcode][0], firstorder_3d_pyradiomics_ref_vals[fname],
        firstorder_3d_pyradiomics_band(fname))) << fname;
}


