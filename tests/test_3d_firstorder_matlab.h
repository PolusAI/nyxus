#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_intensity.h"

// 3D first-order goldens. These are MATLAB reference values - the registry vets these features
// with oracle=matlab, and the file is named for that (SPEC 6.2.1: status+oracle decide the file;
// target_test had kept its original _regression value).
//
// PROVENANCE RECORD MISSING (SPEC 6.4): no MATLAB version, config or generator is written down
// here - only the numbers. Tracked in not_covered.md section C.
//
// NOT WIRED IN: test_all.cc does not #include this file, so none of these assertions run
// (not_covered.md section B.1). The map also covers 17 features whose rows are oracle=pyradiomics
// and 1 that is regression-only; per SPEC 3 those would need a second (matlab) row each.
static std::unordered_map<std::string, double> d3inten_GT {
		{ "3COV",	0.3 },
		{ "3COVERED_IMAGE_INTENSITY_RANGE",	1.0 },
		{ "3ENERGY",		1173350000000.0 },
		{ "3ENTROPY",	4.24 },
		{ "3EXCESS_KURTOSIS",	-1.21 },
		{ "3HYPERFLATNESS",	3.8 },
		{ "3HYPERSKEWNESS",	0.32 },
		{ "3INTEGRATED_INTENSITY",	544286000.0 },
		{ "3INTERQUARTILE_RANGE",	1018.11 },
		{ "3KURTOSIS",	1.78 },
		{ "3MAX",	3024 },
		{ "3MEAN",	1983.32 },
		{ "3MEAN_ABSOLUTE_DEVIATION",		507.29 },
		{ "3MEDIAN",		1964.5 },
		{ "3MEDIAN_ABSOLUTE_DEVIATION",	507.12 },
		{ "3MIN",	1024 },
		{ "3MODE",	1279 },
		{ "3P01",	1039.38 },
		{ "3P10",	1189.05 },
		{ "3P25",	1469.79 },
		{ "3P75",	2487.91 },
		{ "3P90",	2808.61 },
		{ "3P99",	3002.3 },
		{ "3QCOD",	0.26 },
		{ "3RANGE",	2000.0 },
		{ "3ROBUST_MEAN",	0.0 },
		{ "3ROBUST_MEAN_ABSOLUTE_DEVIATION",	392.98 },
		{ "3ROOT_MEAN_SQUARED",	2067.74 },
		{ "3SKEWNESS",	0.075 },
		{ "3STANDARD_DEVIATION",	584.81 },
		{ "3STANDARD_DEVIATION_BIASED",	584.8 },
		{ "3STANDARD_ERROR",		1.12 },
		{ "3UNIFORMITY",	307211000.0 },
		{ "3UNIFORMITY_PIU",	50.59 },
		{ "3VARIANCE",	341998 },
		{ "3VARIANCE_BIASED",		341996 },
};

// returns intensity file path, mask file path, and ROI label
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void assert_3d_firstorder_feature_matlab (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
	Fsettings s;
	
	// get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
	Environment e;
	// (1) slide -> dataset -> prescan 
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back (ipath, mpath);
	ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();
	// (2) properties of specific ROIs sitting in 'e.uniqueLabels'
	clear_slide_rois (e.uniqueLabels, e.roiData);
	ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
	// (3) voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
	// (4) buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

	// (5) feature extraction
	
    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
	D3_VoxelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(r, s, e.dataset));

	// (6) saving values

    f.save_value(r.fvals);

    // we don't expect subfeatures so using subfeature [0]
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3inten_GT[fname], 10.));
}

void test_3d_firstorder_cov_matlab() { 
	assert_3d_firstorder_feature_matlab ( "3COV", Nyxus::Feature3D::COV ); 
}

void test_3d_firstorder_energy_matlab() {
	assert_3d_firstorder_feature_matlab ("3ENERGY", Nyxus::Feature3D::ENERGY);
}

void test_3d_firstorder_entropy_matlab() {
	assert_3d_firstorder_feature_matlab ("3ENTROPY", Nyxus::Feature3D::ENTROPY);
}

void test_3d_firstorder_exckurtosis_matlab() {
	assert_3d_firstorder_feature_matlab ("3EXCESS_KURTOSIS", Nyxus::Feature3D::EXCESS_KURTOSIS);
}

void test_3d_firstorder_hyperflatness_matlab() {
	assert_3d_firstorder_feature_matlab ("3HYPERFLATNESS", Nyxus::Feature3D::HYPERFLATNESS);
}

void test_3d_firstorder_hyperskewness_matlab() {
	assert_3d_firstorder_feature_matlab ("3HYPERSKEWNESS", Nyxus::Feature3D::HYPERSKEWNESS);
}

void test_3d_firstorder_ii_matlab() {
	assert_3d_firstorder_feature_matlab("3INTEGRATED_INTENSITY", Nyxus::Feature3D::INTEGRATED_INTENSITY);
}

void test_3d_firstorder_iqr_matlab() {
	assert_3d_firstorder_feature_matlab ("3INTERQUARTILE_RANGE", Nyxus::Feature3D::INTERQUARTILE_RANGE);
}

void test_3d_firstorder_kurtosis_matlab() {
	assert_3d_firstorder_feature_matlab("3KURTOSIS", Nyxus::Feature3D::KURTOSIS);
}

void test_3d_firstorder_max_matlab() {
	assert_3d_firstorder_feature_matlab("3MAX", Nyxus::Feature3D::MAX);
}

void test_3d_firstorder_mean_matlab() {
	assert_3d_firstorder_feature_matlab("3MEAN", Nyxus::Feature3D::MEAN);
}

void test_3d_firstorder_mad_matlab() {
	assert_3d_firstorder_feature_matlab("3MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEAN_ABSOLUTE_DEVIATION);
}

void test_3d_firstorder_median_matlab() {
	assert_3d_firstorder_feature_matlab("3MEDIAN", Nyxus::Feature3D::MEDIAN);
}

void test_3d_firstorder_medianabsdev_matlab() {
	assert_3d_firstorder_feature_matlab("3MEDIAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEDIAN_ABSOLUTE_DEVIATION);
}

void test_3d_firstorder_min_matlab() {
	assert_3d_firstorder_feature_matlab("3MIN", Nyxus::Feature3D::MIN);
}

void test_3d_firstorder_mode_matlab() {
	assert_3d_firstorder_feature_matlab("3MODE", Nyxus::Feature3D::MODE);
}

void test_3d_firstorder_p01_matlab() {
	assert_3d_firstorder_feature_matlab("3P01", Nyxus::Feature3D::P01);
}

void test_3d_firstorder_p10_matlab() {
	assert_3d_firstorder_feature_matlab("3P10", Nyxus::Feature3D::P10);
}

void test_3d_firstorder_p25_matlab() {
	assert_3d_firstorder_feature_matlab("3P25", Nyxus::Feature3D::P25);
}

void test_3d_firstorder_p75_matlab() {
	assert_3d_firstorder_feature_matlab("3P75", Nyxus::Feature3D::P75);
}

void test_3d_firstorder_p90_matlab() {
	assert_3d_firstorder_feature_matlab("3P90", Nyxus::Feature3D::P90);
}

void test_3d_firstorder_p99_matlab() {
	assert_3d_firstorder_feature_matlab("3P99", Nyxus::Feature3D::P99);
}

void test_3d_firstorder_qcod_matlab() {
	assert_3d_firstorder_feature_matlab("3QCOD", Nyxus::Feature3D::QCOD);
}

void test_3d_firstorder_range_matlab() {
	assert_3d_firstorder_feature_matlab("3RANGE", Nyxus::Feature3D::RANGE);
}

void test_3d_firstorder_robustmean_matlab() {
	assert_3d_firstorder_feature_matlab("3ROBUST_MEAN", Nyxus::Feature3D::ROBUST_MEAN);
}

void test_3d_firstorder_robustmad_matlab() {
	assert_3d_firstorder_feature_matlab("3ROBUST_MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION);
}

void test_3d_firstorder_rms_matlab() {
	assert_3d_firstorder_feature_matlab("3ROOT_MEAN_SQUARED", Nyxus::Feature3D::ROOT_MEAN_SQUARED);
}

void test_3d_firstorder_skewness_matlab() {
	assert_3d_firstorder_feature_matlab("3SKEWNESS", Nyxus::Feature3D::SKEWNESS);
}

void test_3d_firstorder_std_matlab() {
	assert_3d_firstorder_feature_matlab("3STANDARD_DEVIATION", Nyxus::Feature3D::STANDARD_DEVIATION);
}

void test_3d_firstorder_stdbiased_matlab() {
	assert_3d_firstorder_feature_matlab("3STANDARD_DEVIATION_BIASED", Nyxus::Feature3D::STANDARD_DEVIATION_BIASED);
}

void test_3d_firstorder_se_matlab() {
	assert_3d_firstorder_feature_matlab("3STANDARD_ERROR", Nyxus::Feature3D::STANDARD_ERROR);
}

void test_3d_firstorder_uniformity_matlab() {
	assert_3d_firstorder_feature_matlab("3UNIFORMITY", Nyxus::Feature3D::UNIFORMITY);
}

void test_3d_firstorder_uniformitypiu_matlab() {
	assert_3d_firstorder_feature_matlab("3UNIFORMITY_PIU", Nyxus::Feature3D::UNIFORMITY_PIU);
}

void test_3d_firstorder_variance_matlab() {
	assert_3d_firstorder_feature_matlab("3VARIANCE", Nyxus::Feature3D::VARIANCE);
}

void test_3d_firstorder_variancebiased_matlab() {
	assert_3d_firstorder_feature_matlab("3VARIANCE_BIASED", Nyxus::Feature3D::VARIANCE_BIASED);
}

