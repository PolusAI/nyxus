#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_intensity.h"

static std::unordered_map<std::string, float> d3inten_GT {
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

std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3inten_feature (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    clear_slide_rois();
    ASSERT_TRUE(gatherRoisMetrics_3D(ipath, mpath));
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(batch, ipath, mpath));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch));

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // set feature's state
    Environment::ibsi_compliance = false;

    // extract the feature
    LR& r = Nyxus::roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
	D3_VoxelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(r));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3inten_GT[fname], 10.));
}

void test_3inten_cov() { 
	test_3inten_feature ( "3COV", Nyxus::Feature3D::COV ); 
}

void test_3inten_ciir() {
	test_3inten_feature ("3COVERED_IMAGE_INTENSITY_RANGE", Nyxus::Feature3D::COVERED_IMAGE_INTENSITY_RANGE);
}

void test_3inten_energy() {
	test_3inten_feature ("3ENERGY", Nyxus::Feature3D::ENERGY);
}

void test_3inten_entropy() {
	test_3inten_feature ("3ENTROPY", Nyxus::Feature3D::ENTROPY);
}

void test_3inten_exckurtosis() {
	test_3inten_feature ("3EXCESS_KURTOSIS", Nyxus::Feature3D::EXCESS_KURTOSIS);
}

void test_3inten_hyperflatness() {
	test_3inten_feature ("3HYPERFLATNESS", Nyxus::Feature3D::HYPERFLATNESS);
}

void test_3inten_hyperskewness() {
	test_3inten_feature ("3HYPERSKEWNESS", Nyxus::Feature3D::HYPERSKEWNESS);
}

void test_3inten_ii() {
	test_3inten_feature("3INTEGRATED_INTENSITY", Nyxus::Feature3D::INTEGRATED_INTENSITY);
}

void test_3inten_iqr() {
	test_3inten_feature ("3INTERQUARTILE_RANGE", Nyxus::Feature3D::INTERQUARTILE_RANGE);
}

void test_3inten_kurtosis() {
	test_3inten_feature("3KURTOSIS", Nyxus::Feature3D::KURTOSIS);
}

void test_3inten_max() {
	test_3inten_feature("3MAX", Nyxus::Feature3D::MAX);
}

void test_3inten_mean() {
	test_3inten_feature("3MEAN", Nyxus::Feature3D::MEAN);
}

void test_3inten_mad() {
	test_3inten_feature("3MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEAN_ABSOLUTE_DEVIATION);
}

void test_3inten_median() {
	test_3inten_feature("3MEDIAN", Nyxus::Feature3D::MEDIAN);
}

void test_3inten_medianabsdev() {
	test_3inten_feature("3MEDIAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEDIAN_ABSOLUTE_DEVIATION);
}

void test_3inten_min() {
	test_3inten_feature("3MIN", Nyxus::Feature3D::MIN);
}

void test_3inten_mode() {
	test_3inten_feature("3MODE", Nyxus::Feature3D::MODE);
}

void test_3inten_p01() {
	test_3inten_feature("3P01", Nyxus::Feature3D::P01);
}

void test_3inten_p10() {
	test_3inten_feature("3P10", Nyxus::Feature3D::P10);
}

void test_3inten_p25() {
	test_3inten_feature("3P25", Nyxus::Feature3D::P25);
}

void test_3inten_p75() {
	test_3inten_feature("3P75", Nyxus::Feature3D::P75);
}

void test_3inten_p90() {
	test_3inten_feature("3P90", Nyxus::Feature3D::P90);
}

void test_3inten_p99() {
	test_3inten_feature("3P99", Nyxus::Feature3D::P99);
}

void test_3inten_qcod() {
	test_3inten_feature("3QCOD", Nyxus::Feature3D::QCOD);
}

void test_3inten_range() {
	test_3inten_feature("3RANGE", Nyxus::Feature3D::RANGE);
}

void test_3inten_robustmean() {
	test_3inten_feature("3ROBUST_MEAN", Nyxus::Feature3D::ROBUST_MEAN);
}

void test_3inten_dobustmad() {
	test_3inten_feature("3ROBUST_MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION);
}

void test_3inten_rms() {
	test_3inten_feature("3ROOT_MEAN_SQUARED", Nyxus::Feature3D::ROOT_MEAN_SQUARED);
}

void test_3inten_skewness() {
	test_3inten_feature("3SKEWNESS", Nyxus::Feature3D::SKEWNESS);
}

void test_3inten_std() {
	test_3inten_feature("3STANDARD_DEVIATION", Nyxus::Feature3D::STANDARD_DEVIATION);
}

void test_3inten_stdbiased() {
	test_3inten_feature("3STANDARD_DEVIATION_BIASED", Nyxus::Feature3D::STANDARD_DEVIATION_BIASED);
}

void test_3inten_se() {
	test_3inten_feature("3STANDARD_ERROR", Nyxus::Feature3D::STANDARD_ERROR);
}

void test_3inten_uniformity() {
	test_3inten_feature("3UNIFORMITY", Nyxus::Feature3D::UNIFORMITY);
}

void test_3inten_uniformitypiu() {
	test_3inten_feature("3UNIFORMITY_PIU", Nyxus::Feature3D::UNIFORMITY_PIU);
}

void test_3inten_variance() {
	test_3inten_feature("3VARIANCE", Nyxus::Feature3D::VARIANCE);
}

void test_3inten_variancebiased() {
	test_3inten_feature("3VARIANCE_BIASED", Nyxus::Feature3D::VARIANCE_BIASED);
}

