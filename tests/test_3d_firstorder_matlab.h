#pragma once

#include "test_3d_firstorder_common.h"	// assert_3d_firstorder_feature, ref_vals_map

// 3D first-order goldens under the `matlab` oracle token, which names MATLAB reference semantics
// supplied by GNU Octave (SPEC 4).
//
// PROVENANCE (SPEC 6.4)
//   generator : tests/vetting/oracles/gen_firstorder3d_matlab.py + .m
//   oracle    : GNU Octave 11.3.0 + `statistics`
//   fixture   : tests/data/nifti/phantoms/ut_inten.nii + ut_mask57.nii, label 57, n = 274432
//   recipe    : firstorder3d.matlab_ut_phantom -- default-constructed Fsettings, so the histogram
//               statistics use DEFAULT_NUM_HISTO_BINS (24)
//   report    : tests/vetting/audit/firstorder_3d_matlab_vetting_report.md
//
// The oracle runs on the voxels Nyxus featurizes, not on the stored NIfTI voxels. NiftiLoader
// shifts a volume whose whole-volume minimum is negative by -min before the cast to the unsigned
// buffer truncates it; ut_inten.nii stores [-1024, 2000], so the label-57 ROI spans [1024, 3024].
// The generator reproduces that transform; the loader itself is not what this file asserts.

// Every value in this table is Octave's, printed at full precision. Two bands cover them, each set
// from the measured Nyxus-vs-Octave residual rather than from a round number (report 'Result'):
//
//   _EXACT (rel 1e-9)  -- statistics where both compute the same quantity. Worst measured residual
//                         6.6e-14, and eleven of them agree bit-for-bit.
//   _BINNED (rel 5e-3) -- the percentile-derived statistics. Nyxus estimates these from the 100-bin
//                         interpolated histogram in TrivialHistogram::calc_percentiles while the
//                         oracle uses MATLAB prctile, so the residual is the estimator's error, not
//                         float noise. Worst measured 2.3e-3, on 3P01.
//
// agrees_gt's third argument is a DIVISOR (tolerance = golden / frac), so a larger number is a
// tighter band.
static constexpr double FO3D_MATLAB_EXACT = 1.e9;
static constexpr double FO3D_MATLAB_BINNED = 200.;

static ref_vals_map<double> firstorder_3d_matlab_ref_vals
{
	{ "3COV",								0.29486207043456802 },
	{ "3ENERGY",							1173347954776. },
	{ "3ENTROPY",							4.5795418896949416 },
	{ "3EXCESS_KURTOSIS",					-1.2127631603215925 },
	{ "3HYPERFLATNESS",						3.8027657005973312 },
	{ "3HYPERSKEWNESS",						0.32001332615517414 },
	{ "3INTEGRATED_INTENSITY",				544286216. },
	{ "3INTERQUARTILE_RANGE",				1018.5 },
	{ "3KURTOSIS",							1.7872368396784075 },
	{ "3MAX",								3024. },
	{ "3MEAN",								1983.3190590018658 },
	{ "3MEAN_ABSOLUTE_DEVIATION",			507.28947581807233 },
	{ "3MEDIAN",							1964.5 },
	{ "3MEDIAN_ABSOLUTE_DEVIATION",			507.12380480410445 },
	{ "3MIN",								1024. },
	{ "3MODE",								1279. },
	{ "3P01",								1037. },
	{ "3P10",								1188. },
	{ "3P25",								1469. },
	{ "3P75",								2487.5 },
	{ "3P90",								2808. },
	{ "3P99",								3002. },
	{ "3QCOD",								0.25742449134335904 },
	{ "3RANGE",								2000. },
	{ "3ROBUST_MEAN",						1976.5703930353695 },
	{ "3ROBUST_MEAN_ABSOLUTE_DEVIATION",	407.42706314293156 },
	{ "3ROOT_MEAN_SQUARED",					2067.740503875048 },
	{ "3SKEWNESS",							0.074690529125402069 },
	{ "3STANDARD_DEVIATION",				584.80556406962933 },
	{ "3STANDARD_DEVIATION_BIASED",			584.80449858510713 },
	{ "3STANDARD_ERROR",					1.116333919044723 },
	{ "3UNIFORMITY",						0.041991745610363521 },
	{ "3UNIFORMITY_PIU",					50.59288537549407 },
	{ "3VARIANCE",							341997.5477667973 },
	{ "3VARIANCE_BIASED",					341996.3015653785 },
};

static void assert_3d_firstorder_feature_matlab (
	const std::string& fname,
	const Nyxus::Feature3D& expecting_fcode,
	double frac_tolerance = FO3D_MATLAB_EXACT)
{
	auto iter = firstorder_3d_matlab_ref_vals.find(fname);
	ASSERT_TRUE (iter != firstorder_3d_matlab_ref_vals.end()) << fname;
	assert_3d_firstorder_feature (fname, expecting_fcode, iter->second, frac_tolerance);
}

void test_3d_firstorder_cov_matlab() {
	assert_3d_firstorder_feature_matlab ("3COV", Nyxus::Feature3D::COV);
}

void test_3d_firstorder_energy_matlab() {
	assert_3d_firstorder_feature_matlab ("3ENERGY", Nyxus::Feature3D::ENERGY);
}

void test_3d_firstorder_entropy_matlab() {
	assert_3d_firstorder_feature_matlab ("3ENTROPY", Nyxus::Feature3D::ENTROPY);
}

void test_3d_firstorder_excess_kurtosis_matlab() {
	assert_3d_firstorder_feature_matlab ("3EXCESS_KURTOSIS", Nyxus::Feature3D::EXCESS_KURTOSIS);
}

void test_3d_firstorder_hyperflatness_matlab() {
	assert_3d_firstorder_feature_matlab ("3HYPERFLATNESS", Nyxus::Feature3D::HYPERFLATNESS);
}

void test_3d_firstorder_hyperskewness_matlab() {
	assert_3d_firstorder_feature_matlab ("3HYPERSKEWNESS", Nyxus::Feature3D::HYPERSKEWNESS);
}

void test_3d_firstorder_integrated_intensity_matlab() {
	assert_3d_firstorder_feature_matlab ("3INTEGRATED_INTENSITY", Nyxus::Feature3D::INTEGRATED_INTENSITY);
}

void test_3d_firstorder_interquartile_range_matlab() {
	assert_3d_firstorder_feature_matlab ("3INTERQUARTILE_RANGE", Nyxus::Feature3D::INTERQUARTILE_RANGE, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_kurtosis_matlab() {
	assert_3d_firstorder_feature_matlab ("3KURTOSIS", Nyxus::Feature3D::KURTOSIS);
}

void test_3d_firstorder_max_matlab() {
	assert_3d_firstorder_feature_matlab ("3MAX", Nyxus::Feature3D::MAX);
}

void test_3d_firstorder_mean_matlab() {
	assert_3d_firstorder_feature_matlab ("3MEAN", Nyxus::Feature3D::MEAN);
}

void test_3d_firstorder_mean_absolute_deviation_matlab() {
	assert_3d_firstorder_feature_matlab ("3MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEAN_ABSOLUTE_DEVIATION);
}

void test_3d_firstorder_median_matlab() {
	assert_3d_firstorder_feature_matlab ("3MEDIAN", Nyxus::Feature3D::MEDIAN);
}

void test_3d_firstorder_median_absolute_deviation_matlab() {
	assert_3d_firstorder_feature_matlab ("3MEDIAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEDIAN_ABSOLUTE_DEVIATION);
}

void test_3d_firstorder_min_matlab() {
	assert_3d_firstorder_feature_matlab ("3MIN", Nyxus::Feature3D::MIN);
}

void test_3d_firstorder_mode_matlab() {
	assert_3d_firstorder_feature_matlab ("3MODE", Nyxus::Feature3D::MODE);
}

void test_3d_firstorder_p01_matlab() {
	assert_3d_firstorder_feature_matlab ("3P01", Nyxus::Feature3D::P01, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_p10_matlab() {
	assert_3d_firstorder_feature_matlab ("3P10", Nyxus::Feature3D::P10, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_p25_matlab() {
	assert_3d_firstorder_feature_matlab ("3P25", Nyxus::Feature3D::P25, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_p75_matlab() {
	assert_3d_firstorder_feature_matlab ("3P75", Nyxus::Feature3D::P75, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_p90_matlab() {
	assert_3d_firstorder_feature_matlab ("3P90", Nyxus::Feature3D::P90, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_p99_matlab() {
	assert_3d_firstorder_feature_matlab ("3P99", Nyxus::Feature3D::P99, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_qcod_matlab() {
	assert_3d_firstorder_feature_matlab ("3QCOD", Nyxus::Feature3D::QCOD, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_range_matlab() {
	assert_3d_firstorder_feature_matlab ("3RANGE", Nyxus::Feature3D::RANGE);
}

void test_3d_firstorder_robust_mean_matlab() {
	assert_3d_firstorder_feature_matlab ("3ROBUST_MEAN", Nyxus::Feature3D::ROBUST_MEAN, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_robust_mean_absolute_deviation_matlab() {
	assert_3d_firstorder_feature_matlab ("3ROBUST_MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION, FO3D_MATLAB_BINNED);
}

void test_3d_firstorder_root_mean_squared_matlab() {
	assert_3d_firstorder_feature_matlab ("3ROOT_MEAN_SQUARED", Nyxus::Feature3D::ROOT_MEAN_SQUARED);
}

void test_3d_firstorder_skewness_matlab() {
	assert_3d_firstorder_feature_matlab ("3SKEWNESS", Nyxus::Feature3D::SKEWNESS);
}

void test_3d_firstorder_standard_deviation_matlab() {
	assert_3d_firstorder_feature_matlab ("3STANDARD_DEVIATION", Nyxus::Feature3D::STANDARD_DEVIATION);
}

void test_3d_firstorder_standard_deviation_biased_matlab() {
	assert_3d_firstorder_feature_matlab ("3STANDARD_DEVIATION_BIASED", Nyxus::Feature3D::STANDARD_DEVIATION_BIASED);
}

void test_3d_firstorder_standard_error_matlab() {
	assert_3d_firstorder_feature_matlab ("3STANDARD_ERROR", Nyxus::Feature3D::STANDARD_ERROR);
}

void test_3d_firstorder_uniformity_matlab() {
	assert_3d_firstorder_feature_matlab ("3UNIFORMITY", Nyxus::Feature3D::UNIFORMITY);
}

void test_3d_firstorder_uniformity_piu_matlab() {
	assert_3d_firstorder_feature_matlab ("3UNIFORMITY_PIU", Nyxus::Feature3D::UNIFORMITY_PIU);
}

void test_3d_firstorder_variance_matlab() {
	assert_3d_firstorder_feature_matlab ("3VARIANCE", Nyxus::Feature3D::VARIANCE);
}

void test_3d_firstorder_variance_biased_matlab() {
	assert_3d_firstorder_feature_matlab ("3VARIANCE_BIASED", Nyxus::Feature3D::VARIANCE_BIASED);
}
