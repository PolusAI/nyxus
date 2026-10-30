#pragma once

#include "../src/nyx/featureset.h"
#include "test_2d_firstorder_common.h"
#include "test_ref_vals.h"

// Provenance (SPEC 6.4):
//   tool      = MATLAB R2026a, Statistics and Machine Learning Toolbox 26.1; the percentile
//               values (P01 to P99, INTERQUARTILE_RANGE, QCOD) are computed on GNU Octave
//               11.3.0 as quantile(x, p, 1, 7), the rule prctile Method="inclusive" takes, and
//               match numpy.percentile's default to the last digit. MATLAB run pending: the
//               generator's prctile Method="inclusive" calls have not yet been run on MATLAB
//   functions = dot, kurtosis, mad, max, mean, median, min, mode, moment,
//               numel, prctile, range, rms, skewness, sqrt, std, sum, var
//   fixture   = pixelIntensityFeaturesTestData in tests/test_data.h
//   config    = default first-order settings; slide range 0..65535 for
//               COVERED_IMAGE_INTENSITY_RANGE
//   recipe    = firstorder2d.matlab_native
//   generator = tests/vetting/oracles/gen_firstorder2d_matlab.m
//   report    = tests/vetting/audit/firstorder_2d_matlab_vetting_report.md
static const ref_vals_map<double> firstorder_2d_matlab_ref_vals
{
	{ "COV", 0.45233654983996341 },
	{ "COVERED_IMAGE_INTENSITY_RANGE", 0.80889600976577403 },
	{ "ENERGY", 196528957184.0 },
	{ "EXCESS_KURTOSIS", -1.0721112792899095 },
	{ "HYPERFLATNESS", 5.1266592430284605 },
	{ "HYPERSKEWNESS", 1.9782930866053803 },
	{ "INTEGRATED_INTENSITY", 5015224.0 },
	{ "INTERQUARTILE_RANGE", 26116.25 },
	{ "KURTOSIS", 1.9278887207100905 },
	{ "MAX", 64090.0 },
	{ "MEAN", 32566.389610389611 },
	{ "MEAN_ABSOLUTE_DEVIATION", 12833.084499915669 },
	{ "MEDIAN", 29803.5 },
	{ "MIN", 11079.0 },
	{ "MODE", 19552.0 },
	{ "P01", 12527.299999999999 },
	{ "P10", 16329.0 },
	{ "P25", 19552.0 },
	{ "P75", 45668.25 },
	{ "P90", 53295.0 },
	{ "P99", 63245.720000000001 },
	{ "QCOD", 0.40043161441423486 },
	{ "RANGE", 53011.0 },
	{ "ROOT_MEAN_SQUARED", 35723.410526381209 },
	{ "SKEWNESS", 0.45025675970449436 },
	{ "STANDARD_DEVIATION", 14730.968317107667 },
	{ "STANDARD_DEVIATION_BIASED", 14683.062602218628 },
	{ "STANDARD_ERROR", 1187.0552552255667 },
	{ "UNIFORMITY_PIU", 29.477577192725725 },
	{ "VARIANCE", 217001427.55962989 },
	{ "VARIANCE_BIASED", 215592327.38067126 }
};

static double firstorder_2d_matlab_rel_tol(Nyxus::Feature2D feature)
{
	switch (feature)
	{
	// prctile Method="inclusive" takes numpy's default 'linear' ranks i/(n-1), the rule Nyxus takes, so
	// both sides interpolate between the same two pixels by the same fraction and agree to rounding.
	// A percentile read off a 100-bin histogram misses these goldens by 1.6e-3 to 5.0e-2.
	case Nyxus::Feature2D::P01:
	case Nyxus::Feature2D::INTERQUARTILE_RANGE:
	case Nyxus::Feature2D::P10:
	case Nyxus::Feature2D::P25:
	case Nyxus::Feature2D::P75:
	case Nyxus::Feature2D::P90:
	case Nyxus::Feature2D::P99:
	case Nyxus::Feature2D::QCOD:
		return 1.0e-9;
	default:
		// 0.1%: same native statistic on the same integer pixel vector (SPEC 7).
		return 1.0e-3;
	}
}

static void assert_2d_firstorder_value_matlab(
	const std::vector<std::vector<double>>& default_values,
	const std::vector<std::vector<double>>& slide_values,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("MATLAB_ORACLE__") + feature_name);
	FeatureSet features;
	int feature_code = -1;
	ASSERT_TRUE(features.find_2D_FeatureByString(feature_name, feature_code)) << feature_name;
	ASSERT_LT(static_cast<std::size_t>(feature_code), default_values.size()) << feature_name;

	const auto feature = static_cast<Nyxus::Feature2D>(feature_code);
	const auto& values = feature == Nyxus::Feature2D::COVERED_IMAGE_INTENSITY_RANGE
		? slide_values
		: default_values;
	ASSERT_FALSE(values[feature_code].empty()) << feature_name;

	const double actual = values[feature_code][0];
	const double expected = firstorder_2d_matlab_ref_vals.at(feature_name);
	ASSERT_TRUE(agrees_gt(actual, expected, 1.0 / firstorder_2d_matlab_rel_tol(feature)))
		<< feature_name << " actual=" << actual << " MATLAB=" << expected;
}

void test_2d_firstorder_matlab()
{
	std::vector<std::vector<double>> default_values;
	std::vector<std::vector<double>> slide_values;
	calculate_pixel_intensity_feature_values(default_values);
	calculate_pixel_intensity_feature_values(slide_values, Fsettings(), 0, 0.0, 65535.0);

	for (const auto& entry : firstorder_2d_matlab_ref_vals)
		assert_2d_firstorder_value_matlab(default_values, slide_values, entry.first);
}
