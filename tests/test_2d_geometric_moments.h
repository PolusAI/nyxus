#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/2d_geomoments.h"
#include "../src/nyx/features/contour.h"
#include "test_main_nyxus.h"

struct GeomomentGoldenValue
{
	Nyxus::Feature2D feature;
	const char* name;
	double golden_value;
};

static Fsettings make_2d_geomoment_settings()
{
	Fsettings s;
	s.resize(static_cast<int>(NyxSetting::__COUNT__));
	s[static_cast<int>(NyxSetting::SOFTNAN)].rval = 0.0;
	s[static_cast<int>(NyxSetting::TINY)].rval = 0.0;
	s[static_cast<int>(NyxSetting::SINGLEROI)].bval = false;
	s[static_cast<int>(NyxSetting::GREYDEPTH)].ival = 256;
	s[static_cast<int>(NyxSetting::PIXELSIZEUM)].rval = 1.0;
	s[static_cast<int>(NyxSetting::XYRES)].rval = 1.0;
	s[static_cast<int>(NyxSetting::PIXELDISTANCE)].ival = 1;
	s[static_cast<int>(NyxSetting::USEGPU)].bval = false;
	s[static_cast<int>(NyxSetting::VERBOSLVL)].ival = 0;
	s[static_cast<int>(NyxSetting::IBSI)].bval = false;
	return s;
}

static double geomoment_fixture_intensity(int x, int y)
{
	return 10.0 + 3.0 * x + 5.0 * y + static_cast<double>((x * y) % 7);
}

static void load_geomoment_fixture(LR& roidata)
{
	constexpr int width = 48;
	constexpr int height = 40;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const auto intensity = static_cast<PixIntens>(geomoment_fixture_intensity(x, y));
			if (roidata.aux_area == 0)
				init_label_record_3(roidata, x, y, intensity);
			else
				update_label_record_3(roidata, x, y, intensity);
		}
	}

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const auto intensity = static_cast<PixIntens>(geomoment_fixture_intensity(x, y));
			roidata.raw_pixels.push_back(Pixel2(x, y, intensity));
		}
	}

	roidata.make_nonanisotropic_aabb();
	roidata.aux_image_matrix.allocate(
		roidata.aabb.get_width(),
		roidata.aabb.get_height());
	roidata.aux_image_matrix.calculate_from_pixelcloud(roidata.raw_pixels, roidata.aabb);
}

static void calculate_2d_geomoment_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_2d_geomoment_settings();

	LR roidata(1401);
	load_geomoment_fixture(roidata);
	roidata.initialize_fvals();

	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);

	Smoms2D_feature shape_moments;
	shape_moments.calculate(roidata, s);
	shape_moments.save_value(roidata.fvals);

	Imoms2D_feature intensity_moments;
	intensity_moments.calculate(roidata, s);
	intensity_moments.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

// Asymmetric L-shaped binary mask (30x20 rectangle minus the top-right corner x>=18 & y>=12).
// Unlike the symmetric rectangle fixture, this exercises EVERY central-moment order with a distinct
// non-zero value, so the centroid-truncation bug is caught at all orders (not just mu10/mu01). The
// centroid is (12.7857, 8.3571) - a non-half fraction - so truncation shifts it in both axes.
static bool asym_lshape_mask(int x, int y) { return !(x >= 18 && y >= 12); }

static void load_asymmetric_shape_geomoment_fixture(LR& roidata)
{
	constexpr int width = 30, height = 20;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			if (!asym_lshape_mask(x, y))
				continue;
			if (roidata.aux_area == 0)
				init_label_record_3(roidata, x, y, 1);
			else
				update_label_record_3(roidata, x, y, 1);
		}
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			if (asym_lshape_mask(x, y))
				roidata.raw_pixels.push_back(Pixel2(x, y, 1));
	roidata.make_nonanisotropic_aabb();
	roidata.aux_image_matrix.allocate(roidata.aabb.get_width(), roidata.aabb.get_height());
	roidata.aux_image_matrix.calculate_from_pixelcloud(roidata.raw_pixels, roidata.aabb);
}

static void calculate_asymmetric_shape_geomoment_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_2d_geomoment_settings();
	LR roidata(1402);
	load_asymmetric_shape_geomoment_fixture(roidata);
	roidata.initialize_fvals();
	ContourFeature contour; contour.calculate(roidata, s); contour.save_value(roidata.fvals);
	Smoms2D_feature shape_moments; shape_moments.calculate(roidata, s); shape_moments.save_value(roidata.fvals);
	fvals = roidata.fvals;
}

static void assert_2d_geomoment_features(
	const std::vector<std::vector<double>>& fvals,
	const std::vector<GeomomentGoldenValue>& golden_values,
	const std::string& review_prefix)
{
	for (const auto& item : golden_values)
	{
		const double actual = fvals[static_cast<int>(item.feature)][0];
		const double scale = std::max({1.0, std::abs(item.golden_value), std::abs(actual)});
		const double tolerance = 1e-6 * scale;
		const std::string review_name = review_prefix + item.name;
		SCOPED_TRACE(review_name);
		ASSERT_TRUE(std::isfinite(actual)) << review_name << " returned a non-finite value";
		ASSERT_NEAR(actual, item.golden_value, tolerance) << review_name;
	}
}

// Oracle goldens for the ASYMMETRIC L-shape fixture (numpy/skimage analytic; token: analytic),
// generated by tests/vetting/oracles/gen_geomoments_analytic.py. Every order is a distinct non-zero
// value (except the definitional mu10=mu01=0), so the centroid-truncation bug is exposed at ALL orders.
// (Hu invariants are asserted on the rectangle fixture; HU_M5/M6 have a separate known formula defect.)
static const std::vector<GeomomentGoldenValue> oracle_analytic_asymmetric_shape_central_moment_golden_values{
	{Nyxus::Feature2D::CENTRAL_MOMENT_00, "CENTRAL_MOMENT_00", 504.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_01, "CENTRAL_MOMENT_01", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_02, "CENTRAL_MOMENT_02", 15331.714285714284},
	{Nyxus::Feature2D::CENTRAL_MOMENT_03, "CENTRAL_MOMENT_03", 23510.20408163261},
	{Nyxus::Feature2D::CENTRAL_MOMENT_10, "CENTRAL_MOMENT_10", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_11, "CENTRAL_MOMENT_11", -6171.428571428572},
	{Nyxus::Feature2D::CENTRAL_MOMENT_12, "CENTRAL_MOMENT_12", -22334.693877551028},
	{Nyxus::Feature2D::CENTRAL_MOMENT_13, "CENTRAL_MOMENT_13", -371765.5976676385},
	{Nyxus::Feature2D::CENTRAL_MOMENT_20, "CENTRAL_MOMENT_20", 34548.857142857145},
	{Nyxus::Feature2D::CENTRAL_MOMENT_21, "CENTRAL_MOMENT_21", -33502.040816326546},
	{Nyxus::Feature2D::CENTRAL_MOMENT_22, "CENTRAL_MOMENT_22", 929733.1034985424},
	{Nyxus::Feature2D::CENTRAL_MOMENT_23, "CENTRAL_MOMENT_23", -406547.6051645153},
	{Nyxus::Feature2D::CENTRAL_MOMENT_30, "CENTRAL_MOMENT_30", 79346.93877551015},
	{Nyxus::Feature2D::CENTRAL_MOMENT_31, "CENTRAL_MOMENT_31", -838401.166180758},
	{Nyxus::Feature2D::CENTRAL_MOMENT_32, "CENTRAL_MOMENT_32", -620474.4689712632},
	{Nyxus::Feature2D::CENTRAL_MOMENT_33, "CENTRAL_MOMENT_33", -46803800.220146365},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02, "NORM_CENTRAL_MOMENT_02", 0.06035727783176762},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03, "NORM_CENTRAL_MOMENT_03", 0.004122684096324432},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11, "NORM_CENTRAL_MOMENT_11", -0.024295432458697766},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12, "NORM_CENTRAL_MOMENT_12", -0.003916549891508218},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20, "NORM_CENTRAL_MOMENT_20", 0.13601055501565706},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21, "NORM_CENTRAL_MOMENT_21", -0.005874824837262328},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30, "NORM_CENTRAL_MOMENT_30", 0.013914058825094971},
};

// Central/normalized/Hu moment goldens (both oracle_3p sets) are taken about the TRUE fractional
// centroid and verified against an independent numpy/skimage oracle,
// tests/vetting/oracles/gen_geomoments_analytic.py (token: analytic). They were regenerated when the
// centroid-truncation bug in 2d_geomoments_basic.cpp (int-cast of originOfX/Y) was fixed; the previous
// values encoded that bug (e.g. CENTRAL_MOMENT_10/01 = 960 instead of the definitional 0).
static const std::vector<GeomomentGoldenValue> oracle_3p_shape_geomoment_feature_golden_values{
	{Nyxus::Feature2D::SPAT_MOMENT_00, "SPAT_MOMENT_00", 1920},
	{Nyxus::Feature2D::SPAT_MOMENT_01, "SPAT_MOMENT_01", 37440},
	{Nyxus::Feature2D::SPAT_MOMENT_02, "SPAT_MOMENT_02", 985920},
	{Nyxus::Feature2D::SPAT_MOMENT_03, "SPAT_MOMENT_03", 29203200},
	{Nyxus::Feature2D::SPAT_MOMENT_10, "SPAT_MOMENT_10", 45120},
	{Nyxus::Feature2D::SPAT_MOMENT_11, "SPAT_MOMENT_11", 879840},
	{Nyxus::Feature2D::SPAT_MOMENT_12, "SPAT_MOMENT_12", 23169120},
	{Nyxus::Feature2D::SPAT_MOMENT_13, "SPAT_MOMENT_13", 686275200},
	{Nyxus::Feature2D::SPAT_MOMENT_20, "SPAT_MOMENT_20", 1428800},
	{Nyxus::Feature2D::SPAT_MOMENT_21, "SPAT_MOMENT_21", 27861600},
	{Nyxus::Feature2D::SPAT_MOMENT_22, "SPAT_MOMENT_22", 733688800},
	{Nyxus::Feature2D::SPAT_MOMENT_23, "SPAT_MOMENT_23", 21732048000},
	{Nyxus::Feature2D::SPAT_MOMENT_30, "SPAT_MOMENT_30", 50895360},
	{Nyxus::Feature2D::CENTRAL_MOMENT_00, "CENTRAL_MOMENT_00", 1920},
	{Nyxus::Feature2D::CENTRAL_MOMENT_01, "CENTRAL_MOMENT_01", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_02, "CENTRAL_MOMENT_02", 255840},
	{Nyxus::Feature2D::CENTRAL_MOMENT_03, "CENTRAL_MOMENT_03", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_10, "CENTRAL_MOMENT_10", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_11, "CENTRAL_MOMENT_11", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_12, "CENTRAL_MOMENT_12", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_13, "CENTRAL_MOMENT_13", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_20, "CENTRAL_MOMENT_20", 368480},
	{Nyxus::Feature2D::CENTRAL_MOMENT_21, "CENTRAL_MOMENT_21", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_22, "CENTRAL_MOMENT_22", 49099960},
	{Nyxus::Feature2D::CENTRAL_MOMENT_23, "CENTRAL_MOMENT_23", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_30, "CENTRAL_MOMENT_30", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_31, "CENTRAL_MOMENT_31", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_32, "CENTRAL_MOMENT_32", 0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_33, "CENTRAL_MOMENT_33", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02, "NORM_CENTRAL_MOMENT_02", 0.069401041666666663},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03, "NORM_CENTRAL_MOMENT_03", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11, "NORM_CENTRAL_MOMENT_11", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12, "NORM_CENTRAL_MOMENT_12", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20, "NORM_CENTRAL_MOMENT_20", 0.099956597222222221},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21, "NORM_CENTRAL_MOMENT_21", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30, "NORM_CENTRAL_MOMENT_30", 0},
	{Nyxus::Feature2D::HU_M1, "HU_M1", 0.16935763888888888},
	{Nyxus::Feature2D::HU_M2, "HU_M2", 0.00093364197530864214},
	{Nyxus::Feature2D::HU_M3, "HU_M3", 0},
	{Nyxus::Feature2D::HU_M4, "HU_M4", 0},
	{Nyxus::Feature2D::HU_M5, "HU_M5", 0},
	{Nyxus::Feature2D::HU_M6, "HU_M6", 0},
	{Nyxus::Feature2D::HU_M7, "HU_M7", 0},
};

static const std::vector<GeomomentGoldenValue> unvetted_nyxus_regression_shape_geomoment_feature_golden_values{
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_00, "NORM_SPAT_MOMENT_00", 1},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_01, "NORM_SPAT_MOMENT_01", 0.44502457797294748},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_02, "NORM_SPAT_MOMENT_02", 0.26744791666666667},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_03, "NORM_SPAT_MOMENT_03", 0.18079123480150991},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_10, "NORM_SPAT_MOMENT_10", 0.53631167089047516},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_11, "NORM_SPAT_MOMENT_11", 0.23867187500000001},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_12, "NORM_SPAT_MOMENT_12", 0.14343543906367653},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_13, "NORM_SPAT_MOMENT_13", 0.096960449218749994},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_20, "NORM_SPAT_MOMENT_20", 0.38758680555555558},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_21, "NORM_SPAT_MOMENT_21", 0.17248565457024395},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_22, "NORM_SPAT_MOMENT_22", 0.10365928367332176},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_23, "NORM_SPAT_MOMENT_23", 0.070072297169161621},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_30, "NORM_SPAT_MOMENT_30", 0.31508310664815414},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_31, "NORM_SPAT_MOMENT_31", 0.14021972656250001},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_32, "NORM_SPAT_MOMENT_32", 0.084268320449909992},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_33, "NORM_SPAT_MOMENT_33", 0.056964263916015626},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00, "WEIGHTED_SPAT_MOMENT_00", 2498.7137745347236},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_01, "WEIGHTED_SPAT_MOMENT_01", 56490.467057333364},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_02, "WEIGHTED_SPAT_MOMENT_02", 1384474.7445020285},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_03, "WEIGHTED_SPAT_MOMENT_03", 37605700.456672639},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_10, "WEIGHTED_SPAT_MOMENT_10", 66428.864797854694},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_11, "WEIGHTED_SPAT_MOMENT_11", 1505032.6106230586},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_12, "WEIGHTED_SPAT_MOMENT_12", 37215299.240302473},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_20, "WEIGHTED_SPAT_MOMENT_20", 1970501.3917437526},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_21, "WEIGHTED_SPAT_MOMENT_21", 45088115.602599062},
	{Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_30, "WEIGHTED_SPAT_MOMENT_30", 65035567.600335725},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_02, "WEIGHTED_CENTRAL_MOMENT_02", 107348.52519321327},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_03, "WEIGHTED_CENTRAL_MOMENT_03", 1451914.9452746219},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_11, "WEIGHTED_CENTRAL_MOMENT_11", 3220.9031886536227},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_12, "WEIGHTED_CENTRAL_MOMENT_12", 263093.16899784992},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_20, "WEIGHTED_CENTRAL_MOMENT_20", 204475.15323896331},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_21, "WEIGHTED_CENTRAL_MOMENT_21", 368121.24560622667},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_30, "WEIGHTED_CENTRAL_MOMENT_30", 1777311.6081707373},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_02, "WT_NORM_CTR_MOM_02", 0.017193451534902194},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_03, "WT_NORM_CTR_MOM_03", 0.0046521092895110304},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_11, "WT_NORM_CTR_MOM_11", 0.0005158752090264311},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_12, "WT_NORM_CTR_MOM_12", 0.00084298200764803871},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_20, "WT_NORM_CTR_MOM_20", 0.032749715293974795},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_21, "WT_NORM_CTR_MOM_21", 0.0011795045377311554},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_30, "WT_NORM_CTR_MOM_30", 0.0056947191497935716},
	{Nyxus::Feature2D::WEIGHTED_HU_M1, "WEIGHTED_HU_M1", 0.049943166828876992},
	{Nyxus::Feature2D::WEIGHTED_HU_M2, "WEIGHTED_HU_M2", 0.00024306185106698786},
	{Nyxus::Feature2D::WEIGHTED_HU_M3, "WEIGHTED_HU_M3", 1.1262214820995351e-05},
	{Nyxus::Feature2D::WEIGHTED_HU_M4, "WEIGHTED_HU_M4", 7.6749256254095614e-05},
	{Nyxus::Feature2D::WEIGHTED_HU_M5, "WEIGHTED_HU_M5", -3.5041912835034966e-09},
	{Nyxus::Feature2D::WEIGHTED_HU_M6, "WEIGHTED_HU_M6", 0.0046522610672326591},
	{Nyxus::Feature2D::WEIGHTED_HU_M7, "WEIGHTED_HU_M7", -1.3078000499389243e-09},
};

static const std::vector<GeomomentGoldenValue> oracle_3p_intensity_geomoment_feature_golden_values{
	{Nyxus::Feature2D::IMOM_RM_00, "IMOM_RM_00", 346635},
	{Nyxus::Feature2D::IMOM_RM_01, "IMOM_RM_01", 8040325},
	{Nyxus::Feature2D::IMOM_RM_02, "IMOM_RM_02", 227941125},
	{Nyxus::Feature2D::IMOM_RM_03, "IMOM_RM_03", 7040124229},
	{Nyxus::Feature2D::IMOM_RM_10, "IMOM_RM_10", 9253494},
	{Nyxus::Feature2D::IMOM_RM_11, "IMOM_RM_11", 210545686},
	{Nyxus::Feature2D::IMOM_RM_12, "IMOM_RM_12", 5925377592},
	{Nyxus::Feature2D::IMOM_RM_13, "IMOM_RM_13", 182290179472},
	{Nyxus::Feature2D::IMOM_RM_20, "IMOM_RM_20", 310000126},
	{Nyxus::Feature2D::IMOM_RM_21, "IMOM_RM_21", 6998259834},
	{Nyxus::Feature2D::IMOM_RM_22, "IMOM_RM_22", 196352705268},
	{Nyxus::Feature2D::IMOM_RM_23, "IMOM_RM_23", 6030684350556},
	{Nyxus::Feature2D::IMOM_RM_30, "IMOM_RM_30", 11405788056},
	{Nyxus::Feature2D::IMOM_CM_00, "IMOM_CM_00", 346635},
	{Nyxus::Feature2D::IMOM_CM_01, "IMOM_CM_01", 0},
	{Nyxus::Feature2D::IMOM_CM_02, "IMOM_CM_02", 41442859.949947387},
	{Nyxus::Feature2D::IMOM_CM_03, "IMOM_CM_03", -169617579.29906213},
	{Nyxus::Feature2D::IMOM_CM_10, "IMOM_CM_10", 0},
	{Nyxus::Feature2D::IMOM_CM_11, "IMOM_CM_11", -4092475.5980786635},
	{Nyxus::Feature2D::IMOM_CM_12, "IMOM_CM_12", 30294392.627443444},
	{Nyxus::Feature2D::IMOM_CM_13, "IMOM_CM_13", -1149919776.8029833},
	{Nyxus::Feature2D::IMOM_CM_20, "IMOM_CM_20", 62976163.595638104},
	{Nyxus::Feature2D::IMOM_CM_21, "IMOM_CM_21", 26193059.735960968},
	{Nyxus::Feature2D::IMOM_CM_22, "IMOM_CM_22", 7335096618.144124},
	{Nyxus::Feature2D::IMOM_CM_23, "IMOM_CM_23", -23452202622.584995},
	{Nyxus::Feature2D::IMOM_CM_30, "IMOM_CM_30", -232054083.11103761},
	{Nyxus::Feature2D::IMOM_CM_31, "IMOM_CM_31", -1540518263.574806},
	{Nyxus::Feature2D::IMOM_CM_32, "IMOM_CM_32", -16335606634.088764},
	{Nyxus::Feature2D::IMOM_CM_33, "IMOM_CM_33", -319341398857.81061},
	{Nyxus::Feature2D::IMOM_NCM_02, "IMOM_NCM_02", 0.00034490929226411939},
	{Nyxus::Feature2D::IMOM_NCM_03, "IMOM_NCM_03", -2.3976723321608506e-06},
	{Nyxus::Feature2D::IMOM_NCM_11, "IMOM_NCM_11", -3.4059735834985067e-05},
	{Nyxus::Feature2D::IMOM_NCM_12, "IMOM_NCM_12", 4.2823407410130649e-07},
	{Nyxus::Feature2D::IMOM_NCM_20, "IMOM_NCM_20", 0.00052412077838051119},
	{Nyxus::Feature2D::IMOM_NCM_21, "IMOM_NCM_21", 3.7025864231218435e-07},
	{Nyxus::Feature2D::IMOM_NCM_30, "IMOM_NCM_30", -3.280259374880525e-06},
	{Nyxus::Feature2D::IMOM_HU1, "IMOM_HU1", 0.00086903007064463057},
	{Nyxus::Feature2D::IMOM_HU2, "IMOM_HU2", 3.6757019176641553e-08},
	{Nyxus::Feature2D::IMOM_HU3, "IMOM_HU3", 3.3148083570532358e-11},
	{Nyxus::Feature2D::IMOM_HU4, "IMOM_HU4", 1.2244454586070576e-11},
	{Nyxus::Feature2D::IMOM_HU5, "IMOM_HU5", -5.4612988732116916e-22},
	{Nyxus::Feature2D::IMOM_HU6, "IMOM_HU6", -2.3976723312959013e-06},
	{Nyxus::Feature2D::IMOM_HU7, "IMOM_HU7", -1.4580371645729389e-22},
};

static const std::vector<GeomomentGoldenValue> unvetted_nyxus_regression_intensity_geomoment_feature_golden_values{
	{Nyxus::Feature2D::IMOM_NRM_00, "IMOM_NRM_00", 1},
	{Nyxus::Feature2D::IMOM_NRM_01, "IMOM_NRM_01", 0.039397166363954135},
	{Nyxus::Feature2D::IMOM_NRM_02, "IMOM_NRM_02", 0.001897046009773198},
	{Nyxus::Feature2D::IMOM_NRM_03, "IMOM_NRM_03", 9.9517462450555541e-05},
	{Nyxus::Feature2D::IMOM_NRM_10, "IMOM_NRM_10", 0.045341630166175047},
	{Nyxus::Feature2D::IMOM_NRM_11, "IMOM_NRM_11", 0.0017522720110346945},
	{Nyxus::Feature2D::IMOM_NRM_12, "IMOM_NRM_12", 8.3759678499449274e-05},
	{Nyxus::Feature2D::IMOM_NRM_13, "IMOM_NRM_13", 4.3766925283775395e-06},
	{Nyxus::Feature2D::IMOM_NRM_20, "IMOM_NRM_20", 0.0025799842045067059},
	{Nyxus::Feature2D::IMOM_NRM_21, "IMOM_NRM_21", 9.8925677672061715e-05},
	{Nyxus::Feature2D::IMOM_NRM_22, "IMOM_NRM_22", 4.7143264687233126e-06},
	{Nyxus::Feature2D::IMOM_NRM_23, "IMOM_NRM_23", 2.4593093284379381e-07},
	{Nyxus::Feature2D::IMOM_NRM_30, "IMOM_NRM_30", 0.00016122941125190971},
	{Nyxus::Feature2D::IMOM_NRM_31, "IMOM_NRM_31", 6.1552927901407291e-06},
	{Nyxus::Feature2D::IMOM_NRM_32, "IMOM_NRM_32", 2.9283383602959163e-07},
	{Nyxus::Feature2D::IMOM_NRM_33, "IMOM_NRM_33", 1.5262053460086997e-08},
	{Nyxus::Feature2D::IMOM_WRM_00, "IMOM_WRM_00", 512893.00982429727},
	{Nyxus::Feature2D::IMOM_WRM_01, "IMOM_WRM_01", 12144895.632545877},
	{Nyxus::Feature2D::IMOM_WRM_02, "IMOM_WRM_02", 317009217.47022206},
	{Nyxus::Feature2D::IMOM_WRM_03, "IMOM_WRM_03", 8987339922.381422},
	{Nyxus::Feature2D::IMOM_WRM_10, "IMOM_WRM_10", 14267758.335710838},
	{Nyxus::Feature2D::IMOM_WRM_11, "IMOM_WRM_11", 340242564.82820618},
	{Nyxus::Feature2D::IMOM_WRM_12, "IMOM_WRM_12", 8924143257.7546329},
	{Nyxus::Feature2D::IMOM_WRM_20, "IMOM_WRM_20", 445196520.83852273},
	{Nyxus::Feature2D::IMOM_WRM_21, "IMOM_WRM_21", 10672248980.453976},
	{Nyxus::Feature2D::IMOM_WRM_30, "IMOM_WRM_30", 15212433464.582859},
	{Nyxus::Feature2D::IMOM_WCM_02, "IMOM_WCM_02", 29427817.244042881},
	{Nyxus::Feature2D::IMOM_WCM_03, "IMOM_WCM_03", 87161304.307642862},
	{Nyxus::Feature2D::IMOM_WCM_11, "IMOM_WCM_11", 2393476.2305273479},
	{Nyxus::Feature2D::IMOM_WCM_12, "IMOM_WCM_12", -7832526.6657475829},
	{Nyxus::Feature2D::IMOM_WCM_20, "IMOM_WCM_20", 48293221.095873319},
	{Nyxus::Feature2D::IMOM_WCM_21, "IMOM_WCM_21", -2812405.0542011671},
	{Nyxus::Feature2D::IMOM_WCM_30, "IMOM_WCM_30", 141008736.4786922},
	{Nyxus::Feature2D::IMOM_WNCM_02, "IMOM_WNCM_02", 0.00011186764375543395},
	{Nyxus::Feature2D::IMOM_WNCM_03, "IMOM_WNCM_03", 4.6265447913743816e-07},
	{Nyxus::Feature2D::IMOM_WNCM_11, "IMOM_WNCM_11", 9.0986206714986243e-06},
	{Nyxus::Feature2D::IMOM_WNCM_12, "IMOM_WNCM_12", -4.1575256057220215e-08},
	{Nyxus::Feature2D::IMOM_WNCM_20, "IMOM_WNCM_20", 0.00018358306389337081},
	{Nyxus::Feature2D::IMOM_WNCM_21, "IMOM_WNCM_21", -1.4928319462526036e-08},
	{Nyxus::Feature2D::IMOM_WNCM_30, "IMOM_WNCM_30", 7.4847805511392628e-07},
	{Nyxus::Feature2D::IMOM_WHU1, "IMOM_WHU1", 0.00029545070764880473},
	{Nyxus::Feature2D::IMOM_WHU2, "IMOM_WHU2", 5.4742410780560893e-09},
	{Nyxus::Feature2D::IMOM_WHU3, "IMOM_WHU3", 1.0199796997562714e-12},
	{Nyxus::Feature2D::IMOM_WHU4, "IMOM_WHU4", 7.0017028137145065e-13},
	{Nyxus::Feature2D::IMOM_WHU5, "IMOM_WHU5", -1.0389943425054518e-24},
	{Nyxus::Feature2D::IMOM_WHU6, "IMOM_WHU6", 4.6265447915851515e-07},
	{Nyxus::Feature2D::IMOM_WHU7, "IMOM_WHU7", -4.7125728588123637e-25},
};

void test_2d_asymmetric_shape_central_moments_analytic()
{
	std::vector<std::vector<double>> fvals;
	calculate_asymmetric_shape_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, oracle_analytic_asymmetric_shape_central_moment_golden_values, "ASYMMETRIC_SHAPE_ANALYTIC__");
}

// ANALYTIC invariant guarding the centroid-truncation fix (2d_geomoments_basic.cpp): the first central
// moments mu10 and mu01 vanish by definition (moments are taken about the centroid). The (int) cast of a
// fractional centroid broke this - shape gave 960, intensity ~2.4e5. Independent of the big golden vectors.
void test_2d_geometric_moments_first_central_moment_analytic()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	ASSERT_NEAR(fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_10][0], 0.0, 1e-6);
	ASSERT_NEAR(fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_01][0], 0.0, 1e-6);
	ASSERT_NEAR(fvals[(int)Nyxus::Feature2D::IMOM_CM_10][0], 0.0, 1e-4);
	ASSERT_NEAR(fvals[(int)Nyxus::Feature2D::IMOM_CM_01][0], 0.0, 1e-4);
}

void test_2d_shape_geometric_moments_verifiable_with_3p_builtin_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, oracle_3p_shape_geomoment_feature_golden_values, "VERIFIABLE_WITH_3P_BUILTIN_ORACLE__");
}

void test_2d_shape_geometric_moments_unvetted_no_direct_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, unvetted_nyxus_regression_shape_geomoment_feature_golden_values, "UNVETTED_NO_DIRECT_ORACLE__");
}

void test_2d_intensity_geometric_moments_verifiable_with_3p_builtin_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, oracle_3p_intensity_geomoment_feature_golden_values, "VERIFIABLE_WITH_3P_BUILTIN_ORACLE__");
}

void test_2d_intensity_geometric_moments_unvetted_no_direct_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, unvetted_nyxus_regression_intensity_geomoment_feature_golden_values, "UNVETTED_NO_DIRECT_ORACLE__");
}
