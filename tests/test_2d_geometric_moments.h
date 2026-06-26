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
	{Nyxus::Feature2D::CENTRAL_MOMENT_01, "CENTRAL_MOMENT_01", 960},
	{Nyxus::Feature2D::CENTRAL_MOMENT_02, "CENTRAL_MOMENT_02", 256320},
	{Nyxus::Feature2D::CENTRAL_MOMENT_03, "CENTRAL_MOMENT_03", 384000},
	{Nyxus::Feature2D::CENTRAL_MOMENT_10, "CENTRAL_MOMENT_10", 960},
	{Nyxus::Feature2D::CENTRAL_MOMENT_11, "CENTRAL_MOMENT_11", 480},
	{Nyxus::Feature2D::CENTRAL_MOMENT_12, "CENTRAL_MOMENT_12", 128160},
	{Nyxus::Feature2D::CENTRAL_MOMENT_13, "CENTRAL_MOMENT_13", 192000},
	{Nyxus::Feature2D::CENTRAL_MOMENT_20, "CENTRAL_MOMENT_20", 368960},
	{Nyxus::Feature2D::CENTRAL_MOMENT_21, "CENTRAL_MOMENT_21", 184480},
	{Nyxus::Feature2D::CENTRAL_MOMENT_22, "CENTRAL_MOMENT_22", 49256160},
	{Nyxus::Feature2D::CENTRAL_MOMENT_23, "CENTRAL_MOMENT_23", 73792000},
	{Nyxus::Feature2D::CENTRAL_MOMENT_30, "CENTRAL_MOMENT_30", 552960},
	{Nyxus::Feature2D::CENTRAL_MOMENT_31, "CENTRAL_MOMENT_31", 276480},
	{Nyxus::Feature2D::CENTRAL_MOMENT_32, "CENTRAL_MOMENT_32", 73820160},
	{Nyxus::Feature2D::CENTRAL_MOMENT_33, "CENTRAL_MOMENT_33", 110592000},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02, "NORM_CENTRAL_MOMENT_02", 0.069531250000000003},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03, "NORM_CENTRAL_MOMENT_03", 0.0023772680447272832},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11, "NORM_CENTRAL_MOMENT_11", 0.00013020833333333333},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12, "NORM_CENTRAL_MOMENT_12", 0.00079341320992773077},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20, "NORM_CENTRAL_MOMENT_20", 0.10008680555555556},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21, "NORM_CENTRAL_MOMENT_21", 0.0011420791898210656},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30, "NORM_CENTRAL_MOMENT_30", 0.0034232659844072879},
	{Nyxus::Feature2D::HU_M1, "HU_M1", 0.16961805555555556},
	{Nyxus::Feature2D::HU_M2, "HU_M2", 0.00093370979214891993},
	{Nyxus::Feature2D::HU_M3, "HU_M3", 2.1882410402651178e-06},
	{Nyxus::Feature2D::HU_M4, "HU_M4", 3.0166188385260932e-05},
	{Nyxus::Feature2D::HU_M5, "HU_M5", 4.598098572281346e-10},
	{Nyxus::Feature2D::HU_M6, "HU_M6", 0.0023774353872890023},
	{Nyxus::Feature2D::HU_M7, "HU_M7", -2.3604559630908652e-10},
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
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_02, "WEIGHTED_CENTRAL_MOMENT_02", 108271.66085414639},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_03, "WEIGHTED_CENTRAL_MOMENT_03", 1648221.2155424568},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_11, "WEIGHTED_CENTRAL_MOMENT_11", 4109.7206134559083},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_12, "WEIGHTED_CENTRAL_MOMENT_12", 330371.75284171134},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_20, "WEIGHTED_CENTRAL_MOMENT_20", 205330.93384081553},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_21, "WEIGHTED_CENTRAL_MOMENT_21", 496695.03645982972},
	{Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_30, "WEIGHTED_CENTRAL_MOMENT_30", 2136803.5531487665},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_02, "WT_NORM_CTR_MOM_02", 0.017341305008900215},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_03, "WT_NORM_CTR_MOM_03", 0.0052810979515541092},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_11, "WT_NORM_CTR_MOM_11", 0.00065823243217178482},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_12, "WT_NORM_CTR_MOM_12", 0.0010585506185281591},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_20, "WT_NORM_CTR_MOM_20", 0.032886780561097596},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_21, "WT_NORM_CTR_MOM_21", 0.0015914703165204656},
	{Nyxus::Feature2D::WT_NORM_CTR_MOM_30, "WT_NORM_CTR_MOM_30", 0.0068465742104244923},
	{Nyxus::Feature2D::WEIGHTED_HU_M1, "WEIGHTED_HU_M1", 0.050228085569997812},
	{Nyxus::Feature2D::WEIGHTED_HU_M2, "WEIGHTED_HU_M2", 0.00024339488988301761},
	{Nyxus::Feature2D::WEIGHTED_HU_M3, "WEIGHTED_HU_M3", 1.3732402653252524e-05},
	{Nyxus::Feature2D::WEIGHTED_HU_M4, "WEIGHTED_HU_M4", 0.00010972319316066924},
	{Nyxus::Feature2D::WEIGHTED_HU_M5, "WEIGHTED_HU_M5", -4.0924793325555725e-09},
	{Nyxus::Feature2D::WEIGHTED_HU_M6, "WEIGHTED_HU_M6", 0.0052813682812053835},
	{Nyxus::Feature2D::WEIGHTED_HU_M7, "WEIGHTED_HU_M7", -3.2208361663964004e-09},
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
	{Nyxus::Feature2D::IMOM_CM_01, "IMOM_CM_01", 67720},
	{Nyxus::Feature2D::IMOM_CM_02, "IMOM_CM_02", 41456090},
	{Nyxus::Feature2D::IMOM_CM_03, "IMOM_CM_03", -145325666},
	{Nyxus::Feature2D::IMOM_CM_10, "IMOM_CM_10", 240984},
	{Nyxus::Feature2D::IMOM_CM_11, "IMOM_CM_11", -4045396},
	{Nyxus::Feature2D::IMOM_CM_12, "IMOM_CM_12", 57516022},
	{Nyxus::Feature2D::IMOM_CM_13, "IMOM_CM_13", -1233664876},
	{Nyxus::Feature2D::IMOM_CM_20, "IMOM_CM_20", 63143698},
	{Nyxus::Feature2D::IMOM_CM_21, "IMOM_CM_21", 32838808},
	{Nyxus::Feature2D::IMOM_CM_22, "IMOM_CM_22", 7407669574},
	{Nyxus::Feature2D::IMOM_CM_23, "IMOM_CM_23", -20794765652},
	{Nyxus::Feature2D::IMOM_CM_30, "IMOM_CM_30", -100592700},
	{Nyxus::Feature2D::IMOM_CM_31, "IMOM_CM_31", -1511475334},
	{Nyxus::Feature2D::IMOM_CM_32, "IMOM_CM_32", -1566202592},
	{Nyxus::Feature2D::IMOM_CM_33, "IMOM_CM_33", -370723933282},
	{Nyxus::Feature2D::IMOM_NCM_02, "IMOM_NCM_02", 0.00034501939970375497},
	{Nyxus::Feature2D::IMOM_NCM_03, "IMOM_NCM_03", -2.0542878277179582e-06},
	{Nyxus::Feature2D::IMOM_NCM_11, "IMOM_NCM_11", -3.3667914641346339e-05},
	{Nyxus::Feature2D::IMOM_NCM_12, "IMOM_NCM_12", 8.1303232350821154e-07},
	{Nyxus::Feature2D::IMOM_NCM_20, "IMOM_NCM_20", 0.00052551508786851807},
	{Nyxus::Feature2D::IMOM_NCM_21, "IMOM_NCM_21", 4.6420130323825322e-07},
	{Nyxus::Feature2D::IMOM_NCM_30, "IMOM_NCM_30", -1.4219536359618971e-06},
	{Nyxus::Feature2D::IMOM_HU1, "IMOM_HU1", 0.00087053448757227299},
	{Nyxus::Feature2D::IMOM_HU2, "IMOM_HU2", 3.7112807351259333e-08},
	{Nyxus::Feature2D::IMOM_HU3, "IMOM_HU3", 2.6788774435431954e-11},
	{Nyxus::Feature2D::IMOM_HU4, "IMOM_HU4", 2.8991603200922661e-12},
	{Nyxus::Feature2D::IMOM_HU5, "IMOM_HU5", -2.1393783155778043e-23},
	{Nyxus::Feature2D::IMOM_HU6, "IMOM_HU6", -2.0542878280693274e-06},
	{Nyxus::Feature2D::IMOM_HU7, "IMOM_HU7", 2.3835594243218353e-23},
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
	{Nyxus::Feature2D::IMOM_WCM_02, "IMOM_WCM_02", 29664420.570163991},
	{Nyxus::Feature2D::IMOM_WCM_03, "IMOM_WCM_03", 147284035.25422412},
	{Nyxus::Feature2D::IMOM_WCM_11, "IMOM_WCM_11", 2678500.1290065008},
	{Nyxus::Feature2D::IMOM_WCM_12, "IMOM_WCM_12", 19690079.853762936},
	{Nyxus::Feature2D::IMOM_WCM_20, "IMOM_WCM_20", 48636574.87205068},
	{Nyxus::Feature2D::IMOM_WCM_21, "IMOM_WCM_21", 34138173.79962717},
	{Nyxus::Feature2D::IMOM_WCM_30, "IMOM_WCM_30", 259829644.49042335},
	{Nyxus::Feature2D::IMOM_WNCM_02, "IMOM_WNCM_02", 0.00011276707339208047},
	{Nyxus::Feature2D::IMOM_WNCM_03, "IMOM_WNCM_03", 7.8178750321206185e-07},
	{Nyxus::Feature2D::IMOM_WNCM_11, "IMOM_WNCM_11", 1.0182117662266657e-05},
	{Nyxus::Feature2D::IMOM_WNCM_12, "IMOM_WNCM_12", 1.0451545777075632e-07},
	{Nyxus::Feature2D::IMOM_WNCM_20, "IMOM_WNCM_20", 0.00018488829725035266},
	{Nyxus::Feature2D::IMOM_WNCM_21, "IMOM_WNCM_21", 1.8120631752764605e-07},
	{Nyxus::Feature2D::IMOM_WNCM_30, "IMOM_WNCM_30", 1.3791825344547646e-06},
	{Nyxus::Feature2D::IMOM_WHU1, "IMOM_WHU1", 0.00029765537064243316},
	{Nyxus::Feature2D::IMOM_WHU2, "IMOM_WHU2", 5.6161730111679807e-09},
	{Nyxus::Feature2D::IMOM_WHU3, "IMOM_WHU3", 1.1923046864432925e-12},
	{Nyxus::Feature2D::IMOM_WHU4, "IMOM_WHU4", 3.1287168309169019e-12},
	{Nyxus::Feature2D::IMOM_WHU5, "IMOM_WHU5", -5.2494915279842729e-24},
	{Nyxus::Feature2D::IMOM_WHU6, "IMOM_WHU6", 7.8178750331489451e-07},
	{Nyxus::Feature2D::IMOM_WHU7, "IMOM_WHU7", -5.6202519491182231e-24},
};

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
