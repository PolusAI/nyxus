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

struct GeomomentExpectation
{
	Nyxus::Feature2D feature;
	const char* name;
	double expected;
	bool unvetted_no_direct_oracle = false;
};

static GeomomentExpectation unvetted_no_direct_oracle_geomoment(
	Nyxus::Feature2D feature,
	const char* name,
	double expected)
{
	return {feature, name, expected, true};
}

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
	const std::vector<GeomomentExpectation>& expectations)
{
	for (const auto& item : expectations)
	{
		const double actual = fvals[static_cast<int>(item.feature)][0];
		const double scale = std::max({1.0, std::abs(item.expected), std::abs(actual)});
		const double tolerance = 1e-6 * scale;
		const std::string review_name = item.unvetted_no_direct_oracle ?
			std::string("UNVETTED_NO_DIRECT_ORACLE__") + item.name :
			item.name;
		SCOPED_TRACE(review_name);
		ASSERT_TRUE(std::isfinite(actual)) << review_name << " returned a non-finite value";
		ASSERT_NEAR(actual, item.expected, tolerance) << review_name;
	}
}

static const std::vector<GeomomentExpectation> shape_geomoment_expectations{
	{Nyxus::Feature2D::SPAT_MOMENT_00, "SPAT_MOMENT_00", 1920.0},
	{Nyxus::Feature2D::SPAT_MOMENT_01, "SPAT_MOMENT_01", 37440.0},
	{Nyxus::Feature2D::SPAT_MOMENT_02, "SPAT_MOMENT_02", 985920.0},
	{Nyxus::Feature2D::SPAT_MOMENT_03, "SPAT_MOMENT_03", 29203200.0},
	{Nyxus::Feature2D::SPAT_MOMENT_10, "SPAT_MOMENT_10", 45120.0},
	{Nyxus::Feature2D::SPAT_MOMENT_11, "SPAT_MOMENT_11", 879840.0},
	{Nyxus::Feature2D::SPAT_MOMENT_12, "SPAT_MOMENT_12", 23169120.0},
	{Nyxus::Feature2D::SPAT_MOMENT_13, "SPAT_MOMENT_13", 686275200.0},
	{Nyxus::Feature2D::SPAT_MOMENT_20, "SPAT_MOMENT_20", 1428800.0},
	{Nyxus::Feature2D::SPAT_MOMENT_21, "SPAT_MOMENT_21", 27861600.0},
	{Nyxus::Feature2D::SPAT_MOMENT_22, "SPAT_MOMENT_22", 733688800.0},
	{Nyxus::Feature2D::SPAT_MOMENT_23, "SPAT_MOMENT_23", 21732048000.0},
	{Nyxus::Feature2D::SPAT_MOMENT_30, "SPAT_MOMENT_30", 50895360.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_00, "CENTRAL_MOMENT_00", 1920.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_01, "CENTRAL_MOMENT_01", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_02, "CENTRAL_MOMENT_02", 255840.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_03, "CENTRAL_MOMENT_03", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_10, "CENTRAL_MOMENT_10", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_11, "CENTRAL_MOMENT_11", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_12, "CENTRAL_MOMENT_12", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_13, "CENTRAL_MOMENT_13", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_20, "CENTRAL_MOMENT_20", 368480.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_21, "CENTRAL_MOMENT_21", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_22, "CENTRAL_MOMENT_22", 49099960.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_23, "CENTRAL_MOMENT_23", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_30, "CENTRAL_MOMENT_30", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_31, "CENTRAL_MOMENT_31", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_32, "CENTRAL_MOMENT_32", 0.0},
	{Nyxus::Feature2D::CENTRAL_MOMENT_33, "CENTRAL_MOMENT_33", 0.0},
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_00, "NORM_SPAT_MOMENT_00", 1.0),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_01, "NORM_SPAT_MOMENT_01", 0.4450245779729475),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_02, "NORM_SPAT_MOMENT_02", 0.2674479166666667),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_03, "NORM_SPAT_MOMENT_03", 0.1807912348015099),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_10, "NORM_SPAT_MOMENT_10", 0.5363116708904752),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_11, "NORM_SPAT_MOMENT_11", 0.238671875),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_12, "NORM_SPAT_MOMENT_12", 0.14343543906367653),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_13, "NORM_SPAT_MOMENT_13", 0.09696044921875),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_20, "NORM_SPAT_MOMENT_20", 0.3875868055555556),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_21, "NORM_SPAT_MOMENT_21", 0.17248565457024395),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_22, "NORM_SPAT_MOMENT_22", 0.10365928367332176),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_23, "NORM_SPAT_MOMENT_23", 0.07007229716916162),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_30, "NORM_SPAT_MOMENT_30", 0.31508310664815414),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_31, "NORM_SPAT_MOMENT_31", 0.1402197265625),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_32, "NORM_SPAT_MOMENT_32", 0.08426832044990999),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::NORM_SPAT_MOMENT_33, "NORM_SPAT_MOMENT_33", 0.056964263916015626),
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02, "NORM_CENTRAL_MOMENT_02", 0.06940104166666666},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03, "NORM_CENTRAL_MOMENT_03", 0.0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11, "NORM_CENTRAL_MOMENT_11", 0.0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12, "NORM_CENTRAL_MOMENT_12", 0.0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20, "NORM_CENTRAL_MOMENT_20", 0.09995659722222222},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21, "NORM_CENTRAL_MOMENT_21", 0.0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30, "NORM_CENTRAL_MOMENT_30", 0.0},
	{Nyxus::Feature2D::HU_M1, "HU_M1", 0.16935763888888888},
	{Nyxus::Feature2D::HU_M2, "HU_M2", 0.0009336419753086421},
	{Nyxus::Feature2D::HU_M3, "HU_M3", 0.0},
	{Nyxus::Feature2D::HU_M4, "HU_M4", 0.0},
	{Nyxus::Feature2D::HU_M5, "HU_M5", 0.0},
	{Nyxus::Feature2D::HU_M6, "HU_M6", 0.0},
	{Nyxus::Feature2D::HU_M7, "HU_M7", 0.0},
};

static const std::vector<GeomomentExpectation> shape_weighted_geomoment_expectations{
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00, "WEIGHTED_SPAT_MOMENT_00", 1895.0047386658844),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_01, "WEIGHTED_SPAT_MOMENT_01", 35936.03530272888),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_02, "WEIGHTED_SPAT_MOMENT_02", 638259.19023700757),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_03, "WEIGHTED_SPAT_MOMENT_03", 10157210.471993469),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_10, "WEIGHTED_SPAT_MOMENT_10", 43477.893744408153),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_11, "WEIGHTED_SPAT_MOMENT_11", 840156.49488028791),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_12, "WEIGHTED_SPAT_MOMENT_12", 14971355.233518988),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_20, "WEIGHTED_SPAT_MOMENT_20", 998251.12667152286),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_21, "WEIGHTED_SPAT_MOMENT_21", 19364279.49826885),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_30, "WEIGHTED_SPAT_MOMENT_30", 22560196.678065144),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_02, "WEIGHTED_CENTRAL_MOMENT_02", -43215.956990404084),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_03, "WEIGHTED_CENTRAL_MOMENT_03", -307398.96983443695),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_11, "WEIGHTED_CENTRAL_MOMENT_11", 15660.865604279265),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_12, "WEIGHTED_CENTRAL_MOMENT_12", -266466.34221961326),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_20, "WEIGHTED_CENTRAL_MOMENT_20", 719.45517772604308),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_21, "WEIGHTED_CENTRAL_MOMENT_21", -284742.93987422739),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_30, "WEIGHTED_CENTRAL_MOMENT_30", -376113.88641494792),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WT_NORM_CTR_MOM_02, "WT_NORM_CTR_MOM_02", -0.012034374825642609),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WT_NORM_CTR_MOM_03, "WT_NORM_CTR_MOM_03", -0.0019664216948016336),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WT_NORM_CTR_MOM_11, "WT_NORM_CTR_MOM_11", 0.004361091131633606),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WT_NORM_CTR_MOM_12, "WT_NORM_CTR_MOM_12", -0.0017045769429783677),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WT_NORM_CTR_MOM_20, "WT_NORM_CTR_MOM_20", 0.0002003471375382717),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WT_NORM_CTR_MOM_21, "WT_NORM_CTR_MOM_21", -0.0018214917724410384),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WT_NORM_CTR_MOM_30, "WT_NORM_CTR_MOM_30", -0.0024059888891652882),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_HU_M1, "WEIGHTED_HU_M1", -0.011834027688104336),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_HU_M2, "WEIGHTED_HU_M2", 0.00022576488494999378),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_HU_M3, "WEIGHTED_HU_M3", 1.9568245558424062e-05),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_HU_M4, "WEIGHTED_HU_M4", 3.1245039895705273e-05),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_HU_M5, "WEIGHTED_HU_M5", -3.5041912835033034e-09),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_HU_M6, "WEIGHTED_HU_M6", -0.0019662599027957515),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::WEIGHTED_HU_M7, "WEIGHTED_HU_M7", -1.3078000499389235e-09),
};

static const std::vector<GeomomentExpectation> intensity_geomoment_expectations{
	{Nyxus::Feature2D::IMOM_RM_00, "IMOM_RM_00", 346635.0},
	{Nyxus::Feature2D::IMOM_RM_01, "IMOM_RM_01", 8040325.0},
	{Nyxus::Feature2D::IMOM_RM_02, "IMOM_RM_02", 227941125.0},
	{Nyxus::Feature2D::IMOM_RM_03, "IMOM_RM_03", 7040124229.0},
	{Nyxus::Feature2D::IMOM_RM_10, "IMOM_RM_10", 9253494.0},
	{Nyxus::Feature2D::IMOM_RM_11, "IMOM_RM_11", 210545686.0},
	{Nyxus::Feature2D::IMOM_RM_12, "IMOM_RM_12", 5925377592.0},
	{Nyxus::Feature2D::IMOM_RM_13, "IMOM_RM_13", 182290179472.0},
	{Nyxus::Feature2D::IMOM_RM_20, "IMOM_RM_20", 310000126.0},
	{Nyxus::Feature2D::IMOM_RM_21, "IMOM_RM_21", 6998259834.0},
	{Nyxus::Feature2D::IMOM_RM_22, "IMOM_RM_22", 196352705268.0},
	{Nyxus::Feature2D::IMOM_RM_23, "IMOM_RM_23", 6030684350556.0},
	{Nyxus::Feature2D::IMOM_RM_30, "IMOM_RM_30", 11405788056.0},
	{Nyxus::Feature2D::IMOM_CM_00, "IMOM_CM_00", 346635.0},
	{Nyxus::Feature2D::IMOM_CM_01, "IMOM_CM_01", -4.5292836148291826e-10},
	{Nyxus::Feature2D::IMOM_CM_02, "IMOM_CM_02", 41442859.94994738},
	{Nyxus::Feature2D::IMOM_CM_03, "IMOM_CM_03", -169617579.2990637},
	{Nyxus::Feature2D::IMOM_CM_10, "IMOM_CM_10", 1.7789716366678476e-09},
	{Nyxus::Feature2D::IMOM_CM_11, "IMOM_CM_11", -4092475.5980786625},
	{Nyxus::Feature2D::IMOM_CM_12, "IMOM_CM_12", 30294392.62744321},
	{Nyxus::Feature2D::IMOM_CM_13, "IMOM_CM_13", -1149919776.802983},
	{Nyxus::Feature2D::IMOM_CM_20, "IMOM_CM_20", 62976163.595638104},
	{Nyxus::Feature2D::IMOM_CM_21, "IMOM_CM_21", 26193059.735960778},
	{Nyxus::Feature2D::IMOM_CM_22, "IMOM_CM_22", 7335096618.144121},
	{Nyxus::Feature2D::IMOM_CM_23, "IMOM_CM_23", -23452202622.58501},
	{Nyxus::Feature2D::IMOM_CM_30, "IMOM_CM_30", -232054083.11103734},
	{Nyxus::Feature2D::IMOM_CM_31, "IMOM_CM_31", -1540518263.5748055},
	{Nyxus::Feature2D::IMOM_CM_32, "IMOM_CM_32", -16335606634.088945},
	{Nyxus::Feature2D::IMOM_CM_33, "IMOM_CM_33", -319341398857.8106},
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_00, "IMOM_NRM_00", 1.0),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_01, "IMOM_NRM_01", 0.039397166363954135),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_02, "IMOM_NRM_02", 0.001897046009773198),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_03, "IMOM_NRM_03", 9.951746245055554e-05),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_10, "IMOM_NRM_10", 0.04534163016617505),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_11, "IMOM_NRM_11", 0.0017522720110346945),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_12, "IMOM_NRM_12", 8.375967849944927e-05),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_13, "IMOM_NRM_13", 4.3766925283775395e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_20, "IMOM_NRM_20", 0.002579984204506706),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_21, "IMOM_NRM_21", 9.892567767206171e-05),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_22, "IMOM_NRM_22", 4.7143264687233126e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_23, "IMOM_NRM_23", 2.459309328437938e-07),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_30, "IMOM_NRM_30", 0.0001612294112519097),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_31, "IMOM_NRM_31", 6.155292790140729e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_32, "IMOM_NRM_32", 2.928338360295916e-07),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_NRM_33, "IMOM_NRM_33", 1.5262053460086997e-08),
	{Nyxus::Feature2D::IMOM_NCM_02, "IMOM_NCM_02", 0.00034490929226411933},
	{Nyxus::Feature2D::IMOM_NCM_03, "IMOM_NCM_03", -2.397672332160873e-06},
	{Nyxus::Feature2D::IMOM_NCM_11, "IMOM_NCM_11", -3.405973583498506e-05},
	{Nyxus::Feature2D::IMOM_NCM_12, "IMOM_NCM_12", 4.282340741013032e-07},
	{Nyxus::Feature2D::IMOM_NCM_20, "IMOM_NCM_20", 0.0005241207783805112},
	{Nyxus::Feature2D::IMOM_NCM_21, "IMOM_NCM_21", 3.7025864231218165e-07},
	{Nyxus::Feature2D::IMOM_NCM_30, "IMOM_NCM_30", -3.2802593748805212e-06},
	{Nyxus::Feature2D::IMOM_HU1, "IMOM_HU1", 0.0008690300706446306},
	{Nyxus::Feature2D::IMOM_HU2, "IMOM_HU2", 3.675701917664157e-08},
	{Nyxus::Feature2D::IMOM_HU3, "IMOM_HU3", 3.314808357053234e-11},
	{Nyxus::Feature2D::IMOM_HU4, "IMOM_HU4", 1.2244454586070675e-11},
	{Nyxus::Feature2D::IMOM_HU5, "IMOM_HU5", -5.461298873211802e-22},
	{Nyxus::Feature2D::IMOM_HU6, "IMOM_HU6", -2.3976723312959237e-06},
	{Nyxus::Feature2D::IMOM_HU7, "IMOM_HU7", -1.4580371645729144e-22},
};

static const std::vector<GeomomentExpectation> intensity_weighted_geomoment_expectations{
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_00, "IMOM_WRM_00", 335162.57809632458),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_01, "IMOM_WRM_01", 6168510.5631702933),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_02, "IMOM_WRM_02", 103775456.18151155),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_03, "IMOM_WRM_03", 1168564811.2264738),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_10, "IMOM_WRM_10", 7752350.6516130678),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_11, "IMOM_WRM_11", 143242219.36890373),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_12, "IMOM_WRM_12", 2266614348.930891),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_20, "IMOM_WRM_20", 177278715.03645942),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_21, "IMOM_WRM_21", 3041066152.5966654),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WRM_30, "IMOM_WRM_30", 3674357692.8810434),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WCM_02, "IMOM_WCM_02", -9753096.9883113019),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WCM_03, "IMOM_WCM_03", -382371564.23019207),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WCM_11, "IMOM_WCM_11", 563829.80796688376),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WCM_12, "IMOM_WCM_12", -154478453.88649949),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WCM_20, "IMOM_WCM_20", -2034085.8344459396),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WCM_21, "IMOM_WCM_21", -247748623.04706782),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WCM_30, "IMOM_WCM_30", -332022960.85621363),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WNCM_02, "IMOM_WNCM_02", -8.6822342330091672e-05),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WNCM_03, "IMOM_WNCM_03", -5.8795864683994083e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WNCM_11, "IMOM_WNCM_11", 5.0192287292824919e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WNCM_12, "IMOM_WNCM_12", -2.375358191080171e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WNCM_20, "IMOM_WNCM_20", -1.8107489022072529e-05),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WNCM_21, "IMOM_WNCM_21", -3.8095391705309974e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WNCM_30, "IMOM_WNCM_30", -5.1053945702742657e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WHU1, "IMOM_WHU1", -0.0001049298313521642),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WHU2, "IMOM_WHU2", 5.474241078056087e-09),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WHU3, "IMOM_WHU3", 1.019979699756283e-12),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WHU4, "IMOM_WHU4", 7.001702813714549e-13),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WHU5, "IMOM_WHU5", -1.0389943425054495e-24),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WHU6, "IMOM_WHU6", -5.8795864704327473e-06),
	unvetted_no_direct_oracle_geomoment(Nyxus::Feature2D::IMOM_WHU7, "IMOM_WHU7", -4.712572858812538e-25),
};

void test_2d_shape_geometric_moments_mixed_verifiable_and_unvetted()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, shape_geomoment_expectations);
}

void test_2d_shape_weighted_geometric_moments_unvetted_no_direct_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, shape_weighted_geomoment_expectations);
}

void test_2d_intensity_geometric_moments_mixed_verifiable_and_unvetted()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, intensity_geomoment_expectations);
}

void test_2d_intensity_weighted_geometric_moments_unvetted_no_direct_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, intensity_weighted_geomoment_expectations);
}
