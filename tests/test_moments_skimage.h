#pragma once

#include "test_moments_common.h"

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
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02, "NORM_CENTRAL_MOMENT_02", 0.06940104166666666},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03, "NORM_CENTRAL_MOMENT_03", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11, "NORM_CENTRAL_MOMENT_11", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12, "NORM_CENTRAL_MOMENT_12", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20, "NORM_CENTRAL_MOMENT_20", 0.09995659722222222},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21, "NORM_CENTRAL_MOMENT_21", 0},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30, "NORM_CENTRAL_MOMENT_30", 0},
	{Nyxus::Feature2D::HU_M1, "HU_M1", 0.16935763888888888},
	{Nyxus::Feature2D::HU_M2, "HU_M2", 0.0009336419753086421},
	{Nyxus::Feature2D::HU_M3, "HU_M3", 0},
	{Nyxus::Feature2D::HU_M4, "HU_M4", 0},
	{Nyxus::Feature2D::HU_M5, "HU_M5", 4.598098572281346e-10},
	{Nyxus::Feature2D::HU_M6, "HU_M6", 0},
	{Nyxus::Feature2D::HU_M7, "HU_M7", -2.3604559630908652e-10},
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
	{Nyxus::Feature2D::IMOM_CM_01, "IMOM_CM_01", -5.129550117999315e-10},
	{Nyxus::Feature2D::IMOM_CM_02, "IMOM_CM_02", 41442859.94994739},
	{Nyxus::Feature2D::IMOM_CM_03, "IMOM_CM_03", -169617579.29906213},
	{Nyxus::Feature2D::IMOM_CM_10, "IMOM_CM_10", 5.011315806768835e-10},
	{Nyxus::Feature2D::IMOM_CM_11, "IMOM_CM_11", -4092475.5980786635},
	{Nyxus::Feature2D::IMOM_CM_12, "IMOM_CM_12", 30294392.627443444},
	{Nyxus::Feature2D::IMOM_CM_13, "IMOM_CM_13", -1149919776.8029833},
	{Nyxus::Feature2D::IMOM_CM_20, "IMOM_CM_20", 62976163.595638104},
	{Nyxus::Feature2D::IMOM_CM_21, "IMOM_CM_21", 26193059.735960968},
	{Nyxus::Feature2D::IMOM_CM_22, "IMOM_CM_22", 7335096618.144124},
	{Nyxus::Feature2D::IMOM_CM_23, "IMOM_CM_23", -23452202622.584995},
	{Nyxus::Feature2D::IMOM_CM_30, "IMOM_CM_30", -232054083.1110376},
	{Nyxus::Feature2D::IMOM_CM_31, "IMOM_CM_31", -1540518263.574806},
	{Nyxus::Feature2D::IMOM_CM_32, "IMOM_CM_32", -16335606634.088764},
	{Nyxus::Feature2D::IMOM_CM_33, "IMOM_CM_33", -319341398857.8106},
	{Nyxus::Feature2D::IMOM_NCM_02, "IMOM_NCM_02", 0.0003449092922641194},
	{Nyxus::Feature2D::IMOM_NCM_03, "IMOM_NCM_03", -2.3976723321608506e-06},
	{Nyxus::Feature2D::IMOM_NCM_11, "IMOM_NCM_11", -3.405973583498507e-05},
	{Nyxus::Feature2D::IMOM_NCM_12, "IMOM_NCM_12", 4.282340741013065e-07},
	{Nyxus::Feature2D::IMOM_NCM_20, "IMOM_NCM_20", 0.0005241207783805112},
	{Nyxus::Feature2D::IMOM_NCM_21, "IMOM_NCM_21", 3.7025864231218435e-07},
	{Nyxus::Feature2D::IMOM_NCM_30, "IMOM_NCM_30", -3.280259374880525e-06},
	{Nyxus::Feature2D::IMOM_HU1, "IMOM_HU1", 0.0008690300706446306},
	{Nyxus::Feature2D::IMOM_HU2, "IMOM_HU2", 3.7112807351259333e-08},
	{Nyxus::Feature2D::IMOM_HU3, "IMOM_HU3", 2.6788774435431954e-11},
	{Nyxus::Feature2D::IMOM_HU4, "IMOM_HU4", 2.8991603200922661e-12},
	{Nyxus::Feature2D::IMOM_HU5, "IMOM_HU5", -2.1393783155778043e-23},
	{Nyxus::Feature2D::IMOM_HU6, "IMOM_HU6", -2.3976723312959013e-06},
	{Nyxus::Feature2D::IMOM_HU7, "IMOM_HU7", 2.3835594243218353e-23},
};

void test_2d_shape_geometric_moments_verifiable_with_3p_builtin_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, oracle_3p_shape_geomoment_feature_golden_values, "VERIFIABLE_WITH_3P_BUILTIN_ORACLE__");
}

void test_2d_intensity_geometric_moments_verifiable_with_3p_builtin_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, oracle_3p_intensity_geomoment_feature_golden_values, "VERIFIABLE_WITH_3P_BUILTIN_ORACLE__");
}
