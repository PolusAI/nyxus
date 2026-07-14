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
