#pragma once

#include "test_2d_moments_common.h"
#include "test_ref_vals.h"

static const ref_vals_list<GeomomentGoldenValue> moments_2d_skimage_shape_ref_vals{
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
	{Nyxus::Feature2D::HU_M5, "HU_M5", 0},	// exactly 0: the odd etas vanish on this symmetric fixture
	{Nyxus::Feature2D::HU_M6, "HU_M6", 0},
	{Nyxus::Feature2D::HU_M7, "HU_M7", -2.3604559630908652e-10},
};

static const ref_vals_list<GeomomentGoldenValue> moments_2d_skimage_intensity_ref_vals{
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
	{Nyxus::Feature2D::IMOM_HU5, "IMOM_HU5", -1.9898126265836865e-22},
	{Nyxus::Feature2D::IMOM_HU6, "IMOM_HU6", -6.66827346717698e-17},	// distinct from IMOM_NCM_03; a value equal to it means h6 lost the (eta21+eta03) grouping
	{Nyxus::Feature2D::IMOM_HU7, "IMOM_HU7", 2.3835594243218353e-23},
};

// Asymmetric wedge fixture (see load_wedge_fixture in test_2d_moments_common.h): the ONLY fixture in
// this file whose odd-order etas are big enough to vet Hu h5/h6 against skimage above the assertion
// tolerance. Provenance: scikit-image 0.26.0 / numpy 2.4.6, generator
// tests/vetting/oracles/gen_moments_skimage.py (section C), 2026-07-16.
static const ref_vals_list<GeomomentGoldenValue> moments_2d_skimage_wedge_hu_ref_vals{
	{Nyxus::Feature2D::SPAT_MOMENT_00, "SPAT_MOMENT_00", 180},
	{Nyxus::Feature2D::SPAT_MOMENT_10, "SPAT_MOMENT_10", 4560},
	{Nyxus::Feature2D::SPAT_MOMENT_01, "SPAT_MOMENT_01", 420},
	{Nyxus::Feature2D::CENTRAL_MOMENT_20, "CENTRAL_MOMENT_20", 17860},
	{Nyxus::Feature2D::CENTRAL_MOMENT_02, "CENTRAL_MOMENT_02", 699.9999999999999},
	{Nyxus::Feature2D::CENTRAL_MOMENT_11, "CENTRAL_MOMENT_11", 1750},
	{Nyxus::Feature2D::CENTRAL_MOMENT_30, "CENTRAL_MOMENT_30", -99166.66666666698},
	{Nyxus::Feature2D::CENTRAL_MOMENT_03, "CENTRAL_MOMENT_03", 793.3333333333339},
	{Nyxus::Feature2D::CENTRAL_MOMENT_21, "CENTRAL_MOMENT_21", -9916.666666666686},
	{Nyxus::Feature2D::CENTRAL_MOMENT_12, "CENTRAL_MOMENT_12", 1983.3333333333285},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20, "NORM_CENTRAL_MOMENT_20", 0.5512345679012346},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02, "NORM_CENTRAL_MOMENT_02", 0.021604938271604934},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11, "NORM_CENTRAL_MOMENT_11", 0.05401234567901234},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30, "NORM_CENTRAL_MOMENT_30", -0.22813107795136814},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03, "NORM_CENTRAL_MOMENT_03", 0.0018250486236109408},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21, "NORM_CENTRAL_MOMENT_21", -0.022813107795136785},
	{Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12, "NORM_CENTRAL_MOMENT_12", 0.004562621559027338},
	{Nyxus::Feature2D::HU_M1, "HU_M1", 0.5728395061728395},
	{Nyxus::Feature2D::HU_M2, "HU_M2", 0.2921768785246152},
	{Nyxus::Feature2D::HU_M3, "HU_M3", 0.06341348298776381},
	{Nyxus::Feature2D::HU_M4, "HU_M4", 0.05042335332144146},
	{Nyxus::Feature2D::HU_M5, "HU_M5", 0.002851264767850148},	// a 3*(eta30+eta12)^2 -> (3*(eta30+eta12))^2 slip gives 0.0032935269006466807, 442x the tolerance
	{Nyxus::Feature2D::HU_M6, "HU_M6", 0.027252861297278195},	// dropping the (eta21+eta03) grouping gives 0.02916606310377036, 1913x the tolerance
	{Nyxus::Feature2D::HU_M7, "HU_M7", 5.616461607732461e-06},
};

// Vets Hu h5/h6 (and the full moment chain) against scikit-image on an asymmetric fixture, where
// the odd-order etas are large enough that a mis-bracketed h5 or an h6 missing the (eta21+eta03)
// grouping misses by hundreds of times the tolerance. The symmetric rectangle cannot: its odd etas
// vanish, so both formulas agree there.
void test_2d_moments_hu_wedge_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_wedge_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, moments_2d_skimage_wedge_hu_ref_vals, "SKIMAGE_ORACLE__");
}

void test_2d_moments_shape_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, moments_2d_skimage_shape_ref_vals, "SKIMAGE_ORACLE__");
}

void test_2d_moments_intensity_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, moments_2d_skimage_intensity_ref_vals, "SKIMAGE_ORACLE__");
}

// Normalized RAW moments: NORM_SPAT_MOMENT_pq = m_pq / m_00^((p+q)/2 + 1), and IMOM_NRM_pq the
// same over the intensity-weighted raw moments. skimage has no native function for this - its
// moments_normalized() applies that exponent to the CENTRAL moments, which is a different
// quantity Nyxus exposes separately as NORM_CENTRAL_MOMENT_pq / IMOM_NCM_pq. So the oracle here
// is skimage's raw moments carried through skimage's own normalization exponent; the generator
// asserts the two are distinct so a future edit cannot quietly swap one for the other.
// Goldens: tests/vetting/oracles/gen_moments_skimage.py section D.
static const ref_vals_list<GeomomentGoldenValue> moments_2d_skimage_normraw_shape_ref_vals{
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_00, "NORM_SPAT_MOMENT_00", 1.0},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_01, "NORM_SPAT_MOMENT_01", 0.4450245779729475},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_02, "NORM_SPAT_MOMENT_02", 0.2674479166666667},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_03, "NORM_SPAT_MOMENT_03", 0.1807912348015099},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_10, "NORM_SPAT_MOMENT_10", 0.5363116708904752},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_11, "NORM_SPAT_MOMENT_11", 0.238671875},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_12, "NORM_SPAT_MOMENT_12", 0.14343543906367653},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_13, "NORM_SPAT_MOMENT_13", 0.09696044921875},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_20, "NORM_SPAT_MOMENT_20", 0.3875868055555556},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_21, "NORM_SPAT_MOMENT_21", 0.17248565457024395},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_22, "NORM_SPAT_MOMENT_22", 0.10365928367332176},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_23, "NORM_SPAT_MOMENT_23", 0.07007229716916162},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_30, "NORM_SPAT_MOMENT_30", 0.31508310664815414},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_31, "NORM_SPAT_MOMENT_31", 0.1402197265625},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_32, "NORM_SPAT_MOMENT_32", 0.08426832044990999},
	{Nyxus::Feature2D::NORM_SPAT_MOMENT_33, "NORM_SPAT_MOMENT_33", 0.056964263916015626},
};

static const ref_vals_list<GeomomentGoldenValue> moments_2d_skimage_normraw_intensity_ref_vals{
	{Nyxus::Feature2D::IMOM_NRM_00, "IMOM_NRM_00", 1.0},
	{Nyxus::Feature2D::IMOM_NRM_01, "IMOM_NRM_01", 0.039397166363954135},
	{Nyxus::Feature2D::IMOM_NRM_02, "IMOM_NRM_02", 0.001897046009773198},
	{Nyxus::Feature2D::IMOM_NRM_03, "IMOM_NRM_03", 9.951746245055554e-05},
	{Nyxus::Feature2D::IMOM_NRM_10, "IMOM_NRM_10", 0.04534163016617505},
	{Nyxus::Feature2D::IMOM_NRM_11, "IMOM_NRM_11", 0.0017522720110346945},
	{Nyxus::Feature2D::IMOM_NRM_12, "IMOM_NRM_12", 8.375967849944927e-05},
	{Nyxus::Feature2D::IMOM_NRM_13, "IMOM_NRM_13", 4.3766925283775395e-06},
	{Nyxus::Feature2D::IMOM_NRM_20, "IMOM_NRM_20", 0.002579984204506706},
	{Nyxus::Feature2D::IMOM_NRM_21, "IMOM_NRM_21", 9.892567767206171e-05},
	{Nyxus::Feature2D::IMOM_NRM_22, "IMOM_NRM_22", 4.7143264687233126e-06},
	{Nyxus::Feature2D::IMOM_NRM_23, "IMOM_NRM_23", 2.459309328437938e-07},
	{Nyxus::Feature2D::IMOM_NRM_30, "IMOM_NRM_30", 0.0001612294112519097},
	{Nyxus::Feature2D::IMOM_NRM_31, "IMOM_NRM_31", 6.155292790140729e-06},
	{Nyxus::Feature2D::IMOM_NRM_32, "IMOM_NRM_32", 2.928338360295916e-07},
	{Nyxus::Feature2D::IMOM_NRM_33, "IMOM_NRM_33", 1.5262053460086997e-08},
};

void test_2d_moments_normraw_shape_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, moments_2d_skimage_normraw_shape_ref_vals, "SKIMAGE_ORACLE__");
}

void test_2d_moments_normraw_intensity_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_2d_geomoment_feature_values(fvals);
	assert_2d_geomoment_features(fvals, moments_2d_skimage_normraw_intensity_ref_vals, "SKIMAGE_ORACLE__");
}
