#pragma once

#include "test_2d_ngldm_common.h"   // gtest, <string>, assert_ngldm_feature_against_golden_values
#include "test_ref_vals.h"          // ref_vals_map

// mirp goldens for the 2D NGLDM family (SPEC 6.4 provenance).
// tool=mirp 2.6.0; env=nyxus_mirp (conda); recipe=ngldm.ibsi_phantom_2d, i.e. the four IBSI
// digital-phantom slices from test_data.h featurised one at a time and averaged, with
// by_slice=True, base_discretisation_method="none" (the phantom is already discrete 1..6),
// ngldm_distance=1 and ngldm_difference_level=0 (the alpha=0, d=1 coarseness the IBSI NGLDM
// definition uses); generator=tests/vetting/oracles/gen_ngldm_mirp.py.
//
// The fixture -- the settings recipe and the four-slice averaging -- is shared through
// test_2d_ngldm_common.h, so this file borrows no scaffolding from the IBSI oracle file and the
// two tables stay out of each other's scope (SPEC 6.3.1).
//
// mirp and Nyxus implement the same definition over the same neighbourhood and agree to 2.9e-16
// worst case, so these are asserted at the SPEC 7 "exact" tier. test_2d_ngldm_ibsi.h pins the same
// quantities as published IBSI consensus values, which are quoted to three significant figures --
// that file therefore asserts at rel=1e-2 and this one at rel=1e-9. The two are complementary: IBSI
// fixes the definition, mirp fixes the digits.
//
// NGLDM_GLM and NGLDM_DCM are absent by design: mirp exposes no grey-level-mean or
// dependence-count-mean column because they are not IBSI NGLDM features. They stay regression rows
// in test_2d_ngldm_regression.h.
static const ref_vals_map<double> ngldm_2d_mirp_ref_vals
{
	{"NGLDM_LDE", 0.15807024738501638},
	{"NGLDM_HDE", 19.173821809425526},
	{"NGLDM_LGLCE", 0.7017531915300232},
	{"NGLDM_HGLCE", 7.486949604403165},
	{"NGLDM_LDLGLE", 0.047290498640367454},
	{"NGLDM_LDHGLE", 3.064914180133555},
	{"NGLDM_HDLGLE", 17.59968920804189},
	{"NGLDM_HDHGLE", 49.477721878224976},
	{"NGLDM_GLNU", 10.24637942896457},
	{"NGLDM_GLNUN", 0.5618604963062601},
	{"NGLDM_DCNU", 3.9646456828345373},
	{"NGLDM_DCNUN", 0.21177218060411693},
	{"NGLDM_GLV", 2.7037332451477987},
	{"NGLDM_DCP", 1.0},
	{"NGLDM_DCV", 2.729504577399913},
	{"NGLDM_DCENT", 2.7142924232815497},
	{"NGLDM_DCENE", 0.17025209750162384}
};

// frac_tolerance = 1e9, i.e. rel=1e-9 (agrees_gt divides the golden by this factor). The measured
// residual is 2.9e-16, so the bound is seven orders of magnitude above the noise and still tight
// enough that any change in the NGLDM definition or the neighbourhood walk fails it.
void assert_ngldm_feature_mirp(const Feature2D& feature_, const std::string& feature_name)
{
	assert_ngldm_feature_against_golden_values(
		feature_,
		feature_name,
		ngldm_2d_mirp_ref_vals,
		"MIRP_ORACLE__",
		1e9);
}

void test_2d_ngldm_lde_mirp()      { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_LDE, "NGLDM_LDE"); }
void test_2d_ngldm_hde_mirp()      { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_HDE, "NGLDM_HDE"); }
void test_2d_ngldm_lglce_mirp()    { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_LGLCE, "NGLDM_LGLCE"); }
void test_2d_ngldm_hglce_mirp()    { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_HGLCE, "NGLDM_HGLCE"); }
void test_2d_ngldm_ldlgle_mirp()   { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_LDLGLE, "NGLDM_LDLGLE"); }
void test_2d_ngldm_ldhgle_mirp()   { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_LDHGLE, "NGLDM_LDHGLE"); }
void test_2d_ngldm_hdlgle_mirp()   { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_HDLGLE, "NGLDM_HDLGLE"); }
void test_2d_ngldm_hdhgle_mirp()   { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_HDHGLE, "NGLDM_HDHGLE"); }
void test_2d_ngldm_glnu_mirp()     { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_GLNU, "NGLDM_GLNU"); }
void test_2d_ngldm_glnun_mirp()    { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_GLNUN, "NGLDM_GLNUN"); }
void test_2d_ngldm_dcnu_mirp()     { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_DCNU, "NGLDM_DCNU"); }
void test_2d_ngldm_dcnun_mirp()    { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_DCNUN, "NGLDM_DCNUN"); }
void test_2d_ngldm_glv_mirp()      { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_GLV, "NGLDM_GLV"); }
void test_2d_ngldm_dcp_mirp()      { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_DCP, "NGLDM_DCP"); }
void test_2d_ngldm_dcv_mirp()      { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_DCV, "NGLDM_DCV"); }
void test_2d_ngldm_dcent_mirp()    { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_DCENT, "NGLDM_DCENT"); }
void test_2d_ngldm_dcene_mirp()    { assert_ngldm_feature_mirp(Nyxus::Feature2D::NGLDM_DCENE, "NGLDM_DCENE"); }

// ---------------------------------------------------------------------------------------------------
// The negative coarse grey depth, recipe ngldm.mirp_fbn20_negdepth: GREYDEPTH = -20 bins the ROI into
// 20 PyRadiomics-style bins over its own [min, max], which is IBSI's fixed bin number, and MIRP is run
// with base_discretisation_method="fixed_bin_number", n_bins=20 -- each tool discretising for itself,
// so the binning is part of what is compared. Fixture bench_square12_graded (test_main_nyxus.h), with
// five of its 144 pixels exactly on a bin edge. NGLDM_DCP is left out: both tools return 1 whenever
// every pixel has a same-level neighbour.
//
// Provenance (SPEC 6.4): tool=mirp 2.6.0, by_slice=True, distance 1, difference level 0;
// generator=tests/vetting/oracles/gen_negative_greydepth_mirp.py (re-verifies every pin with --check).
// Measured residual 2.1e-16 worst case; asserted at rel=1e-9, and at rel=1e-6 for NGLDM_DCENT, the
// family's one logarithm, whose last bits move between platforms.
static const ref_vals_map<double> ngldm_2d_mirp_negdepth_ref_vals{
		{ "NGLDM_LDE",	0.8128858024691358 },		// ngl_lde_d1_a0.0_2d_fbn_n20
		{ "NGLDM_HDE",	1.9027777777777777 },		// ngl_hde_d1_a0.0_2d_fbn_n20
		{ "NGLDM_LGLCE",	0.08136848819766401 },		// ngl_lgce_d1_a0.0_2d_fbn_n20
		{ "NGLDM_HGLCE",	142.54861111111111 },		// ngl_hgce_d1_a0.0_2d_fbn_n20
		{ "NGLDM_LDLGLE",	0.059117449770570624 },		// ngl_ldlge_d1_a0.0_2d_fbn_n20
		{ "NGLDM_LDHGLE",	120.47762345679013 },		// ngl_ldhge_d1_a0.0_2d_fbn_n20
		{ "NGLDM_HDLGLE",	0.20685947399060212 },		// ngl_hdlge_d1_a0.0_2d_fbn_n20
		{ "NGLDM_HDHGLE",	246.60416666666666 },		// ngl_hdhge_d1_a0.0_2d_fbn_n20
		{ "NGLDM_GLNU",	7.819444444444445 },		// ngl_glnu_d1_a0.0_2d_fbn_n20
		{ "NGLDM_GLNUN",	0.054301697530864196 },		// ngl_glnu_norm_d1_a0.0_2d_fbn_n20
		{ "NGLDM_DCNU",	88.93055555555556 },		// ngl_dcnu_d1_a0.0_2d_fbn_n20
		{ "NGLDM_DCNUN",	0.6175733024691358 },		// ngl_dcnu_norm_d1_a0.0_2d_fbn_n20
		{ "NGLDM_GLV",	34.475260416666664 },		// ngl_gl_var_d1_a0.0_2d_fbn_n20
		{ "NGLDM_DCV",	0.27006172839506165 },		// ngl_dc_var_d1_a0.0_2d_fbn_n20
		{ "NGLDM_DCENT",	4.9476106370788075 },		// ngl_dc_entr_d1_a0.0_2d_fbn_n20
		{ "NGLDM_DCENE",	0.03607253086419753 }		// ngl_dc_energy_d1_a0.0_2d_fbn_n20
};

void test_2d_ngldm_negative_greydepth_mirp()
{
	std::vector<NyxusPixel> intensity, mask;
	Nyxus::make_2d_square12_graded_phantom (intensity, mask);

	Fsettings s = make_ngldm2d_settings (false);
	s[(int)NyxSetting::GREYDEPTH].ival = -20;

	LR roidata;
	Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
	NGLDMfeature f;
	f.calculate (roidata, s);
	roidata.initialize_fvals();
	f.save_value (roidata.fvals);

	for (const auto& [name, golden] : ngldm_2d_mirp_negdepth_ref_vals)
	{
		SCOPED_TRACE ("MIRP_ORACLE__" + name);
		const double frac = name == "NGLDM_DCENT" ? 1e6 : 1e9;
		ASSERT_TRUE (agrees_gt (roidata.fvals[(int)Nyxus::UserFacingFeatureNames.at(name)][0], golden, frac)) << name;
	}
}