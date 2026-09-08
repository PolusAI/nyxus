#pragma once

// 3D GLDZM drift guards on bench_ut57_3d, the family's default-mode configuration.
//
// These are NOT oracle values: they are pinned Nyxus output on tests/data/nifti/phantoms/ut_inten.nii
// + ut_mask57.nii, label 57, at GREYDEPTH=64 and IBSI=false, and passing them establishes nothing
// (SPEC 1). Recorded at full %.17g precision so the guard detects movement rather than absorbing it.
//
// The family IS vetted, at a different configuration: test_3d_gldzm_mirp.h pins MIRP on the
// compatibility phantom at IBSI=true, where the two tools read the same grey levels and agree to an
// absolute 5.7e-15. This configuration cannot be compared with MIRP because the two sides do not
// discretise alike -- MIRP's fixed_bin_number spreads this ROI over levels 1-64 and Nyxus' GREYDEPTH
// binning lands it on 22-64, 43 of them distinct. That gap is a property of the binning, not of the
// GLDZM, and it is the same one every Nyxus texture family meets on a CT-like fixture;
// tests/vetting/audit/gldzm_3d_mirp_vetting_report.md sizes it on this phantom.
//
// 3GLDZM_GLM and 3GLDZM_ZDM are pinned here and nowhere else: MIRP emits no dzm_gl_mean or
// dzm_zd_mean column and IBSI defines neither, so no oracle reaches them at any configuration.
//
// Regenerate the table with test_3d_gldzm_dump_regression() below.
//
// This file holds the family's default-mode pin table, and test_3d_coverage_common.h reads its keys
// to satisfy SPEC 1. Hence const: a default-insert here would both pass a bogus assertion against a 0
// golden and add a phantom feature name to that set.

// Only what the fixture header does not already supply: <iomanip> for the precision the failure
// message and the regeneration dump print at. gtest, <iostream>, <string>, <vector> and the
// Environment / roi_cache graph arrive through it.
#include <iomanip>

#include "test_3d_gldzm_common.h"   // bench_ut57_3d, make_gldzm3d_settings, extract_3d_gldzm, agrees_gt
#include "test_ref_vals.h"          // ref_vals_map

static const ref_vals_map<double> gldzm_3d_regression_ref_vals{
	{"3GLDZM_SDE",        0.47006537949579702},
	{"3GLDZM_LDE",         7.4337854742574017},
	{"3GLDZM_LGLZE",   0.00043490836742224412},
	{"3GLDZM_HGLZE",       2685.0693909588167},
	{"3GLDZM_SDLGLE", 0.00015407786939386234},
	{"3GLDZM_SDHGLE",      1540.7841395501789},
	{"3GLDZM_LDLGLE",  0.0045802962960003408},
	{"3GLDZM_LDHGLE",      14203.744217420099},
	{"3GLDZM_GLNU",        1349.3969192278446},
	{"3GLDZM_GLNUN",     0.033098602350507607},
	{"3GLDZM_ZDNU",        10424.140327209399},
	{"3GLDZM_ZDNUN",      0.25568790814612574},
	{"3GLDZM_ZP",         0.14855774836753732},
	{"3GLDZM_GLM",         50.994186759547695},
	{"3GLDZM_GLV",         84.66230769118728 },
	{"3GLDZM_ZDM",         2.3107753440113812},
	{"3GLDZM_ZDV",         2.0941027837664841},
	{"3GLDZM_ZDE",         6.4697858656991167},
};
// rel=1e-9. A drift guard compares the program against its own recorded output, so the only thing
// it can catch is movement, and the band should be as tight as the value is reproducible.
// agrees_gt divides the golden by this, so a larger argument is a tighter band.
static const double gldzm_3d_regression_frac_tolerance = 1.e9;

void assert_3d_gldzm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	// the table is const and read through .at(), so a missing key throws rather than being
	// default-inserted as a 0 golden and compared against; check it up front to fail by name
	ASSERT_TRUE(gldzm_3d_regression_ref_vals.count(fname) > 0) << fname;

	auto [ipath, mpath, label] = get_3d_segmented_phantom();

	// make it find the feature code by name ... and that it's the feature we expect
	Environment e;
	int fcode = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	ASSERT_TRUE((int)expecting_fcode == fcode);

	std::vector<std::vector<double>> fvals;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(fvals, ipath, mpath, label, make_gldzm3d_settings(64, false)));

	ASSERT_TRUE(agrees_gt(fvals[fcode][0], gldzm_3d_regression_ref_vals.at(fname), gldzm_3d_regression_frac_tolerance))
		<< fname << " actual=" << std::setprecision(17) << fvals[fcode][0];
}

// Regenerates every golden in gldzm_3d_regression_ref_vals at full precision, in the exact shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_GLDZM_DUMP_REGRESSION*
// and paste the output over the table above. It goes through the same extract_3d_gldzm helper and
// the same settings the assert helper uses, so the two cannot drift apart.
void test_3d_gldzm_dump_regression()
{
	auto [ipath, mpath, label] = get_3d_segmented_phantom();

	Environment e;
	std::vector<std::vector<double>> fvals;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(fvals, ipath, mpath, label, make_gldzm3d_settings(64, false)));

	std::cout << "[3DGLDZM-REGEN]\n";
	for (const auto& nv : gldzm_3d_regression_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DGLDZM-REGEN]\t{\"" << nv.first << "\",\t"
		          << std::setprecision(17) << fvals[fcode][0] << "},\n";
	}
}

void test_3d_gldzm_sde_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_SDE, "3GLDZM_SDE");
}

void test_3d_gldzm_lde_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_LDE, "3GLDZM_LDE");
}

void test_3d_gldzm_lglze_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_LGLZE, "3GLDZM_LGLZE");
}

void test_3d_gldzm_hglze_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_HGLZE, "3GLDZM_HGLZE");
}

void test_3d_gldzm_sdlgle_regression() {
	assert_3d_gldzm_feature_regression(Nyxus::Feature3D::GLDZM_SDLGLE, "3GLDZM_SDLGLE");
}

void test_3d_gldzm_sdhgle_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_SDHGLE, "3GLDZM_SDHGLE");
}

void test_3d_gldzm_ldlgle_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_LDLGLE, "3GLDZM_LDLGLE");
}

void test_3d_gldzm_ldhgle_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_LDHGLE, "3GLDZM_LDHGLE");
}

void test_3d_gldzm_glnu_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_GLNU, "3GLDZM_GLNU");
}

void test_3d_gldzm_glnun_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_GLNUN, "3GLDZM_GLNUN");
}

void test_3d_gldzm_zdnu_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_ZDNU, "3GLDZM_ZDNU");
}

void test_3d_gldzm_zdnun_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_ZDNUN, "3GLDZM_ZDNUN");
}

void test_3d_gldzm_zp_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_ZP, "3GLDZM_ZP");
}

void test_3d_gldzm_glm_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_GLM, "3GLDZM_GLM");
}

void test_3d_gldzm_glv_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_GLV, "3GLDZM_GLV");
}

void test_3d_gldzm_zdm_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_ZDM, "3GLDZM_ZDM");
}

void test_3d_gldzm_zdv_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_ZDV, "3GLDZM_ZDV");
}

void test_3d_gldzm_zde_regression() {
	assert_3d_gldzm_feature_regression (Nyxus::Feature3D::GLDZM_ZDE, "3GLDZM_ZDE");
}

