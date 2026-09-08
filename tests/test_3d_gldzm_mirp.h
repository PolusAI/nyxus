#pragma once

// 3D GLDZM vs MIRP 2.6.0 on the GLDZM compatibility phantom.
//
// Recipe gldzm3d.mirp_compat_phantom: bench_compat_gldzm_3d (label 57) at native 1x1x1 spacing;
// on the MIRP side by_slice=False and base_discretisation_method="none", on the Nyxus side IBSI=true,
// which switches the family's binning off. Both tools therefore read the phantom's voxel values as
// grey levels 1..8 directly, and the comparison measures the GLDZM rather than a discretisation.
//
// MIRP is the only mainstream oracle for this family: PyRadiomics implements no GLDZM at all.
//
// The phantom's 182 zones follow from how it is built (test_3d_gldzm_common.h): 189 bricks of one
// grey level each, less the seven that the corner-touching implant joins into one zone at
// 26-connectivity. 3GLDZM_ZP is that count over the ROI's 1512 voxels.
//
// 3GLDZM_GLM and 3GLDZM_ZDM are absent below because MIRP emits no dzm_gl_mean or dzm_zd_mean column
// and IBSI defines neither, so no oracle in the tool matrix reaches them; they are pinned as drift
// guards in test_3d_gldzm_regression.h instead.
//
// Goldens and their reproduction: tests/vetting/oracles/gen_gldzm3d_mirp.py, which re-verifies every
// pin below against a live MIRP run. Measurements, including what each of the phantom's three design
// choices is worth: tests/vetting/audit/gldzm_3d_mirp_vetting_report.md.

// Only what the fixture header does not already supply: <iomanip> for the failure message's
// precision. gtest, <string>, <vector> and the Environment / roi_cache graph arrive through it.
#include <iomanip>

#include "test_3d_gldzm_common.h"   // the compat phantom, make_gldzm3d_settings, extract_3d_gldzm, agrees_gt
#include "test_ref_vals.h"          // ref_vals_map

static const ref_vals_map<double> gldzm_3d_mirp_ref_vals{
	{"3GLDZM_SDE",       0.8238705738705739},     // dzm_sde_3d
	{"3GLDZM_LDE",       2.4615384615384617},     // dzm_lde_3d
	{"3GLDZM_LGLZE",     0.13946884110787175},    // dzm_lgze_3d
	{"3GLDZM_HGLZE",     27.53846153846154},      // dzm_hgze_3d
	{"3GLDZM_SDLGLE",    0.12001901217090417},    // dzm_sdlge_3d
	{"3GLDZM_SDHGLE",    23.23946886446887},      // dzm_sdhge_3d
	{"3GLDZM_LDLGLE",    0.31127085357952705},    // dzm_ldlge_3d
	{"3GLDZM_LDHGLE",    60.19230769230769},      // dzm_ldhge_3d
	{"3GLDZM_GLNU",      23.384615384615383},     // dzm_glnu_3d
	{"3GLDZM_GLNUN",     0.12848689771766694},    // dzm_glnu_norm_3d
	{"3GLDZM_ZDNU",      121.0},                  // dzm_zdnu_3d
	{"3GLDZM_ZDNUN",     0.6648351648351648},     // dzm_zdnu_norm_3d
	{"3GLDZM_ZP",        0.12037037037037036},    // dzm_z_perc_3d
	{"3GLDZM_GLV",       4.792899408284024},      // dzm_gl_var_3d
	{"3GLDZM_ZDV",       0.5746890472165197},     // dzm_zd_var_3d
	{"3GLDZM_ZDE",       3.7485223944249175},     // dzm_zd_entr_3d
};
// SPEC 7's exact tier: an absolute 1e-9 band. Thirteen of the sixteen values are bit-identical to
// MIRP's and the worst residual over the other three is an absolute 5.7e-15, on 3GLDZM_ZDE -- the
// only feature whose formula takes a logarithm, which Nyxus evaluates as log2(p + EPS). The band is
// five orders wider than that, so nothing about it is doing work a looser one would hide.
static const double gldzm_3d_mirp_abs_tolerance = 1.e-9;

void assert_3d_gldzm_feature_mirp (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	// the table is const and read through .at(), so a missing key throws rather than being
	// default-inserted as a 0 golden and compared against; check it up front to fail by name
	ASSERT_TRUE(gldzm_3d_mirp_ref_vals.count(fname) > 0) << fname;

	auto [ipath, mpath, label] = get_3d_compat_gldzm_phantom();

	// make it find the feature code by name ... and that it's the feature we expect
	Environment e;
	int fcode = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	ASSERT_TRUE((int)expecting_fcode == fcode);

	std::vector<std::vector<double>> fvals;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(fvals, ipath, mpath, label, make_gldzm3d_settings(64, true)));

	ASSERT_NEAR(fvals[fcode][0], gldzm_3d_mirp_ref_vals.at(fname), gldzm_3d_mirp_abs_tolerance)
		<< fname << " actual=" << std::setprecision(17) << fvals[fcode][0];
}

void test_3d_gldzm_sde_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_SDE, "3GLDZM_SDE");
}

void test_3d_gldzm_lde_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_LDE, "3GLDZM_LDE");
}

void test_3d_gldzm_lglze_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_LGLZE, "3GLDZM_LGLZE");
}

void test_3d_gldzm_hglze_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_HGLZE, "3GLDZM_HGLZE");
}

void test_3d_gldzm_sdlgle_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_SDLGLE, "3GLDZM_SDLGLE");
}

void test_3d_gldzm_sdhgle_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_SDHGLE, "3GLDZM_SDHGLE");
}

void test_3d_gldzm_ldlgle_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_LDLGLE, "3GLDZM_LDLGLE");
}

void test_3d_gldzm_ldhgle_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_LDHGLE, "3GLDZM_LDHGLE");
}

void test_3d_gldzm_glnu_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_GLNU, "3GLDZM_GLNU");
}

void test_3d_gldzm_glnun_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_GLNUN, "3GLDZM_GLNUN");
}

void test_3d_gldzm_zdnu_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_ZDNU, "3GLDZM_ZDNU");
}

void test_3d_gldzm_zdnun_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_ZDNUN, "3GLDZM_ZDNUN");
}

void test_3d_gldzm_zp_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_ZP, "3GLDZM_ZP");
}

void test_3d_gldzm_glv_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_GLV, "3GLDZM_GLV");
}

void test_3d_gldzm_zdv_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_ZDV, "3GLDZM_ZDV");
}

void test_3d_gldzm_zde_mirp() {
	assert_3d_gldzm_feature_mirp (Nyxus::Feature3D::GLDZM_ZDE, "3GLDZM_ZDE");
}
