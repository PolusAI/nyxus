#pragma once

// Drift guard for the 3D GLSZM family on the segmented phantom. Claims no oracle (SPEC 2): the
// values below are Nyxus' own output, so movement is the only thing they can catch.
//
// Recipe glszm3d.regression_ut_phantom: bench_ut_phantom_3d (label 57) with GREYDEPTH=64,
// IBSI=false and GLSZM_GREYDEPTH=64, whose positive sign selects MATLAB-style binning into that
// many levels. That is the recipe make_3d_coverage_settings() runs every 3D family at on this same
// phantom, so the sweep and this file describe one configuration.
//
// Pins are the program's own %.17g output -- a value truncated to five digits eats a third of a
// rel=1e-3 band before the test starts. Regenerate them with
//     runAllTests --gtest_filter=*3D_GLSZM_DUMP_REGRESSION*
// and see tests/vetting/audit/glszm_3d_golden_regen.md.

// Only what nothing this file already includes supplies: <iomanip> for the dump helper's
// setprecision. gtest, <iostream>, <string>, <vector>, the phantoms and the Environment graph all
// arrive through the common header.
#include <iomanip>

#include "test_3d_glszm_common.h"  // gtest, <iostream>, the phantoms, the settings recipe, extract_3d_glszm, agrees_gt
#include "test_ref_vals.h"         // ref_vals_map

static const ref_vals_map<double> glszm_3d_regression_ref_vals
{
	{"3GLSZM_SAE", 0.5641059480170818},
	{"3GLSZM_LAE", 15936.373617209154},
	{"3GLSZM_LGLZE", 0.00043490836742224412},
	{"3GLSZM_HGLZE", 2685.0693909588167},
	{"3GLSZM_SALGLE", 0.00023107613419972374},
	{"3GLSZM_SAHGLE", 1570.8808393460524},
	{"3GLSZM_LALGLE", 18.952399724529606},
	{"3GLSZM_LAHGLE", 16944465.274890237},
	{"3GLSZM_GLN", 1349.3969192278446},
	{"3GLSZM_GLNN", 0.033098602350507607},
	{"3GLSZM_SZN", 12492.8725011651},
	{"3GLSZM_SZNN", 0.30643068265508355},
	{"3GLSZM_ZP", 0.14855774836753732},
	{"3GLSZM_GLV", 84.662307691187365},
	{"3GLSZM_ZV", 15891.062018725946},
	{"3GLSZM_ZE", 7.344185889911989}
};

// agrees_gt divides the golden by this, so a larger argument is a tighter band. A snapshot of the
// program's own arithmetic on a fixed input reproduces exactly, so the band is the exact tier and
// anything looser would simply stop guarding.
static const double glszm_3d_regression_frac_tolerance = 1e9;

void assert_3d_glszm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	// a name with no pin is a failure, not a comparison against whatever a lookup would invent
	auto iter = glszm_3d_regression_ref_vals.find(fname);
	ASSERT_TRUE(iter != glszm_3d_regression_ref_vals.end());

	int fcode = -1;
	ASSERT_NO_FATAL_FAILURE(resolve_3d_glszm_fcode (fcode, expecting_fcode, fname));

	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	Fsettings s = make_glszm3d_settings (64/*greydepth*/, 64/*matlab-style binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, s));

	ASSERT_TRUE (agrees_gt (fvals[fcode][0], iter->second,
	                        glszm_3d_regression_frac_tolerance)) << fname;
}

// Regenerates every pin above at full precision, in the shape the table wants. Run it with
//     runAllTests --gtest_filter=*3D_GLSZM_DUMP_REGRESSION*
void test_3d_glszm_dump_regression()
{
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	Fsettings s = make_glszm3d_settings (64/*greydepth*/, 64/*matlab-style binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, s));

	Environment e;
	std::cout << "[3DGLSZM-REGR] cube " << cube.width() << "x" << cube.height() << "x" << cube.depth()
	          << ", intensity range [" << lo << ", " << hi << "]\n";
	for (const auto& nv : glszm_3d_regression_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DGLSZM-REGR]\t{\"" << nv.first << "\", "
		          << std::setprecision(17) << fvals[fcode][0] << "},\tpinned "
		          << std::setprecision(17) << nv.second << "\n";
	}
}

void test_3d_glszm_sae_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_SAE, "3GLSZM_SAE");
}

void test_3d_glszm_lae_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_LAE, "3GLSZM_LAE");
}

void test_3d_glszm_lglze_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_LGLZE, "3GLSZM_LGLZE");
}

void test_3d_glszm_hglze_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_HGLZE, "3GLSZM_HGLZE");
}

void test_3d_glszm_salgle_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_SALGLE, "3GLSZM_SALGLE");
}

void test_3d_glszm_sahgle_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_SAHGLE, "3GLSZM_SAHGLE");
}

void test_3d_glszm_lalgle_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_LALGLE, "3GLSZM_LALGLE");
}

void test_3d_glszm_lahgle_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_LAHGLE, "3GLSZM_LAHGLE");
}

void test_3d_glszm_gln_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_GLN, "3GLSZM_GLN");
}

void test_3d_glszm_glnn_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_GLNN, "3GLSZM_GLNN");
}

void test_3d_glszm_szn_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_SZN, "3GLSZM_SZN");
}

void test_3d_glszm_sznn_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_SZNN, "3GLSZM_SZNN");
}

void test_3d_glszm_zp_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_ZP, "3GLSZM_ZP");
}

void test_3d_glszm_glv_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_GLV, "3GLSZM_GLV");
}

void test_3d_glszm_zv_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_ZV, "3GLSZM_ZV");
}

void test_3d_glszm_ze_regression() {
	assert_3d_glszm_feature_regression (Nyxus::Feature3D::GLSZM_ZE, "3GLSZM_ZE");
}
