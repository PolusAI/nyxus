#pragma once

#include <gtest/gtest.h>
#include <iomanip>
#include <tuple>
#include "../src/nyx/environment.h"           // Environment
#include "../src/nyx/feature_settings.h"      // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"            // Nyxus::Feature3D
#include "../src/nyx/globals.h"               // clear_slide_rois, gatherRoisMetrics_3D, scanTrivialRois_3D, allocateTrivialRoisBuffers_3D
#include "../src/nyx/roi_cache.h"             // LR
#include "../src/nyx/slideprops.h"            // SlideProps, scan_slide_props
#include "../src/nyx/features/3d_gldzm.h"     // D3_GLDZM_feature
#include "../src/nyx/helpers/fsystem.h"       // fs::exists
#include "test_main_nyxus.h"                  // agrees_gt
#include "test_ref_vals.h"                    // ref_vals_map, and the <string> / <vector> it already includes

// 3D GLDZM drift guards. These are NOT oracle values and this family is NOT vetted: every row in
// oracle_coverage.csv is status=regression, and passing these establishes nothing (SPEC 1).
//
// PROVENANCE: pinned Nyxus output on tests/data/nifti/phantoms/ut_inten.nii + ut_mask57.nii,
// label 57, at GREYDEPTH=64 and IBSI=false -- the settings the helper below applies. Recorded at
// full %.17g precision so the guard detects movement rather than absorbing it: a pin rounded to a
// few significant figures spends most of its band before the test starts.
//
// WHAT THESE VALUES ARE NOT. MIRP implements IBSI GLDZM (PyRadiomics has no GLDZM at all) and
// disagrees with every one of the 16 features it computes, several by more than an order of
// magnitude -- LDE 314.01 against 11.23, ZDV 79.72 against 3.25, LDHGLE 734618 against 10882. The
// disagreement is not a tolerance question: an independent implementation that grows zones with
// 26-connectivity and measures distance with a city-block distance transform reproduces MIRP to
// ratio 1.0000 on 14 of 16 features, so the definition MIRP computes is reachable and Nyxus is not
// computing it. tests/vetting/audit/gldzm_3d_mirp_vetting_report.md carries the measurements and
// the three defects behind them.
//
// So these pins are a change detector for the eventual fix, not an endorsement. Do not promote any
// row to status=vetted on the strength of them.
//
// 3GLDZM_GLM and 3GLDZM_ZDM have no counterpart in MIRP or IBSI at all (MIRP emits no dzm_gl_mean
// or dzm_zd_mean), so they stay regression-only whatever happens to the rest.
//
// The coverage sweep used to keep a second copy of this table; it is gone, and this file is now the
// family's only pin table -- test_3d_coverage_common.h reads its keys to satisfy SPEC 1. So the
// table is const: a default-insert here would both pass a bogus assertion and add a phantom feature
// name to that set.
static const ref_vals_map<double> gldzm_3d_regression_ref_vals{
	{"3GLDZM_SDE",        0.022387420258025731},
	{"3GLDZM_LDE",          314.01248309662088},
	{"3GLDZM_LGLZE",     0.0005581993242951194},
	{"3GLDZM_HGLZE",        2342.4734665801629},
	{"3GLDZM_SDLGLE",   1.8362515436029654e-05},
	{"3GLDZM_SDHGLE",       61.230746106573264},
	{"3GLDZM_LDLGLE",      0.16729167507144088},
	{"3GLDZM_LDHGLE",       734618.35720259824},
	{"3GLDZM_GLNU",         3435.1800942680934},
	{"3GLDZM_GLNUN",      0.026851399515903585},
	{"3GLDZM_ZDNU",         4330.2817177741472},
	{"3GLDZM_ZDNUN",      0.033848043255251946},
	{"3GLDZM_ZP",          0.46617376982276121},
	{"3GLDZM_GLM",          47.230300235279401},
	{"3GLDZM_GLV",          111.77220626552925},
	{"3GLDZM_ZDM",          15.306504185784746},
	{"3GLDZM_ZDV",          79.723412707174901},
	{"3GLDZM_ZDE",          10.230312642315166},
};
// rel=1e-9. A drift guard compares the program against its own recorded output, so the only thing
// it can catch is movement, and the band should be as tight as the value is reproducible.
// agrees_gt divides the golden by this, so a larger argument is a tighter band.
static const double gldzm_3d_regression_frac_tolerance = 1.e9;


static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void assert_3d_gldzm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	// the table is const and read through .at(), so a missing key throws rather than being
	// default-inserted as a 0 golden and compared against; check it up front to fail by name
	ASSERT_TRUE(gldzm_3d_regression_ref_vals.count(fname) > 0) << fname;

	// get segment info
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	// mock the 3D workflow
	Environment e;
	// (1) slide -> dataset -> prescan 
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
	ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();
	// (2) properties of specific ROIs sitting in 'e.uniqueLabels'
	clear_slide_rois(e.uniqueLabels, e.roiData);
	ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
	// (3) voxel clouds
	std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
	ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
	// (4) buffers
	ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

	// (5) feature settings
	Fsettings s;
	s.resize((int)NyxSetting::__COUNT__);
	s[(int)NyxSetting::SOFTNAN].rval = 0.0;
	s[(int)NyxSetting::TINY].rval = 0.0;
	s[(int)NyxSetting::SINGLEROI].bval = false;
	s[(int)NyxSetting::GREYDEPTH].ival = 64;
	s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
	s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
	s[(int)NyxSetting::USEGPU].bval = false;
	s[(int)NyxSetting::VERBOSLVL].ival = 0;
	s[(int)NyxSetting::IBSI].bval = false;
	//

	// (6) feature extraction

	// make it find the feature code by name
	int fcode = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	// ... and that it's the feature we expect
	ASSERT_TRUE((int)expecting_fcode == fcode);

	// extract the feature
	LR& r = e.roiData[label];
	ASSERT_NO_THROW(r.initialize_fvals());
	D3_GLDZM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));

	// (6) saving values

	f.save_value(r.fvals);

	// we have just 1 value, no need to aggregate subfeatures
	double atot = r.fvals[fcode][0];

	// verdict
	ASSERT_TRUE(agrees_gt(atot, gldzm_3d_regression_ref_vals.at(fname), gldzm_3d_regression_frac_tolerance))
		<< fname << " actual=" << std::setprecision(17) << atot;

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

