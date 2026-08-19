#pragma once

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <tuple>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_ngldm.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"   // ref_vals_map, and the <string> / <vector> it already includes

// Drift guards on the segmented phantom (ut_inten.nii + ut_mask57.nii, label 57) at 64 grey levels,
// ibsi=false. Nyxus' own output, so these claim no oracle (SPEC 1) -- and unlike every other 3D
// family, no feature here is vetted against one either.
//
// Regenerate with test_3d_ngldm_dump_regression() below.
//
// THESE VALUES ARE KNOWN TO BE WRONG, and are pinned anyway. A config-matched MIRP 2.6.0 run
// disagrees on 16 of the 17 features it can compute, by up to 50x, and two causes are visible in
// src/nyx/features/3d_ngldm.cpp: the NGLD matrix is built over the ROI bounding box rather than the
// ROI (background voxels included), and the neighbourhood table holds 24 shifts where a 3D
// Chebyshev-1 neighbourhood has 26. A drift guard's job is to notice change, not to bless the
// values, so they are pinned at full precision until the implementation is fixed -- at which point
// every number below has to be regenerated.
//
// Measurements, both causes, and the reproduction:
// tests/vetting/audit/ngldm_3d_mirp_vetting_report.md.
//
// 3NGLDM_GLM and 3NGLDM_DCM have no counterpart in any tool -- MIRP's NGLDM emits no gl_mean /
// dc_mean column -- so they cannot be vetted even once the implementation is corrected.
static ref_vals_map<double> ngldm_3d_regression_ref_vals{
		{ "3NGLDM_LDE",	0.10159976999534079 },
		{ "3NGLDM_HDE",	261.01822590738425 },
		{ "3NGLDM_LGLCE",	0.00035968375469422158 },
		{ "3NGLDM_HGLCE",	740.43602941176471 },
		{ "3NGLDM_LDLGLE",	5.8337460459982142e-05 },
		{ "3NGLDM_LDHGLE",	73.919882197712482 },
		{ "3NGLDM_HDLGLE",	0.025201544837470152 },
		{ "3NGLDM_HDHGLE",	20099.770197121401 },
		{ "3NGLDM_GLNU",	115443.18172715895 },
		{ "3NGLDM_GLNUN",	0.22575716076180957 },
		{ "3NGLDM_DCNU",	85056.840050062572 },
		{ "3NGLDM_DCNUN",	0.16633455892143026 },
		{ "3NGLDM_DCP",	1.0 },
		{ "3NGLDM_GLM",	16.955115769712151 },
		{ "3NGLDM_GLV",	190.08150972702501 },
		{ "3NGLDM_DCM",	13.485998122653307 },
		{ "3NGLDM_DCV",	86.17064428912758 },
		{ "3NGLDM_DCENT",	5.2277449211654039 },
		{ "3NGLDM_DCENE",	0.14348407632898436 }
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void assert_3d_ngldm_feature_regression (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
	// check that requested feature exists -- operator[] below would otherwise default-insert a 0
	// golden and compare against it
	auto iter = ngldm_3d_regression_ref_vals.find(fname);
	ASSERT_TRUE(iter != ngldm_3d_regression_ref_vals.end());

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
	D3_NGLDM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));

	// (6) saving values

	f.save_value(r.fvals);

	// we have just 1 value, no need to aggregate subfeatures
	double atot = r.fvals[fcode][0];

	// verdict. frac_tolerance = 1e9, i.e. rel=1e-9: Nyxus' own values pinned to full precision, so
	// the guard catches any change at all. The previous 10% band sat on two- and three-significant-
	// figure pins, which cannot detect the implementation fix these values are waiting for.
	ASSERT_TRUE(agrees_gt(atot, ngldm_3d_regression_ref_vals[fname], 1e9));
}

// Regenerates every golden in ngldm_3d_regression_ref_vals at full precision, in the exact shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_NGLDM_DUMP_REGRESSION*
// and paste the output over the table above. It uses the same settings the shared assert helper
// sets, so the two cannot drift apart. This is the function to re-run once the two defects recorded
// in the audit report are fixed -- every pin above changes then.
void test_3d_ngldm_dump_regression()
{
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	Environment e;
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
	ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();

	clear_slide_rois(e.uniqueLabels, e.roiData);
	ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

	std::vector<int> batch = { label };
	ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
	ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

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

	LR& r = e.roiData[label];
	ASSERT_NO_THROW(r.initialize_fvals());
	D3_NGLDM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	std::cout << "[3DNGLDM-REGEN]\n";
	for (const auto& nv : ngldm_3d_regression_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DNGLDM-REGEN]\t\t{ \"" << nv.first << "\",\t"
		          << std::setprecision(17) << r.fvals[fcode][0] << " },\n";
	}
}

void test_3d_ngldm_lde_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_LDE", Feature3D::NGLDM_LDE);
}

void test_3d_ngldm_hde_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_HDE", Feature3D::NGLDM_HDE);
}

void test_3d_ngldm_lglce_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_LGLCE", Feature3D::NGLDM_LGLCE);
}

void test_3d_ngldm_hglce_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_HGLCE", Feature3D::NGLDM_HGLCE);
}

void test_3d_ngldm_ldlgle_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_LDLGLE", Feature3D::NGLDM_LDLGLE);
}

void test_3d_ngldm_ldhgle_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_LDHGLE", Feature3D::NGLDM_LDHGLE);
}

void test_3d_ngldm_hdlgle_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_HDLGLE", Feature3D::NGLDM_HDLGLE);
}

void test_3d_ngldm_hdhgle_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_HDHGLE", Feature3D::NGLDM_HDHGLE);
}

void test_3d_ngldm_glnu_regression() {
	assert_3d_ngldm_feature_regression("3NGLDM_GLNU", Feature3D::NGLDM_GLNU);
}

void test_3d_ngldm_glnun_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_GLNUN", Feature3D::NGLDM_GLNUN);
}

void test_3d_ngldm_dcnu_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCNU", Feature3D::NGLDM_DCNU);
}

void test_3d_ngldm_dcnun_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCNUN", Feature3D::NGLDM_DCNUN);
}

void test_3d_ngldm_dcp_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCP", Feature3D::NGLDM_DCP);
}

void test_3d_ngldm_glm_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_GLM", Feature3D::NGLDM_GLM);
}

void test_3d_ngldm_glv_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_GLV", Feature3D::NGLDM_GLV);
}

void test_3d_ngldm_dcm_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCM", Feature3D::NGLDM_DCM);
}

void test_3d_ngldm_dcv_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCV", Feature3D::NGLDM_DCV);
}

void test_3d_ngldm_dcent_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCENT", Feature3D::NGLDM_DCENT);
}

void test_3d_ngldm_dcene_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCENE", Feature3D::NGLDM_DCENE);
}



