#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_gldzm.h"

static std::unordered_map<std::string, float> d3gldzm_GT{
	{"3GLDZM_SDE",		0.0224},
	{"3GLDZM_LDE",       314.0},
	{"3GLDZM_LGLZE",     0.0006},
	{"3GLDZM_HGLZE",     2342.5},
	{"3GLDZM_SDLGLE",		0.000018},
	{"3GLDZM_SDHGLE",    61.2},
	{"3GLDZM_LDLGLE",    0.17},
	{"3GLDZM_LDHGLE",    734618.0},
	{"3GLDZM_GLNU",      3435.2},
	{"3GLDZM_GLNUN",     0.027},
	{"3GLDZM_ZDNU",      4330.3},
	{"3GLDZM_ZDNUN",     0.034},
	{"3GLDZM_ZP",        0.47},
	{"3GLDZM_GLM",       47.2},
	{"3GLDZM_GLV",       112.0},
	{"3GLDZM_ZDM",       222},
	{"3GLDZM_ZDV",       79.7},
	{"3GLDZM_ZDE",       10.23}
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3gldzm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
#if 0
	// get segment info
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	// mock the 3D workflow
	Environment e;
	clear_slide_rois (e.uniqueLabels, e.roiData);
	ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
	std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
	ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
	ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

	// make it find the feature code by name
	int fcode = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	// ... and that it's the feature we expect
	ASSERT_TRUE((int)expecting_fcode == fcode);

	// set feature's state
	Environment::ibsi_compliance = false;

	// extract the feature
	LR& r = e.roiData[label];
	ASSERT_NO_THROW(r.initialize_fvals());
	D3_GLDZM_feature f;
	Fsettings s;
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	// aggregate all the angles
	double atot = r.fvals[fcode][0];

	// verdict
	ASSERT_TRUE(agrees_gt(atot, d3gldzm_GT[fname], 10.));
#endif

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
	ASSERT_TRUE(agrees_gt(atot, d3gldzm_GT[fname], 10.));

}

void test_3GLDZM_SDE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_SDE, "3GLDZM_SDE");
}

void test_3GLDZM_LDE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_LDE, "3GLDZM_LDE");
}

void test_3GLDZM_LGLZE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_LGLZE, "3GLDZM_LGLZE");
}

void test_3GLDZM_HGLZE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_HGLZE, "3GLDZM_HGLZE");
}

void test_3GLDZM_SDLGLE() {
	test_3gldzm_feature(Nyxus::Feature3D::GLDZM_SDLGLE, "3GLDZM_SDLGLE");
}

void test_3GLDZM_SDHGLE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_SDHGLE, "3GLDZM_SDHGLE");
}

void test_3GLDZM_LDLGLE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_LDLGLE, "3GLDZM_LDLGLE");
}

void test_3GLDZM_LDHGLE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_LDHGLE, "3GLDZM_LDHGLE");
}

void test_3GLDZM_GLNU() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_GLNU, "3GLDZM_GLNU");
}

void test_3GLDZM_GLNUN() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_GLNUN, "3GLDZM_GLNUN");
}

void test_3GLDZM_ZDNU() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_ZDNU, "3GLDZM_ZDNU");
}

void test_3GLDZM_ZDNUN() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_ZDNUN, "3GLDZM_ZDNUN");
}

void test_3GLDZM_ZP() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_ZP, "3GLDZM_ZP");
}

void test_3GLDZM_GLM() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_GLM, "3GLDZM_GLM");
}

void test_3GLDZM_GLV() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_GLV, "3GLDZM_GLV");
}

void test_3GLDZM_ZDV() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_ZDV, "3GLDZM_ZDV");
}

void test_3GLDZM_ZDE() {
	test_3gldzm_feature (Nyxus::Feature3D::GLDZM_ZDE, "3GLDZM_ZDE");
}

