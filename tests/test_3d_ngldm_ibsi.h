#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_ngldm.h"

static std::unordered_map<std::string, double> d3ngldm_GT{
		{ "3NGLDM_LDE",	0.1 },
		{ "3NGLDM_HDE",	261.0 },
		{ "3NGLDM_LGLCE",	0.00036 },
		{ "3NGLDM_HGLCE",	740.0 },
		{ "3NGLDM_LDLGLE",	5.8e-05 },
		{ "3NGLDM_LDHGLE",	74.0 },
		{ "3NGLDM_HDLGLE",	0.025 },
		{ "3NGLDM_HDHGLE",	20030.0 },
		{ "3NGLDM_GLNU",	115443.0 },
		{ "3NGLDM_GLNUN",	0.23 },
		{ "3NGLDM_DCNU",	115443.0 },
		{ "3NGLDM_DCNUN",	0.23 },
		{ "3NGLDM_DCP",	1.0 },
		{ "3NGLDM_GLM",	17.0 },
		{ "3NGLDM_GLV",	190.0 },
		{ "3NGLDM_DCM",	13.5 },
		{ "3NGLDM_DCV",	86.17 },
		{ "3NGLDM_DCENT",	5.23 },
		{ "3NGLDM_DCENE",	0.14 }
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3ngldm_feature (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
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
	D3_NGLDM_feature f;
	Fsettings s;
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	// aggregate all the angles
	double atot = r.fvals[fcode][0];

	// verdict
	ASSERT_TRUE(agrees_gt(atot, d3ngldm_GT[fname], 10.));
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
	D3_NGLDM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));

	// (6) saving values

	f.save_value(r.fvals);

	// we have just 1 value, no need to aggregate subfeatures
	double atot = r.fvals[fcode][0];

	// verdict
	ASSERT_TRUE(agrees_gt(atot, d3ngldm_GT[fname], 10.));

}

void test_3ngldm_lde() {
	test_3ngldm_feature ("3NGLDM_LDE", Feature3D::NGLDM_LDE);
}

void test_3ngldm_hde() {
	test_3ngldm_feature ("3NGLDM_HDE", Feature3D::NGLDM_HDE);
}

void test_3ngldm_lglce() {
	test_3ngldm_feature ("3NGLDM_LGLCE", Feature3D::NGLDM_LGLCE);
}

void test_3ngldm_hglce() {
	test_3ngldm_feature ("3NGLDM_HGLCE", Feature3D::NGLDM_HGLCE);
}

void test_3ngldm_ldlgle() {
	test_3ngldm_feature ("3NGLDM_LDLGLE", Feature3D::NGLDM_LDLGLE);
}

void test_3ngldm_ldhgle() {
	test_3ngldm_feature ("3NGLDM_LDHGLE", Feature3D::NGLDM_LDHGLE);
}

void test_3ngldm_hdlgle() {
	test_3ngldm_feature ("3NGLDM_HDLGLE", Feature3D::NGLDM_HDLGLE);
}

void test_3ngldm_hdhgle() {
	test_3ngldm_feature ("3NGLDM_HDHGLE", Feature3D::NGLDM_HDHGLE);
}

void test_3ngldm_glnu() {
	test_3ngldm_feature("3NGLDM_GLNU", Feature3D::NGLDM_GLNU);
}

void test_3ngldm_glnun() {
	test_3ngldm_feature ("3NGLDM_GLNUN", Feature3D::NGLDM_GLNUN);
}

void test_3ngldm_dcnu() {
	test_3ngldm_feature ("3NGLDM_DCNU", Feature3D::NGLDM_DCNU);
}

void test_3ngldm_dcnun() {
	test_3ngldm_feature ("3NGLDM_DCNUN", Feature3D::NGLDM_DCNUN);
}

void test_3ngldm_dcp() {
	test_3ngldm_feature ("3NGLDM_DCP", Feature3D::NGLDM_DCP);
}

void test_3ngldm_glm() {
	test_3ngldm_feature ("3NGLDM_GLM", Feature3D::NGLDM_GLM);
}

void test_3ngldm_glv() {
	test_3ngldm_feature ("3NGLDM_GLV", Feature3D::NGLDM_GLV);
}

void test_3ngldm_dcm() {
	test_3ngldm_feature ("3NGLDM_DCM", Feature3D::NGLDM_DCM);
}

void test_3ngldm_dcv() {
	test_3ngldm_feature ("3NGLDM_DCV", Feature3D::NGLDM_DCV);
}

void test_3ngldm_dcent() {
	test_3ngldm_feature ("3NGLDM_DCENT", Feature3D::NGLDM_DCENT);
}

void test_3ngldm_dcene() {
	test_3ngldm_feature ("3NGLDM_DCENE", Feature3D::NGLDM_DCENE);
}



