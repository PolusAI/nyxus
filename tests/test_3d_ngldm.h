#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_ngldm.h"

static std::unordered_map<std::string, float> d3ngldm_GT{
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

std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3ngldm_feature (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
	// get segment info
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	// mock the 3D workflow
	clear_slide_rois();
	ASSERT_TRUE(gatherRoisMetrics_3D(ipath, mpath));
	std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
	ASSERT_TRUE(scanTrivialRois_3D(batch, ipath, mpath));
	ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch));

	// make it find the feature code by name
	int fcode = -1;
	ASSERT_TRUE(theFeatureSet.find_3D_FeatureByString(fname, fcode));
	// ... and that it's the feature we expect
	ASSERT_TRUE((int)expecting_fcode == fcode);

	// set feature's state
	Environment::ibsi_compliance = false;

	// extract the feature
	LR& r = Nyxus::roiData[label];
	ASSERT_NO_THROW(r.initialize_fvals());
	D3_NGLDM_feature f;
	ASSERT_NO_THROW(f.calculate(r));
	f.save_value(r.fvals);

	// aggregate all the angles
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



