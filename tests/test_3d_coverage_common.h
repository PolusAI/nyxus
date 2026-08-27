#pragma once

#include <initializer_list>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../src/nyx/featureset.h"
#include "../src/nyx/features/3d_gldm.h"
#include "../src/nyx/features/3d_gldzm.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/features/3d_glszm.h"
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/features/3d_ngldm.h"
#include "../src/nyx/features/3d_ngtdm.h"
#include "../src/nyx/features/3d_surface.h"

#include "test_ref_vals.h"

// The legacy oracle-backed coverage sweep (a generic TEST_P over every 3D family's featureset, with a
// bypass-FeatureManager/hand-construct-the-calculator harness) has now been retired family by family:
// every one of the 9 3D families either has direct named oracle tests for its oracle-backed features
// (glcm, gldm, glrlm, glszm, ngtdm, morphology, firstorder) or was never in the sweep to begin with.
// With no family left to instantiate it, its TEST_P fixture and phantom-running harness were deleted
// rather than kept idle. What remains is the classification data those retired tests read their pins
// from (which family a feature belongs to, which features an oracle backs, which features a named
// test individually pins) and the global completeness guard below.
namespace
{
struct Feature3DCoverageCase
{
	std::string name;
	Nyxus::Feature3D code;
};

// Which features an external reference actually backs. Derived wholly from the reference tables in the
// per-family oracle headers -- it holds no values of its own, so it is an index rather than a
// reference table, so a golden can only be declared in the oracle header that owns it.
static const std::set<std::string>& externally_vetted_3d_feature_names()
{
	static const std::set<std::string> names = [] {
		std::set<std::string> out;
		auto add_keys = [&out](const auto& m) {
			for (const auto& kv : m)
				out.insert(kv.first);
		};
		add_keys(firstorder_3d_pyradiomics_ref_vals);
		add_keys(firstorder_3d_matlab_ref_vals);
		add_keys(glcm_3d_pyradiomics_ref_vals);
		add_keys(gldm_3d_pyradiomics_ref_vals);
		add_keys(glrlm_3d_pyradiomics_ref_vals);
		add_keys(glszm_3d_pyradiomics_ref_vals);
		add_keys(ngtdm_3d_pyradiomics_ref_vals);
		add_keys(morphology_3d_mirp_volume_ref_vals);
		return out;
	}();
	return names;
}

static std::vector<Feature3DCoverageCase> feature_3d_cases(bool require_oracle_backed)
{
	std::vector<Feature3DCoverageCase> out;
	const auto& embedded = externally_vetted_3d_feature_names();
	for (const auto& kv : Nyxus::UserFacing_3D_featureNames)
	{
		const bool is_oracle_backed = embedded.find(kv.first) != embedded.end();
		if (is_oracle_backed == require_oracle_backed)
			out.push_back({ kv.first, kv.second });
	}
	return out;
}

// Which calculator featureset each public 3D feature belongs to. Each feature belongs to exactly one
// family (first match wins), used below to name an uncovered feature's family in the completeness guard.
static const std::vector<std::pair<std::string, std::set<Nyxus::Feature3D>>>& feature_3d_family_table()
{
	static const std::vector<std::pair<std::string, std::set<Nyxus::Feature3D>>> table = [] {
		auto mk = [](std::initializer_list<Nyxus::Feature3D> fs) {
			return std::set<Nyxus::Feature3D>(fs.begin(), fs.end());
		};
		std::vector<std::pair<std::string, std::set<Nyxus::Feature3D>>> t;
		t.emplace_back("glcm", mk(D3_GLCM_feature::featureset));
		t.emplace_back("gldm", mk(D3_GLDM_feature::featureset));
		t.emplace_back("gldzm", mk(D3_GLDZM_feature::featureset));
		t.emplace_back("glrlm", mk(D3_GLRLM_feature::featureset));
		t.emplace_back("glszm", mk(D3_GLSZM_feature::featureset));
		t.emplace_back("ngldm", mk(D3_NGLDM_feature::featureset));
		t.emplace_back("ngtdm", mk(D3_NGTDM_feature::featureset));
		t.emplace_back("morphology", mk(D3_SurfaceFeature::featureset));
		t.emplace_back("firstorder", mk(D3_VoxelIntensityFeatures::featureset));
		return t;
	}();
	return table;
}

static std::string family_of_3d_feature(Nyxus::Feature3D code)
{
	for (const auto& fam : feature_3d_family_table())
		if (fam.second.count(code))
			return fam.first;
	return "unknown";
}

// Features pinned by individually named tests -- glcm's and glrlm's "_grey64_regression" tests;
// morphology's "_regression" ones, plus its five PCA axis features, which a MIRP oracle test pins at
// rel=1e-9 and so needs no snapshot of its own; gldzm's and ngldm's "_regression" ones, each of which
// is that whole family; firstorder's "_regression" ones. Read straight off the tables those tests
// assert against, the same way externally_vetted_3d_feature_names() is read off the oracle tables, so
// migrating the next family is one add_keys() line and nothing has to be counted or opted out by hand.
static const std::set<std::string>& individually_pinned_3d_feature_names()
{
	static const std::set<std::string> names = [] {
		std::set<std::string> out;
		auto add_keys = [&out](const auto& m) {
			for (const auto& kv : m)
				out.insert(kv.first);
		};
		add_keys(glcm_3d_regression_grey64_ref_vals);
		add_keys(gldzm_3d_regression_ref_vals);
		add_keys(glrlm_3d_regression_grey64_ref_vals);
		add_keys(firstorder_3d_regression_ref_vals);
		add_keys(morphology_3d_regression_ref_vals);
		add_keys(morphology_3d_mirp_pca_ref_vals);
		add_keys(ngldm_3d_regression_ref_vals);
		return out;
	}();
	return names;
}

TEST(TEST_NYXUS, TEST_3D_FEATURE_COVERAGE_COUNTS)
{
	EXPECT_EQ(213u, Nyxus::UserFacing_3D_featureNames.size());
	EXPECT_EQ(110u, feature_3d_cases(true).size());
	EXPECT_EQ(103u, feature_3d_cases(false).size());
	EXPECT_EQ(Nyxus::UserFacing_3D_featureNames.size(), feature_3d_cases(true).size() + feature_3d_cases(false).size());

	// SPEC 1: every public feature with no oracle behind it still has to have a named regression pin.
	for (const auto& c : feature_3d_cases(false))
		EXPECT_TRUE(individually_pinned_3d_feature_names().count(c.name) > 0)
			<< c.name << " (family " << family_of_3d_feature(c.code)
			<< ") has no oracle and no named regression pin";
}
}
