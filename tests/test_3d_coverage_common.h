#pragma once

#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/roi_cache.h"
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

namespace
{
struct Feature3DCoverageCase
{
	std::string name;
	Nyxus::Feature3D code;
};

struct Computed3DFeatureValues
{
	std::vector<std::vector<double>> values;
	std::string setup_error;
};

static std::size_t setting_index(NyxSetting setting)
{
	return static_cast<std::size_t>(setting);
}

static int feature_code_value(Nyxus::Feature3D feature)
{
	return static_cast<int>(feature);
}

static std::size_t feature_index_from_int(int feature)
{
	return static_cast<std::size_t>(feature);
}

static std::size_t feature_code_index(Nyxus::Feature3D feature)
{
	return feature_index_from_int(feature_code_value(feature));
}

static std::string sanitize_3d_feature_test_name(const testing::TestParamInfo<Feature3DCoverageCase>& info)
{
	std::string s = "F_" + info.param.name;
	for (char& c : s)
		if (!std::isalnum(static_cast<unsigned char>(c)))
			c = '_';
	return s;
}

static Fsettings make_3d_coverage_settings()
{
	Fsettings s;
	s.resize(setting_index(NyxSetting::__COUNT__));
	s[setting_index(NyxSetting::SOFTNAN)].rval = 0.0;
	s[setting_index(NyxSetting::TINY)].rval = 0.0;
	s[setting_index(NyxSetting::SINGLEROI)].bval = false;
	s[setting_index(NyxSetting::GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::PIXELSIZEUM)].rval = 100;
	s[setting_index(NyxSetting::PIXELDISTANCE)].ival = 5;
	s[setting_index(NyxSetting::USEGPU)].bval = false;
	s[setting_index(NyxSetting::VERBOSLVL)].ival = 0;
	s[setting_index(NyxSetting::IBSI)].bval = false;
	s[setting_index(NyxSetting::GLCM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::GLCM_OFFSET)].ival = 1;
	s[setting_index(NyxSetting::GLCM_SPARSEINTENS)].bval = true;
	s[setting_index(NyxSetting::GLDM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::GLRLM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::GLSZM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::NGTDM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::NGTDM_RADIUS)].ival = 1;
	return s;
}

static std::tuple<std::string, std::string, int> get_3d_coverage_phantom()
{
	fs::path this_fpath(__FILE__);
	fs::path pp = this_fpath.parent_path();
	fs::path f1("/data/nifti/phantoms/ut_inten.nii");
	fs::path f2("/data/nifti/phantoms/ut_mask57.nii");
	return {
		(pp.string() + f1.make_preferred().string()),
		(pp.string() + f2.make_preferred().string()),
		57
	};
}

static void initialize_3d_feature_values_as_unwritten(std::vector<std::vector<double>>& values)
{
	values.resize(static_cast<std::size_t>(Nyxus::FeatureIMQ::_COUNT_));
	const double sentinel = std::numeric_limits<double>::quiet_NaN();
	for (auto& v : values)
		v.assign(1, sentinel);
}

static Computed3DFeatureValues build_computed_3d_feature_values()
{
	Computed3DFeatureValues out;
	initialize_3d_feature_values_as_unwritten(out.values);

	try
	{
		auto [ipath, mpath, label] = get_3d_coverage_phantom();
		if (!fs::exists(ipath) || !fs::exists(mpath))
		{
			std::ostringstream ss;
			ss << "missing 3D coverage phantom: " << ipath << " or " << mpath;
			out.setup_error = ss.str();
			return out;
		}

		Fsettings s = make_3d_coverage_settings();
		auto prepare_roi = [&](Environment& e) -> bool {
			e.dataset.dataset_props.reserve(1);
			SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
			if (!scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()))
			{
				out.setup_error = "scan_slide_props failed for 3D coverage phantom";
				return false;
			}
			e.dataset.update_dataset_props_extrema();

			clear_slide_rois(e.uniqueLabels, e.roiData);
			if (!gatherRoisMetrics_3D(e, 0, ipath, mpath, 0))
			{
				out.setup_error = "gatherRoisMetrics_3D failed for 3D coverage phantom";
				return false;
			}

			std::vector<int> batch = { label };
			if (!scanTrivialRois_3D(e, batch, ipath, mpath, 0))
			{
				out.setup_error = "scanTrivialRois_3D failed for 3D coverage phantom";
				return false;
			}
			allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache);
			return true;
		};

		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_VoxelIntensityFeatures f;
			f.calculate(r, s, e.dataset);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_SurfaceFeature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLCM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLDM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLDZM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLRLM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLSZM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_NGLDM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_NGTDM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
	}
	catch (const std::exception& e)
	{
		out.setup_error = e.what();
	}

	return out;
}

static const Computed3DFeatureValues& computed_3d_feature_values()
{
	static const Computed3DFeatureValues values = build_computed_3d_feature_values();
	return values;
}

static std::set<Nyxus::Feature3D> implemented_3d_feature_codes()
{
	std::set<Nyxus::Feature3D> out;
	auto add = [&out](std::initializer_list<Nyxus::Feature3D> features) {
		out.insert(features.begin(), features.end());
	};
	add(D3_VoxelIntensityFeatures::featureset);
	add(D3_SurfaceFeature::featureset);
	add(D3_GLCM_feature::featureset);
	add(D3_GLDM_feature::featureset);
	add(D3_GLDZM_feature::featureset);
	add(D3_GLRLM_feature::featureset);
	add(D3_GLSZM_feature::featureset);
	add(D3_NGLDM_feature::featureset);
	add(D3_NGTDM_feature::featureset);
	return out;
}

// The MIRP volume goldens and their measured bands live with the assertions that own them, in
// test_3d_morphology_mirp.h (SPEC 6.3.1) -- the same arrangement as the per-family
// *_3d_pyradiomics_ref_vals tables this header already reads from their own oracle files.

// Which features an external reference actually backs. Derived wholly from the reference tables below and in the
// per-family oracle headers -- it holds no values of its own, so it is an index rather than a
// reference table (it used to hard-code the three morphology keys, which made it a quiet third place
// where a golden could be declared).
static const std::set<std::string>& externally_vetted_3d_feature_names()
{
	static const std::set<std::string> names = [] {
		std::set<std::string> out;
		auto add_keys = [&out](const auto& m) {
			for (const auto& kv : m)
				out.insert(kv.first);
		};
		add_keys(firstorder_3d_pyradiomics_ref_vals);
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


// Coverage-sweep regression baselines. The tables themselves live in the per-family
// test_3d_<family>_coverage.h files, beside the suites that assert them (SPEC 6.3.1); this header
// only holds the registry they publish into. The indirection is forced by include order -- the
// per-family files include this one, so the shared TEST_P bodies cannot name tables declared later.
// Registration is a file-scope initialiser, so every table is in place before main() runs, while the
// tables are only read inside test bodies.
using CoverageBaselineTable = ref_vals_map<std::vector<double>>;
using CoverageBaselineRegistry = std::map<std::string, const CoverageBaselineTable*>;

static CoverageBaselineRegistry& coverage_baselines()
{
	static CoverageBaselineRegistry r;
	return r;
}

static bool register_coverage_baseline(const std::string& family, const CoverageBaselineTable* t)
{
	coverage_baselines()[family] = t;
	return true;
}

static const CoverageBaselineTable* regression_coverage_table_for_family(const std::string& family)
{
	auto it = coverage_baselines().find(family);
	return it == coverage_baselines().end() ? nullptr : it->second;
}

static std::size_t regression_coverage_ref_vals_total()
{
	std::size_t n = 0;
	for (const auto& kv : coverage_baselines())
		n += kv.second->size();
	return n;
}

static double relative_absdiff_pct(double actual, double expected)
{
	double denom = std::abs(expected);
	if (denom == 0.0)
		return actual == expected ? 0.0 : std::numeric_limits<double>::infinity();
	return 100.0 * std::abs(actual - expected) / denom;
}

static void assert_mirp_volume_shape_agreement(const Feature3DCoverageCase& c)
{
	auto it = morphology_3d_mirp_volume_ref_vals.find(c.name);
	ASSERT_TRUE(it != morphology_3d_mirp_volume_ref_vals.end()) << c.name;
	auto tol = morphology_3d_mirp_volume_ref_tols.find(c.name);
	ASSERT_TRUE(tol != morphology_3d_mirp_volume_ref_tols.end()) << c.name << " has a golden but no stated band";

	const auto& computed = computed_3d_feature_values();
	ASSERT_TRUE(computed.setup_error.empty()) << computed.setup_error;
	const std::size_t fcode_index = feature_code_index(c.code);
	ASSERT_LT(fcode_index, computed.values.size()) << c.name;
	ASSERT_FALSE(computed.values[fcode_index].empty()) << c.name;
	const double actual = computed.values[fcode_index][0];
	const double expected = it->second;
	ASSERT_TRUE(std::isfinite(actual)) << c.name;
	ASSERT_LE(relative_absdiff_pct(actual, expected), tol->second) << c.name << " actual=" << actual << " mirp=" << expected << " band=" << tol->second << "%";
}

static void assert_oracle_backed_agreement(const Feature3DCoverageCase& c)
{
	if (firstorder_3d_pyradiomics_ref_vals.find(c.name) != firstorder_3d_pyradiomics_ref_vals.end())
		assert_3d_firstorder_feature_pyradiomics(c.code, c.name);
	else if (glcm_3d_pyradiomics_ref_vals.find(c.name) != glcm_3d_pyradiomics_ref_vals.end())
		assert_3d_glcm_feature_pyradiomics(c.code, c.name);
	else if (gldm_3d_pyradiomics_ref_vals.find(c.name) != gldm_3d_pyradiomics_ref_vals.end())
		assert_3d_gldm_feature_pyradiomics(c.code, c.name);
	else if (glrlm_3d_pyradiomics_ref_vals.find(c.name) != glrlm_3d_pyradiomics_ref_vals.end())
		assert_3d_glrlm_feature_pyradiomics(c.code, c.name);
	else if (glszm_3d_pyradiomics_ref_vals.find(c.name) != glszm_3d_pyradiomics_ref_vals.end())
		assert_3d_glszm_feature_pyradiomics(c.code, c.name);
	else if (ngtdm_3d_pyradiomics_ref_vals.find(c.name) != ngtdm_3d_pyradiomics_ref_vals.end())
		assert_3d_ngtdm_feature_pyradiomics(c.code, c.name);
	else if (morphology_3d_mirp_volume_ref_vals.find(c.name) != morphology_3d_mirp_volume_ref_vals.end())
		assert_mirp_volume_shape_agreement(c);
	else
		FAIL() << c.name << " is marked WITH_3P_EMBEDDED_GT but no embedded oracle helper was found";
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

// Per-family partition of the coverage sweep. Each public 3D feature belongs to exactly one calculator
// featureset; first match wins so every case lands in exactly one family (the 94+119 split is preserved
// regardless of any incidental featureset overlap). The per-family test_3d_<family>_coverage.h files
// re-instantiate the two parameterized suites below, filtered through feature_3d_cases_for_family().
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

static std::vector<Feature3DCoverageCase> feature_3d_cases_for_family(const std::string& family, bool require_oracle_backed)
{
	std::vector<Feature3DCoverageCase> out;
	for (const auto& c : feature_3d_cases(require_oracle_backed))
		if (family_of_3d_feature(c.code) == family)
			out.push_back(c);
	return out;
}

static void assert_3d_feature_is_registered_and_computable(const Feature3DCoverageCase& c)
{
	FeatureSet fs;
	int fcode = -1;
	ASSERT_TRUE(fs.find_3D_FeatureByString(c.name, fcode)) << c.name;
	ASSERT_EQ(feature_code_value(c.code), fcode) << c.name;
	ASSERT_GE(fcode, 0) << c.name;

	const auto implemented = implemented_3d_feature_codes();
	ASSERT_TRUE(implemented.find(c.code) != implemented.end()) << c.name << " is public but not in any 3D feature method featureset";

	const auto& computed = computed_3d_feature_values();
	ASSERT_TRUE(computed.setup_error.empty()) << computed.setup_error;
	const std::size_t fcode_index = feature_index_from_int(fcode);
	ASSERT_LT(fcode_index, computed.values.size()) << c.name;
	const auto& vals = computed.values[fcode_index];
	ASSERT_FALSE(vals.empty()) << c.name;
	EXPECT_TRUE(std::any_of(vals.begin(), vals.end(), [](double v) { return std::isfinite(v); })) << c.name << " was not written by the 3D feature calculators";
}

static double local_regression_tolerance(double expected)
{
	return std::max(1.0e-9, std::abs(expected) * 1.0e-6);
}

static void assert_unvetted_local_regression_agreement(const Feature3DCoverageCase& c)
{
	const std::string family = family_of_3d_feature(c.code);
	const auto* gt = regression_coverage_table_for_family(family);
	ASSERT_TRUE(gt != nullptr) << c.name << " belongs to family '" << family << "', which has no coverage baseline table";
	auto it = gt->find(c.name);
	ASSERT_TRUE(it != gt->end()) << c.name << " has no coverage baseline in " << family << "_3d_regression_coverage_ref_vals";

	const auto& computed = computed_3d_feature_values();
	ASSERT_TRUE(computed.setup_error.empty()) << computed.setup_error;
	const std::size_t fcode_index = feature_code_index(c.code);
	ASSERT_LT(fcode_index, computed.values.size()) << c.name;

	const auto& actual = computed.values[fcode_index];
	const auto& expected = it->second;
	ASSERT_EQ(expected.size(), actual.size()) << c.name;
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		ASSERT_TRUE(std::isfinite(actual[i])) << c.name << "[" << i << "]";
		const double tolerance = local_regression_tolerance(expected[i]);
		EXPECT_NEAR(actual[i], expected[i], tolerance)
			<< c.name << "[" << i << "] actual=" << actual[i]
			<< " expected=" << expected[i] << " tolerance=" << tolerance;
	}
}

class Test3DFeature_WITH_3P_EMBEDDED_GT : public testing::TestWithParam<Feature3DCoverageCase> {};

TEST_P(Test3DFeature_WITH_3P_EMBEDDED_GT, PublicFeatureIsComputableAndHasEmbeddedOracle)
{
	const auto& c = GetParam();
	ASSERT_TRUE(externally_vetted_3d_feature_names().find(c.name) != externally_vetted_3d_feature_names().end()) << c.name;
	assert_3d_feature_is_registered_and_computable(c);
	assert_oracle_backed_agreement(c);
}

// INSTANTIATE_TEST_SUITE_P for this fixture now lives in the per-family test_3d_<family>_coverage.h
// files (one instantiation per family, unique prefix), so the coverage sweep is organized by family.

class Test3DFeature_UNVETTED_LOCAL_REGRESSION : public testing::TestWithParam<Feature3DCoverageCase> {};

TEST_P(Test3DFeature_UNVETTED_LOCAL_REGRESSION, PublicFeatureIsComputableButHasNoEmbeddedOracleYet)
{
	const auto& c = GetParam();
	ASSERT_TRUE(externally_vetted_3d_feature_names().find(c.name) == externally_vetted_3d_feature_names().end()) << c.name;
	assert_3d_feature_is_registered_and_computable(c);
	assert_unvetted_local_regression_agreement(c);
}

// INSTANTIATE_TEST_SUITE_P for this fixture likewise lives in the per-family files.

// Families migrated off the generic sweep: their previously-unvetted features are now individually
// named drift-guard tests in test_3d_<family>_regression.h (e.g. the "_grey64_regression" tests for
// glcm) instead of a table read by the parameterized Test3DFeature_UNVETTED_LOCAL_REGRESSION suite.
// feature_3d_cases(false) still lists these families' features (their oracle-backed status hasn't
// changed), so without this set the loop below would demand a coverage_baselines() entry that no
// longer exists. Update this set -- and the two counts in TEST_3D_FEATURE_COVERAGE_COUNTS below --
// every time another family is migrated; SPEC 1 still requires every public feature to be checked
// somewhere, and this is the manual bookkeeping that keeps that true without a local table.
static const std::set<std::string>& families_with_individually_ported_regression()
{
	static const std::set<std::string> families = { "glcm" };
	return families;
}

TEST(TEST_NYXUS, TEST_3D_FEATURE_COVERAGE_COUNTS)
{
	EXPECT_EQ(213u, Nyxus::UserFacing_3D_featureNames.size());
	EXPECT_EQ(94u, feature_3d_cases(true).size());
	EXPECT_EQ(119u, feature_3d_cases(false).size());
	EXPECT_EQ(Nyxus::UserFacing_3D_featureNames.size(), feature_3d_cases(true).size() + feature_3d_cases(false).size());
	// 119 total unvetted features minus the 36 glcm ported to named regression.h tests.
	EXPECT_EQ(83u, regression_coverage_ref_vals_total());
	for (const auto& c : feature_3d_cases(false))
	{
		const std::string family = family_of_3d_feature(c.code);
		if (families_with_individually_ported_regression().count(family))
			continue;
		const auto* gt = regression_coverage_table_for_family(family);
		ASSERT_TRUE(gt != nullptr) << c.name << " family=" << family;
		EXPECT_TRUE(gt->find(c.name) != gt->end()) << c.name << " missing from " << family << "_3d_regression_coverage_ref_vals";
	}
}
}
