#pragma once

// Drift guards on the segmented phantom (ut_inten.nii + ut_mask57.nii, label 57) at 64 grey levels,
// NGTDM_GREYDEPTH=64, NGTDM_RADIUS=1, ibsi=false -- recipe ngtdm3d.regression_ut_phantom. Nyxus' own
// output, so these claim no oracle (SPEC 1); the family's oracle assertions live in
// test_3d_ngtdm_pyradiomics.h and run on a different fixture.
//
// Regenerate with test_3d_ngtdm_dump_regression() below.
//
// At NGTDM_GREYDEPTH=64 the binning is MATLAB-style, which makes bin 1 the background level: a voxel
// binned there is not a matrix row of its own, but it still counts towards its neighbours'
// neighbourhood means. That is Nyxus' convention for this family and is what these values are of.

// Only what nothing this file already includes supplies: <iomanip> for the dump helper's
// setprecision. <iostream>, <string>, <tuple>, <vector> and gtest arrive through the common header
// and are not repeated.
#include <iomanip>

#include "test_3d_ngtdm_common.h"  // gtest, <iostream>, <tuple>, the settings recipe, extract_3d_ngtdm, agrees_gt
#include "test_ref_vals.h"         // ref_vals_map

static const ref_vals_map<double> ngtdm_3d_regression_ref_vals
{
	{"3NGTDM_COARSENESS", 4.1746559837294642e-05},
	{"3NGTDM_CONTRAST",   0.63226607482802633},
	{"3NGTDM_BUSYNESS",   44.389552850401223},
	{"3NGTDM_COMPLEXITY", 2819.3512285176689},
	{"3NGTDM_STRENGTH",   0.024654440905359544}
};

// Defined in test_3d_glcm_pyradiomics.h, which the translation unit includes first.
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// rel=1e-9: agrees_gt divides the golden by this, so a larger argument is a tighter band. A
// regression pin is the program's own output at full precision, so movement is the only thing it can
// catch and the band should be as tight as the pin's precision allows.
static const double ngtdm_3d_regression_frac_tolerance = 1e9;

// The settings every assertion in this file runs on, so the guards and the dump helper cannot drift
// apart in them.
static Fsettings make_ngtdm3d_regression_settings()
{
	return make_ngtdm3d_settings (64/*greydepth*/, 64/*ngtdm greydepth*/, 1/*radius*/);
}

void assert_3d_ngtdm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	// a name with no golden is a failure, not a comparison against whatever a lookup would invent
	auto iter = ngtdm_3d_regression_ref_vals.find(fname);
	ASSERT_TRUE(iter != ngtdm_3d_regression_ref_vals.end());

	int fcode = -1;
	ASSERT_NO_FATAL_FAILURE(resolve_3d_ngtdm_fcode (fcode, expecting_fcode, fname));

	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, label,
	                                          make_ngtdm3d_regression_settings()));

	ASSERT_TRUE (agrees_gt (fvals[fcode][0], iter->second,
	                        ngtdm_3d_regression_frac_tolerance)) << fname;
}

// Regenerates every golden in ngtdm_3d_regression_ref_vals at full precision, in the exact shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_NGTDM_DUMP_REGRESSION*
// and paste the output over the table above. It uses the same settings the shared assert helper
// does, so the two cannot drift apart.
void test_3d_ngtdm_dump_regression()
{
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, label,
	                                          make_ngtdm3d_regression_settings()));

	Environment e;
	std::cout << "[3DNGTDM-REGEN]\n";
	for (const auto& nv : ngtdm_3d_regression_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DNGTDM-REGEN]\t{\"" << nv.first << "\",\t"
		          << std::setprecision(17) << fvals[fcode][0] << "},\n";
	}
}

void test_3d_ngtdm_coarseness_regression()
{
	assert_3d_ngtdm_feature_regression (Nyxus::Feature3D::NGTDM_COARSENESS, "3NGTDM_COARSENESS");
}

void test_3d_ngtdm_contrast_regression()
{
	assert_3d_ngtdm_feature_regression (Nyxus::Feature3D::NGTDM_CONTRAST, "3NGTDM_CONTRAST");
}

void test_3d_ngtdm_busyness_regression()
{
	assert_3d_ngtdm_feature_regression (Nyxus::Feature3D::NGTDM_BUSYNESS, "3NGTDM_BUSYNESS");
}

void test_3d_ngtdm_complexity_regression()
{
	assert_3d_ngtdm_feature_regression (Nyxus::Feature3D::NGTDM_COMPLEXITY, "3NGTDM_COMPLEXITY");
}

void test_3d_ngtdm_strength_regression()
{
	assert_3d_ngtdm_feature_regression (Nyxus::Feature3D::NGTDM_STRENGTH, "3NGTDM_STRENGTH");
}
