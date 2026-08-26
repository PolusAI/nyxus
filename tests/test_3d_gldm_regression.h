#pragma once

// 3D GLDM drift guards on the segmented ut phantom.
//
// Recipe gldm3d.regression_ut_phantom: bench_ut57_3d (phantoms/ut_inten.nii + ut_mask57.nii,
// label 57) with GREYDEPTH=64, IBSI=false and GLDM_GREYDEPTH=64, whose positive sign selects
// MATLAB-style level-count binning -- a different configuration from the PyRadiomics recipe next
// door, not a second opinion on it.
//
// These pins are Nyxus' own output. They establish no vetting (SPEC 1); what they catch is movement.
// Regenerate them with
//     runAllTests --gtest_filter=*3D_GLDM_DUMP_REGRESSION*

// Only what nothing this file already includes supplies: <iomanip> for the dump helper's
// setprecision. gtest, <iostream>, <string>, <vector>, the phantoms and the Environment graph all
// arrive through the common header, and the table aliases through test_ref_vals.h.
#include <iomanip>

#include "test_3d_gldm_common.h"   // gtest, <iostream>, the phantoms, the settings recipe, extract_3d_gldm, agrees_gt
#include "test_ref_vals.h"         // ref_vals_map

static const ref_vals_map<double> gldm_3d_regression_ref_vals
{
	{"3GLDM_DE", 8.403471863314282},
	{"3GLDM_DN", 32088.282649253732},
	{"3GLDM_DNN", 0.11692616986814122},
	{"3GLDM_DV", 14.61601502403008},
	{"3GLDM_GLN", 6480.8302238805973},
	{"3GLDM_GLV", 153.09468390995261},
	{"3GLDM_HGLE", 1957.1945545708954},
	{"3GLDM_LDE", 40.639575559701491},
	{"3GLDM_LDHGLE", 46806.000728777988},
	{"3GLDM_LDLGLE", 0.051952008428935097},
	{"3GLDM_LGLE", 0.00073572128237550161},
	{"3GLDM_SDE", 0.15360647002552097},
	{"3GLDM_SDHGLE", 390.54742995738991},
	{"3GLDM_SDLGLE", 7.4259950258209084e-05}
};

// agrees_gt divides the golden by this, so a larger argument is a tighter band. A snapshot is the
// program's own output round-tripped through seventeen digits, so the only honest band is one that
// leaves no room for drift: rel=1e-9. This is deliberately NOT called SPEC 7's exact tier, which is
// an absolute 1e-9 and belongs to an oracle comparison -- there is no oracle here.
static const double gldm_3d_regression_frac_tolerance = 1e9;

static void assert_3d_gldm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	auto iter = gldm_3d_regression_ref_vals.find (fname);
	ASSERT_TRUE (iter != gldm_3d_regression_ref_vals.end()) << fname;

	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	Fsettings s = make_gldm3d_settings (64/*greydepth*/, 64/*matlab level-count binning*/);

	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldm (fvals, cube, lo, hi, ipath, mpath, label, s));

	int fcode = -1;
	ASSERT_NO_FATAL_FAILURE(resolve_3d_gldm_fcode (fcode, expecting_fcode, fname));

	ASSERT_TRUE (agrees_gt (fvals[fcode][0], iter->second, gldm_3d_regression_frac_tolerance)) << fname;
}

// Regenerates every pin above at full precision, in the shape the table wants. Run it with
//     runAllTests --gtest_filter=*3D_GLDM_DUMP_REGRESSION*
void test_3d_gldm_dump_regression()
{
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	Fsettings s = make_gldm3d_settings (64/*greydepth*/, 64/*matlab level-count binning*/);

	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldm (fvals, cube, lo, hi, ipath, mpath, label, s));

	Environment e;
	std::cout << "[3DGLDM-REGR] cube " << cube.width() << "x" << cube.height() << "x" << cube.depth()
	          << ", intensity range [" << lo << ", " << hi << "]\n";
	for (const auto& nv : gldm_3d_regression_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DGLDM-REGR]\t{\"" << nv.first << "\", "
		          << std::setprecision(17) << fvals[fcode][0] << "},\tpinned "
		          << std::setprecision(17) << nv.second << "\n";
	}
}

void test_3d_gldm_de_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DE, "3GLDM_DE"); }
void test_3d_gldm_dn_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DN, "3GLDM_DN"); }
void test_3d_gldm_dnn_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DNN, "3GLDM_DNN"); }
void test_3d_gldm_dv_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_DV, "3GLDM_DV"); }
void test_3d_gldm_gln_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_GLN, "3GLDM_GLN"); }
void test_3d_gldm_glv_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_GLV, "3GLDM_GLV"); }
void test_3d_gldm_hgle_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_HGLE, "3GLDM_HGLE"); }
void test_3d_gldm_lde_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_LDE, "3GLDM_LDE"); }
void test_3d_gldm_ldhgle_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_LDHGLE, "3GLDM_LDHGLE"); }
void test_3d_gldm_ldlgle_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_LDLGLE, "3GLDM_LDLGLE"); }
void test_3d_gldm_lgle_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_LGLE, "3GLDM_LGLE"); }
void test_3d_gldm_sde_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE"); }
void test_3d_gldm_sdhgle_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_SDHGLE, "3GLDM_SDHGLE"); }
void test_3d_gldm_sdlgle_regression() { assert_3d_gldm_feature_regression(Nyxus::Feature3D::GLDM_SDLGLE, "3GLDM_SDLGLE"); }
