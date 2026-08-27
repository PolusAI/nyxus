#pragma once

// 3D GLDM drift guards on the segmented ut phantom, plus the guard on the degenerate ROI.
//
// Recipe gldm3d.regression_ut_phantom: bench_ut57_3d (phantoms/ut_inten.nii + ut_mask57.nii,
// label 57) with GREYDEPTH=64, IBSI=false and GLDM_GREYDEPTH=64, whose positive sign selects
// MATLAB-style level-count binning -- a different configuration from the PyRadiomics recipe next
// door, not a second opinion on it.
//
// Recipe gldm3d.regression_ut_phantom_nobinning: the same phantom and label with GLDM_GREYDEPTH=0.
// Zero is not one config among several -- it is the value a settings vector carries when nothing
// names the family's binning, i.e. what env_features.cpp's zero-fill leaves behind and what every
// run that omits --3gldm/greydepth computes. The one above cannot stand in for it: the two differ by
// three orders of magnitude on the same ROI (3GLDM_HGLE 1957.19 against 4275550.79), so a pin taken
// at 64 says nothing about what the default emits.
//
// Recipe gldm3d.regression_constant_roi: a synthetic nonempty ROI of one intensity at the same
// GLDM_GREYDEPTH=0, reachable wherever a segmentation lands on a flat region. All fourteen come back
// as the no-value sentinel. That is a defect, not a convention -- PyRadiomics 3.0.1 computes a full
// GLDM there -- and the guard pins what Nyxus does today so the day it changes is visible;
// tests/vetting/matrix/gldm3d.md carries the cell and the measured oracle values.
//
// These pins are Nyxus' own output. They establish no vetting (SPEC 1); what they catch is movement.
// Regenerate them with
//     runAllTests --gtest_filter=*3D_GLDM_DUMP_REGRESSION*
//     runAllTests --gtest_filter=*3D_GLDM_DUMP_NOBINNING_REGRESSION*

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

// The same fourteen at the compiled default GLDM_GREYDEPTH=0, on the same ROI. Nothing in the tree
// asserted the default configuration before this table: the family's binning is the only setting it
// reads, and every pin stood at a value only an explicit --3gldm/greydepth reaches.
static const ref_vals_map<double> gldm_3d_regression_nobinning_ref_vals
{
	{"3GLDM_DE", 11.489092381392878},
	{"3GLDM_DN", 196010.87360074627},
	{"3GLDM_DNN", 0.71424204757734622},
	{"3GLDM_DV", 0.6119105729315345},
	{"3GLDM_GLN", 157.9715485074627},
	{"3GLDM_GLV", 341996.30156539206},
	{"3GLDM_HGLE", 4275550.7913654381},
	{"3GLDM_LDE", 2.1725163246268657},
	{"3GLDM_LDHGLE", 7987719.3384444965},
	{"3GLDM_LDLGLE", 9.362437020082869e-07},
	{"3GLDM_LGLE", 3.400101714136789e-07},
	{"3GLDM_SDE", 0.86996111189993075},
	{"3GLDM_SDHGLE", 3831567.8645316572},
	{"3GLDM_SDLGLE", 2.8075039742774167e-07}
};

// agrees_gt divides the golden by this, so a larger argument is a tighter band. A snapshot is the
// program's own output round-tripped through seventeen digits, so the only honest band is one that
// leaves no room for drift: rel=1e-9. This is deliberately NOT called SPEC 7's exact tier, which is
// an absolute 1e-9 and belongs to an oracle comparison -- there is no oracle here.
static const double gldm_3d_regression_frac_tolerance = 1e9;

// The no-value sentinel the degenerate-ROI guard runs under. Deliberately a value no GLDM feature
// can take -- production defaults it to 0.0, which is also what several of the fourteen legitimately
// compute, so a guard run at the default could not tell the two apart.
static const double gldm_3d_regression_softnan = -98765.0;

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

static void assert_3d_gldm_feature_nobinning_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	auto iter = gldm_3d_regression_nobinning_ref_vals.find (fname);
	ASSERT_TRUE (iter != gldm_3d_regression_nobinning_ref_vals.end()) << fname;

	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	// GREYDEPTH stays at the neighbouring recipe's 64 because this family never reads it, so the two
	// tables differ in the one setting under test and nothing else.
	Fsettings s = make_gldm3d_settings (64/*greydepth, inert here*/, 0/*the compiled default: no binning*/);

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

// Regenerates the default-configuration pins at full precision. Run it with
//     runAllTests --gtest_filter=*3D_GLDM_DUMP_NOBINNING_REGRESSION*
void test_3d_gldm_dump_nobinning_regression()
{
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	Fsettings s = make_gldm3d_settings (64/*greydepth, inert here*/, 0/*the compiled default: no binning*/);

	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldm (fvals, cube, lo, hi, ipath, mpath, label, s));

	Environment e;
	std::cout << "[3DGLDM-NOBIN] cube " << cube.width() << "x" << cube.height() << "x" << cube.depth()
	          << ", intensity range [" << lo << ", " << hi << "]\n";
	for (const auto& nv : gldm_3d_regression_nobinning_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DGLDM-NOBIN]\t{\"" << nv.first << "\", "
		          << std::setprecision(17) << fvals[fcode][0] << "},\tpinned "
		          << std::setprecision(17) << nv.second << "\n";
	}
}

void test_3d_gldm_de_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_DE, "3GLDM_DE"); }
void test_3d_gldm_dn_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_DN, "3GLDM_DN"); }
void test_3d_gldm_dnn_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_DNN, "3GLDM_DNN"); }
void test_3d_gldm_dv_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_DV, "3GLDM_DV"); }
void test_3d_gldm_gln_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_GLN, "3GLDM_GLN"); }
void test_3d_gldm_glv_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_GLV, "3GLDM_GLV"); }
void test_3d_gldm_hgle_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_HGLE, "3GLDM_HGLE"); }
void test_3d_gldm_lde_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_LDE, "3GLDM_LDE"); }
void test_3d_gldm_ldhgle_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_LDHGLE, "3GLDM_LDHGLE"); }
void test_3d_gldm_ldlgle_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_LDLGLE, "3GLDM_LDLGLE"); }
void test_3d_gldm_lgle_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_LGLE, "3GLDM_LGLE"); }
void test_3d_gldm_sde_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE"); }
void test_3d_gldm_sdhgle_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_SDHGLE, "3GLDM_SDHGLE"); }
void test_3d_gldm_sdlgle_nobinning_regression() { assert_3d_gldm_feature_nobinning_regression(Nyxus::Feature3D::GLDM_SDLGLE, "3GLDM_SDLGLE"); }

// A nonempty ROI of a single intensity. PyRadiomics 3.0.1 builds a full dependence matrix on the
// same voxels -- one grey level, dependences 1..4 -- and Nyxus returns the sentinel, so this pins a
// divergence rather than a convention. Run at GLDM_GREYDEPTH=0 because that is where the intercept
// stands alone: a negative control restricting it to empty cubes has the arithmetic underneath
// reproduce PyRadiomics to the last bit on all seven dependence-axis features, so nothing below the
// guard is wrong. At radiomics binning the same control stays green instead -- to_grayscale_radiomix()
// divides by max - min, which is zero here, so every voxel bins to background and the Nz == 0 branch
// emits the same sentinel by a second route. Cell and values: tests/vetting/matrix/gldm3d.md.
void test_3d_gldm_constant_roi_regression()
{
	Fsettings s = make_gldm3d_settings (64/*greydepth, inert here*/, 0/*the compiled default: no binning*/);
	s[(int)NyxSetting::SOFTNAN].rval = gldm_3d_regression_softnan;

	LR r;
	r.aux_image_cube.allocate (4/*width*/, 4/*height*/, 3/*depth*/);
	for (auto& v : r.aux_image_cube)
		v = 7;
	r.aux_min = r.aux_max = 7;
	ASSERT_NO_THROW (r.initialize_fvals());

	D3_GLDM_feature f;
	ASSERT_NO_THROW (f.calculate (r, s));
	ASSERT_NO_THROW (f.save_value (r.fvals));

	// The fourteen spelled out rather than walked off featureset, because a name the scanner can read
	// is what files this case under the features it answers for. The tally below is what keeps the
	// list from falling behind the enum it stands in for.
	int checked = 0;
	Environment e;
	for (const std::string& fname : {
		"3GLDM_SDE", "3GLDM_LDE", "3GLDM_GLN", "3GLDM_DN", "3GLDM_DNN", "3GLDM_GLV", "3GLDM_DV",
		"3GLDM_DE", "3GLDM_LGLE", "3GLDM_HGLE", "3GLDM_SDLGLE", "3GLDM_SDHGLE", "3GLDM_LDLGLE",
		"3GLDM_LDHGLE" })
	{
		int fcode = -1;
		ASSERT_TRUE (e.theFeatureSet.find_3D_FeatureByString (fname, fcode)) << fname;
		EXPECT_EQ (r.fvals[fcode][0], gldm_3d_regression_softnan) << fname;
		checked++;
	}
	ASSERT_EQ ((size_t)checked, D3_GLDM_feature::featureset.size());
}
