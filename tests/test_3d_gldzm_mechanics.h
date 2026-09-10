#pragma once

// 3D GLDZM mechanics: what the family does with a ROI voxel whose grey level is 0.
//
// A GLDZM grey level is 1-based, and two of the three binning schemes hand back a 0 for a voxel that
// is genuinely the ROI's: `IBSI=true` bins nothing, so a raw intensity of 0 stays 0, and the
// radiomics scheme maps 0 to 0 by construction. Only the MATLAB scheme cannot, because it sends 0 to
// level 1. `prepare_GLDZM_matrix_kit` lifts the ROI's levels by one where that happens, which keeps
// those voxels in a zone.
//
// bench_gldzm_zerolevel_3d is the fixture that separates lifting them from dropping them, and it is
// the only one in the family that can: the compatibility phantom's levels are 1..8 and the segmented
// phantom is binned MATLAB-style, so neither ever produces a level 0. Nor can MIRP judge this --
// its GLDZM has the same problem with a 0 in its input -- so the expected values below are derived,
// not pinned from a tool or from Nyxus' own output.
//
// THE DERIVATION, which is the whole reason this fixture is 64 voxels and not more:
//
//   the ROI          a 4x4x4 cube, so 64 voxels, and every one of them is at city-block distance
//                    1 or 2 from the border
//   the levels       raw 0 where x+y+z is even and 5 where it is odd; the lift makes them 1 and 6
//   the zones        TWO. Two voxels of one parity class always touch at least at a corner, so each
//                    class is a single 26-connected component
//   the distances    both zones reach the ROI's surface, so both sit at distance 1: Nd = 1
//   the matrix       Ng = 6 rows (1..max), one entry in row 1 and one in row 6, both in column 1
//
// Everything below follows from that 2-entry matrix and Nv = 64. Dropping the zero-level voxels
// instead would leave ONE zone and halve `3GLDZM_ZP` to 0.015625, which is what makes these
// assertions discriminating rather than decorative.

// Only what the fixture header does not already supply: <iomanip> for the failure message's
// precision. gtest, <string>, <vector> and the Environment / roi_cache graph arrive through it.
#include <iomanip>

#include "test_3d_gldzm_common.h"   // the zero-level phantom, make_gldzm3d_settings, extract_3d_gldzm
#include "test_ref_vals.h"          // ref_vals_map

// Derived by hand from the fixture's construction, above. Not an oracle table and not a snapshot:
// every entry is arithmetic on a matrix a reader can write down.
static const ref_vals_map<double> gldzm_3d_mechanics_ref_vals{
	{"3GLDZM_SDE",                       1.0},   // m_1 / 1^2 / Ns = 2 / 2
	{"3GLDZM_LDE",                       1.0},   // m_1 * 1^2 / Ns
	{"3GLDZM_LGLZE",     0.51388888888888884},   // (1/1^2 + 1/6^2) / 2
	{"3GLDZM_HGLZE",                    18.5},   // (1^2 + 6^2) / 2
	{"3GLDZM_SDLGLE",    0.51388888888888884},   // the same, all at d = 1
	{"3GLDZM_SDHGLE",                   18.5},
	{"3GLDZM_LDLGLE",    0.51388888888888884},
	{"3GLDZM_LDHGLE",                   18.5},
	{"3GLDZM_GLNU",                      1.0},   // (1^2 + 1^2) / 2
	{"3GLDZM_GLNUN",                     0.5},
	{"3GLDZM_ZDNU",                      2.0},   // 2^2 / 2
	{"3GLDZM_ZDNUN",                     1.0},
	{"3GLDZM_ZP",                    0.03125},   // 2 zones over 64 voxels
	{"3GLDZM_GLM",                       3.5},   // (1 + 6) / 2
	{"3GLDZM_GLV",                      6.25},   // ((1-3.5)^2 + (6-3.5)^2) / 2
	{"3GLDZM_ZDM",                       1.0},   // every zone is at distance 1
	{"3GLDZM_ZDV",                       0.0},   // ... so their distances have no spread
	{"3GLDZM_ZDE",                       1.0},   // -2 * (0.5 * log2 0.5)
};

// SPEC 7's exact tier. These are exact binary fractions or short arithmetic on them, so the only
// slack any of them needs is the EPS inside the entropy's logarithm.
static const double gldzm_3d_mechanics_tolerance = 1.e-9;

// The zone count, read off ZP, is the single number the whole behaviour turns on: 2 if the
// zero-level voxels are kept, 1 if they are dropped.
static const double gldzm_3d_mechanics_zones = 2.0;
static const double gldzm_3d_mechanics_roi_voxels = 64.0;

void test_3d_gldzm_zero_level_voxels_are_zoned_mechanics()
{
	auto [ipath, mpath, label] = get_3d_gldzm_zerolevel_phantom();

	std::vector<std::vector<double>> fvals;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(fvals, ipath, mpath, label, make_gldzm3d_settings(64, true)));

	Environment e;
	for (const auto& nv : gldzm_3d_mechanics_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode)) << nv.first;
		EXPECT_NEAR(fvals[fcode][0], nv.second, gldzm_3d_mechanics_tolerance)
			<< nv.first << " actual=" << std::setprecision(17) << fvals[fcode][0];
	}

	// Stated again as the count it is, so a failure names the thing that went wrong rather than
	// leaving a reader to divide two feature values in their head.
	int zp = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString("3GLDZM_ZP", zp));
	EXPECT_NEAR(fvals[zp][0] * gldzm_3d_mechanics_roi_voxels, gldzm_3d_mechanics_zones, 1.e-9)
		<< "the ROI's zero-level voxels are not in a zone: " << std::setprecision(17)
		<< fvals[zp][0] * gldzm_3d_mechanics_roi_voxels << " zones over "
		<< gldzm_3d_mechanics_roi_voxels << " voxels, expected " << gldzm_3d_mechanics_zones;
}

// GREYDEPTH=0 and IBSI=true are two ways of asking for the same thing -- calculate() overwrites the
// grey depth with 0 when IBSI is set, and `ibsi_grey_binning` is `== 0` -- so they are one config
// point of the matrix (tests/vetting/matrix/gldzm3d.md) and must return one set of values. This is
// what holds the two spellings together; without it the matrix's claim is only a reading of the
// source.
void test_3d_gldzm_no_binning_spellings_agree_mechanics()
{
	auto [ipath, mpath, label] = get_3d_gldzm_zerolevel_phantom();

	std::vector<std::vector<double>> by_ibsi, by_greydepth;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(by_ibsi, ipath, mpath, label, make_gldzm3d_settings(64, true)));
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(by_greydepth, ipath, mpath, label, make_gldzm3d_settings(0, false)));

	Environment e;
	for (const auto& nv : gldzm_3d_mechanics_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode)) << nv.first;
		EXPECT_DOUBLE_EQ(by_greydepth[fcode][0], by_ibsi[fcode][0])
			<< nv.first << " differs between GREYDEPTH=0,IBSI=false and GREYDEPTH=64,IBSI=true";
	}
}

// The third of the family's three binning points. `GREYDEPTH` negative selects the radiomics
// scheme, which is a real production configuration (`--coarseGrayDepth` takes a negative value) and
// is the one point of the matrix no assertion reached.
//
// On bench_compat_gldzm_3d at a bin count of 8 the scheme is the IDENTITY on the fixture's levels --
// `to_grayscale_radiomix` maps 1..8 over a bin width of 7/8 back onto 1..8, the top level clipping
// into the last bin -- so this point computes the GLDZM of the same grey levels the vetted no-binning
// point does, and MIRP's goldens for that point apply here unchanged. That is measured below rather
// than argued: the assertion is that the two points return the same numbers on this fixture.
//
// It does NOT generalise to a bin count that actually re-bins. There the scheme is a Nyxus
// convention with the same lower-bin-edge gap the MATLAB one has, and no tool reproduces it;
// tests/vetting/matrix/gldzm3d.md records that limit with this cell.
void test_3d_gldzm_radiomics_binning_is_identity_here_mechanics()
{
	auto [ipath, mpath, label] = get_3d_compat_gldzm_phantom();

	std::vector<std::vector<double>> by_radiomics, by_no_binning;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(by_radiomics, ipath, mpath, label, make_gldzm3d_settings(-8, false)));
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(by_no_binning, ipath, mpath, label, make_gldzm3d_settings(64, true)));

	Environment e;
	for (const auto& nv : gldzm_3d_mechanics_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode)) << nv.first;
		EXPECT_NEAR(by_radiomics[fcode][0], by_no_binning[fcode][0], 1.e-9)
			<< nv.first << " differs between GREYDEPTH=-8,IBSI=false and the no-binning point, so the "
			<< "radiomics scheme is not the identity on this fixture after all";
	}
}
