#pragma once

// 3D GLDZM mechanics: what the family does with a ROI voxel whose grey level is 0, and the bound on
// the out-of-core border distance (at the end of the file).
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
// precision, <climits> and <stdexcept> for the out-of-core bound. gtest, <string>, <vector> and the
// Environment / roi_cache graph arrive through it.
#include <climits>
#include <iomanip>
#include <stdexcept>

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

// The derived table and the zone count, asserted against one extraction.
static void assert_3d_gldzm_zero_level_voxels_are_zoned (const std::vector<std::vector<double>>& fvals)
{
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

void test_3d_gldzm_zero_level_voxels_are_zoned_mechanics()
{
	auto [ipath, mpath, label] = get_3d_gldzm_zerolevel_phantom();

	std::vector<std::vector<double>> fvals;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(fvals, ipath, mpath, label, make_gldzm3d_settings(64, true)));
	assert_3d_gldzm_zero_level_voxels_are_zoned (fvals);
}

// The same fixture through the radiomics scheme, which gathers the ROI's grey levels as a set rather
// than as a 1..max ladder, so its lift is a separate code path from the no-binning point's.
//
// At a bin count of 5 the scheme maps the fixture's raw levels onto themselves: `to_grayscale_radiomix`
// sends 0 to 0 by construction, and 5 over a bin width of (5-0)/5 = 1 to bin 6, which clips into the
// last bin, 5. The lift then makes them 1 and 6 -- the levels of the no-binning point -- so the
// derived table above applies unchanged. A set gathered without the lift would hold 0 and 5, and a
// level-1 zone would have no row to go in.
void test_3d_gldzm_zero_level_voxels_are_zoned_radiomics_mechanics()
{
	auto [ipath, mpath, label] = get_3d_gldzm_zerolevel_phantom();

	std::vector<std::vector<double>> fvals;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldzm(fvals, ipath, mpath, label, make_gldzm3d_settings(-5, false)));
	assert_3d_gldzm_zero_level_voxels_are_zoned (fvals);
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

// ---- The out-of-core border distance ----
//
// osized_calculate() keeps each ROI voxel's distance to the ROI border in 16 bits, with the top
// value as its "unsettled" mark and 0 as "not the ROI's". A voxel's distance is at most its distance
// to the nearest face of the bounding box, so the largest one a box produces is
// (shortest side + 1) / 2. A box is refused when that reaches the unsettled mark, which takes a
// shortest side of 131069; the long sides do not enter into it.

// One synthetic ROI filled for both paths: the voxel list and cube calculate() reads, and the
// disk-backed cloud osized_calculate() streams. 'keep' says which voxels of the w x h x d box are the
// ROI's -- it must keep one on every face, so the box is the ROI's tight bounding box -- and 'inten'
// what each carries.
template <class Keep, class Inten>
static void make_gldzm3d_both_paths_roi (LR& r, int w, int h, int d, Keep keep, Inten inten, const std::string& cloud_name)
{
	r.aabb.init_x (0); r.aabb.update_x (w - 1);
	r.aabb.init_y (0); r.aabb.update_y (h - 1);
	r.aabb.init_z (0); r.aabb.update_z (d - 1);

	r.raw_voxels_NT.init (r.label, cloud_name);
	for (int z = 0; z < d; z++)
	{
		r.raw_voxels_NT.begin_slab (z);
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				if (! keep (x, y, z))
					continue;
				Pixel3 p (x, y, z, (PixIntens) inten (x, y, z));
				r.raw_pixels_3D.push_back (p);
				r.raw_voxels_NT.add_voxel (p);
				r.aux_min = (std::min) (r.aux_min, p.inten);
				r.aux_max = (std::max) (r.aux_max, p.inten);
			}
	}
	r.aux_area = (unsigned int) r.raw_pixels_3D.size();
	r.aux_image_cube.calculate_from_pixelcloud (r.raw_pixels_3D, r.aabb);
}

// A box 131071 voxels long on a 5x5 cross-section returns the in-RAM values out-of-core. What this
// discriminates: a refusal keyed on the LONGEST side -- whether as the side itself or as
// (side + 1) / 2 against 16 bits -- refuses this ROI, although its deepest voxel is 3 steps from the
// border.
//
// The levels put zones at every depth the box has: the outer shell alternates 1 and 2 every seven
// voxels, the ring below it is one level-3 zone, and the centre line alternates 5 and 6 every
// thousand voxels. A notch in one face every thousand voxels brings the ring and one centre segment
// per notch closer to the border.
void test_3d_gldzm_ooc_long_box_matches_in_ram_mechanics()
{
	const int W = 131071, H = 5, D = 5;
	ASSERT_FALSE(D3_GLDZM_feature::ooc_border_distance_fits (W, W, W))
		<< "the fixture's long side must be one the bound refuses when it is the shortest";

	auto keep = [](int x, int y, int z) { return ! (y == 0 && x % 1000 == 500); };
	auto inten = [](int x, int y, int z) -> int
	{
		if (y == 2 && z == 2)
			return 5 + (x / 1000) % 2;
		if (y >= 1 && y <= 3 && z >= 1 && z <= 3)
			return 3;
		return 1 + (x / 7) % 2;
	};

	LR r (1);
	make_gldzm3d_both_paths_roi (r, W, H, D, keep, inten, "gldzm3d_ooc_long_box");
	const Fsettings s = make_gldzm3d_settings (64, true);

	std::vector<std::vector<double>> in_ram, ooc;
	{
		D3_GLDZM_feature f;
		r.initialize_fvals();
		ASSERT_NO_THROW(f.calculate (r, s));
		f.save_value (r.fvals);
		in_ram = r.fvals;
	}
	{
		D3_GLDZM_feature f;
		ImageLoader dummy;	// osized_calculate reads the cloud, not the loader
		r.initialize_fvals();
		ASSERT_NO_THROW(f.osized_calculate (r, s, dummy)) << "a long, thin ROI is within the bound";
		f.save_value (r.fvals);
		ooc = r.fvals;
	}
	r.raw_voxels_NT.clear();

	Environment e;
	for (const auto& nv : gldzm_3d_mechanics_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode)) << nv.first;
		EXPECT_DOUBLE_EQ(ooc[fcode][0], in_ram[fcode][0]) << nv.first << " differs out-of-core";
	}

	// the fixture has zones deeper than the surface, so the distances are exercised past 1
	int zdm = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString("3GLDZM_ZDM", zdm));
	EXPECT_GT(in_ram[zdm][0], 1.0) << "every zone of the fixture is at distance 1";
}

// A box whose border distances reach the unsettled mark is refused before anything is allocated or
// read. What this discriminates: without the refusal the pass allocates its distance buffer first --
// 131069^3 voxels, 4.5 PB -- and fails with bad_alloc, which names neither the ROI nor the cause.
// The cloud holds two voxels; the box is what is too large.
void test_3d_gldzm_ooc_too_thick_box_refused_mechanics()
{
	const int side = 131069;	// (side + 1) / 2 == 65535, the unsettled mark

	LR r (1);
	r.aabb.init_x (0); r.aabb.update_x (side - 1);
	r.aabb.init_y (0); r.aabb.update_y (side - 1);
	r.aabb.init_z (0); r.aabb.update_z (side - 1);
	r.raw_voxels_NT.init (r.label, "gldzm3d_ooc_too_thick");
	r.raw_voxels_NT.begin_slab (0);
	r.raw_voxels_NT.add_voxel (Pixel3 (0, 0, 0, (PixIntens) 1));
	r.raw_voxels_NT.add_voxel (Pixel3 (1, 0, 0, (PixIntens) 2));
	r.aux_min = 1;
	r.aux_max = 2;
	r.aux_area = 2;

	D3_GLDZM_feature f;
	ImageLoader dummy;
	std::string refusal, other;
	try
	{
		f.osized_calculate (r, make_gldzm3d_settings (64, true), dummy);
	}
	catch (const std::runtime_error& ex)
	{
		refusal = ex.what();
	}
	catch (const std::exception& ex)
	{
		other = ex.what();
	}
	r.raw_voxels_NT.clear();

	ASSERT_FALSE(refusal.empty()) << "the box was not refused"
		<< (other.empty() ? std::string() : "; the pass failed with: " + other);
	EXPECT_NE(refusal.find ("border distance"), std::string::npos) << refusal;
	EXPECT_NE(refusal.find ("131069x131069x131069"), std::string::npos) << "the refusal names the box: " << refusal;
}

// The bound itself, at the sides where it turns. What this discriminates: an off-by-one either way
// -- refusing the box whose deepest voxel is 65534 steps in, or accepting the one whose deepest voxel
// would be stored as the unsettled mark -- and a bound on the longest side instead of the shortest.
void test_3d_gldzm_ooc_border_distance_bound_mechanics()
{
	EXPECT_TRUE(D3_GLDZM_feature::ooc_border_distance_fits (1, 1, 1));
	EXPECT_TRUE(D3_GLDZM_feature::ooc_border_distance_fits (131068, 131068, 131068))
		<< "deepest voxel 65534 steps in: the largest distance that fits";
	EXPECT_FALSE(D3_GLDZM_feature::ooc_border_distance_fits (131069, 131069, 131069))
		<< "deepest voxel 65535 steps in: the unsettled mark";
	EXPECT_FALSE(D3_GLDZM_feature::ooc_border_distance_fits (131070, 131070, 131070));
	EXPECT_TRUE(D3_GLDZM_feature::ooc_border_distance_fits (INT_MAX, INT_MAX, 131068))
		<< "the long sides do not bound the distance";
	EXPECT_FALSE(D3_GLDZM_feature::ooc_border_distance_fits (INT_MAX, 131069, INT_MAX))
		<< "the shortest side does, on whichever axis it lies";
}
