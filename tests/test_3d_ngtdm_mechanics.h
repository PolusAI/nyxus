#pragma once

// Mechanics of the 3D NGTDM family: the settings a run reaches the feature with, rather than the
// values it computes. Claims no oracle (SPEC 2).

// Nothing of its own: gtest, <cmath> for std::isfinite, <string> for to_string, the phantom, the
// mock workflow and the Environment graph all arrive with the common header.
#include "test_3d_ngtdm_common.h"  // gtest, <cmath>, <string>, the phantom, extract_3d_ngtdm

// NGTDM_RADIUS is the Chebyshev radius of the neighbourhood a voxel's dependency is measured over,
// and the family is undefined at 0: gather_zones() then visits only the centre voxel, skips it, and
// no voxel is recorded as having a neighbour, so the matrix stays empty and every feature is reported
// as the soft-NaN value. This asserts that a run which calls no set_metaparam("3ngtdm/radius=...") reaches the
// feature at exactly 1, which is the same guarantee compile_feature_settings() already gives
// GLCM_OFFSET a few lines up, for the identical reason.
//
// The radius is pinned rather than bounded below. Every radius from 1 up is finite, so a >= 1 check
// would pass on a default that had drifted to 2 -- a different neighbourhood, and different values
// for all five features, which is what ngtdm3d.pyradiomics_binwidth1_r2 measures.
void test_3d_ngtdm_default_radius_mechanics()
{
	Environment e;
	e.compile_feature_settings();
	ASSERT_EQ (STNGS_NGTDM_RADIUS (e.fsett_D3_NGTDM), 1);

	auto [ipath, mpath, label] = get_3d_compat_ngtdm_phantom();
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, label, e.fsett_D3_NGTDM));

	for (auto fc : D3_NGTDM_feature::featureset)
	{
		SCOPED_TRACE ("3D feature code " + std::to_string ((int)fc));
		ASSERT_TRUE (std::isfinite (fvals[(int)fc][0]));
	}
}

// A ROI none of whose voxels has a ROI neighbour has an empty NGTDM, and the family reports every
// feature as the soft-NaN value rather than a number. This is the IBSI convention, which MIRP follows
// as well (it reports NaN for the same ROI): a voxel with no ROI neighbour is in no matrix row and
// counts towards none of n_i, Nvp or Ngp. PyRadiomics instead keeps such a voxel as a row with
// s_i = 0, so any ROI holding one -- not only one made of nothing else -- differs from PyRadiomics on
// all five features; test_3d_ngtdm_mirp.h holds a ROI that mixes the two kinds of voxel to MIRP.
//
// Label 58 of the ball phantom is two voxels at
// opposite corners of a 9x9x9 box, at two different levels, so it gets past the single-level check
// and reaches the neighbourhood scan; its bounding box is the whole volume, and every cell of it but
// those two is background. A neighbourhood that counted background cells would find neighbours for
// both voxels -- and for the background cells themselves, if those were taken for centres -- and
// hand back five finite values.
//
// The sentinel is a value no NGTDM feature of a real ROI takes, so equality with it is the refusal
// path having run, not a feature that happened to come out at the default soft-NaN of 0.
void test_3d_ngtdm_isolated_voxels_mechanics()
{
	const double sentinel = -12345.0;
	auto [ipath, mpath, label, isolated_label] = get_3d_ngtdm_ball_phantom();
	Fsettings s = make_ngtdm3d_settings (100/*greydepth*/, 0/*no ngtdm binning*/, 1/*radius*/);
	s[(int)NyxSetting::SOFTNAN].rval = sentinel;

	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	Ngtdm3dMatrix m;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, isolated_label, s, &m));
	ASSERT_EQ (cube.size(), size_t(9 * 9 * 9));
	ASSERT_EQ (m.I.size(), size_t(2));		// the two levels of its two voxels, nothing from the background

	for (auto fc : D3_NGTDM_feature::featureset)
	{
		SCOPED_TRACE ("3D feature code " + std::to_string ((int)fc));
		ASSERT_EQ (fvals[(int)fc][0], sentinel);
	}

	// At radius 8 the two corners are within reach of each other, so the same ROI has a matrix and
	// five values -- the sentinel above came from the empty matrix, not from the ROI being refused for
	// being two voxels.
	s[(int)NyxSetting::NGTDM_RADIUS].ival = 8;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, isolated_label, s));
	for (auto fc : D3_NGTDM_feature::featureset)
	{
		SCOPED_TRACE ("radius 8, 3D feature code " + std::to_string ((int)fc));
		ASSERT_NE (fvals[(int)fc][0], sentinel);
	}
}

// The same empty matrix through osized_calculate(), which streams the ROI from its disk-backed voxel
// cloud. The ROI is label 58's layout -- two voxels at opposite corners of a 9x9x9 box, levels 1 and
// 3 -- built straight into the cloud, so the out-of-core pass runs without a ROI large enough to be
// sent there by a featurisation. Through the Python API the soft-NaN value cannot be told from a NaN,
// since the output stage writes any non-finite value as the soft-NaN value, so this is where the
// out-of-core empty-matrix return is observed. At radius 8 the corners reach each other and the same
// pass gives five values, equal to calculate()'s on the same ROI.
void test_3d_ngtdm_ooc_empty_matrix_mechanics()
{
	const double sentinel = -12345.0;
	const int n = 9;

	LR r (1);
	r.aabb.init_x (0); r.aabb.update_x (n - 1);
	r.aabb.init_y (0); r.aabb.update_y (n - 1);
	r.aabb.init_z (0); r.aabb.update_z (n - 1);
	r.raw_voxels_NT.init (r.label, "ngtdm3d_ooc_empty_matrix");
	for (int z = 0; z < n; z++)
	{
		r.raw_voxels_NT.begin_slab (z);
		if (z == 0 || z == n - 1)
		{
			Pixel3 p (z, z, z, (PixIntens) (z == 0 ? 1 : 3));
			r.raw_pixels_3D.push_back (p);
			r.raw_voxels_NT.add_voxel (p);
		}
	}
	r.aux_min = 1;
	r.aux_max = 3;
	r.aux_area = 2;
	r.aux_image_cube.calculate_from_pixelcloud (r.raw_pixels_3D, r.aabb);

	Fsettings s = make_ngtdm3d_settings (100/*greydepth*/, 0/*no ngtdm binning*/, 1/*radius*/);
	s[(int)NyxSetting::SOFTNAN].rval = sentinel;

	auto run = [&r, &s](bool ooc) -> std::vector<std::vector<double>>
	{
		D3_NGTDM_feature f;
		r.initialize_fvals();
		if (ooc)
		{
			ImageLoader dummy;	// osized_calculate reads the cloud, not the loader
			f.osized_calculate (r, s, dummy);
		}
		else
			f.calculate (r, s);
		f.save_value (r.fvals);
		return r.fvals;
	};

	auto ooc = run (true);
	for (auto fc : D3_NGTDM_feature::featureset)
	{
		SCOPED_TRACE ("radius 1, 3D feature code " + std::to_string ((int)fc));
		EXPECT_EQ (ooc[(int)fc][0], sentinel);
	}

	s[(int)NyxSetting::NGTDM_RADIUS].ival = 8;
	ooc = run (true);
	auto in_ram = run (false);
	r.raw_voxels_NT.clear();
	for (auto fc : D3_NGTDM_feature::featureset)
	{
		SCOPED_TRACE ("radius 8, 3D feature code " + std::to_string ((int)fc));
		EXPECT_TRUE (std::isfinite (ooc[(int)fc][0]));
		EXPECT_NE (ooc[(int)fc][0], sentinel);
		EXPECT_DOUBLE_EQ (ooc[(int)fc][0], in_ram[(int)fc][0]);
	}
}
