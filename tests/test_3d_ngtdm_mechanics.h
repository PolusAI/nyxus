#pragma once

// Mechanics of the 3D NGTDM family: the settings a run reaches the feature with, rather than the
// values it computes. Claims no oracle (SPEC 2).

// Nothing of its own: gtest, <cmath> for std::isfinite, <string> for to_string, the phantom, the
// mock workflow and the Environment graph all arrive with the common header.
#include "test_3d_ngtdm_common.h"  // gtest, <cmath>, <string>, the phantom, extract_3d_ngtdm

// NGTDM_RADIUS is the Chebyshev radius of the neighbourhood a voxel's dependency is measured over,
// and the family is undefined at 0: gather_zones() then visits only the centre voxel, skips it, and
// no voxel is recorded as having a neighbour, so the matrix stays empty and every feature divides by
// zero. This asserts that a run which calls no set_metaparam("3ngtdm/radius=...") reaches the
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
