#pragma once

// Mechanics of the 3D GLSZM family: the settings a run reaches the feature with, rather than the
// values it computes. Claims no oracle (SPEC 2).

// Nothing of its own: gtest, <cmath> for std::isfinite, <string> for to_string, the phantoms, the
// mock workflow and the Environment graph all arrive with the common header.
#include "test_3d_glszm_common.h"  // gtest, <cmath>, <string>, the phantoms, extract_3d_glszm

// compile_feature_settings() zero-fills every family's settings vector and writes back only the
// entries someone remembered, so a family's zero is either its documented default or a degenerate
// value nothing runs at. For GLSZM_GREYDEPTH zero is the documented default -- it selects no binning
// at all, which is the IBSI reading of the raw levels, and --3glszm/greydepth is what overrides it.
// This asserts that a run which passes no such flag still reaches the feature with settings that
// produce numbers, which is the check that would have caught NGTDM_RADIUS coming through at 0.
void test_3d_glszm_default_greydepth_mechanics()
{
	Environment e;
	e.compile_feature_settings();
	ASSERT_EQ (STNGS_GLSZM_GREYDEPTH (e.fsett_D3_GLSZM), 0);

	auto [ipath, mpath, label] = get_3d_compat_phantom();
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, e.fsett_D3_GLSZM));

	for (auto fc : D3_GLSZM_feature::featureset)
	{
		SCOPED_TRACE ("3D feature code " + std::to_string ((int)fc));
		ASSERT_TRUE (std::isfinite (fvals[(int)fc][0]));
	}
}
