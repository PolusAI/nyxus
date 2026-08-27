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

// IBSI=true is not a third binning scheme, and this is the assertion that says so rather than the
// comment that used to assume it. calculate() overwrites the family's binning with 0 when IBSI is on,
// and at 0 the two expressions that still read the flag resolve to the same numbers as the ones
// beside them: I is built as the contiguous run 1..max, so the position of a level in it is level-1,
// which is what the IBSI row index computes, and I.size() is max(I), which is what the IBSI Ng
// computes.
//
// It runs on the gapped-level fixture because that is where the two would part company if the
// reasoning were wrong -- with levels 1, 3, 5 the position of a level and the level itself are
// different numbers. The comparison is exact: one code path on one input, so anything short of bit
// equality is a difference in behaviour. GLSZM_GREYDEPTH is passed as 64 on the IBSI side, which a
// run that honoured it would bin into 64 MATLAB levels, so the overwrite is measured too.
void test_3d_glszm_ibsi_equals_no_binning_mechanics()
{
	std::vector<std::vector<double>> ibsi_vals, plain_vals;
	D3_GLSZM_feature f_ibsi, f_plain;

	Fsettings s_ibsi = make_glszm3d_settings (64/*greydepth*/, 64/*overwritten with 0*/, true/*ibsi*/);
	ASSERT_NO_FATAL_FAILURE(run_3d_glszm_on_volume (
		ibsi_vals, glszm_3d_gapped_volume, 3/*width*/, 3/*height*/, 3/*depth*/, s_ibsi, f_ibsi));

	Fsettings s_plain = make_glszm3d_settings (64/*greydepth*/, 0/*no binning*/, false/*ibsi*/);
	ASSERT_NO_FATAL_FAILURE(run_3d_glszm_on_volume (
		plain_vals, glszm_3d_gapped_volume, 3/*width*/, 3/*height*/, 3/*depth*/, s_plain, f_plain));

	for (auto fc : D3_GLSZM_feature::featureset)
	{
		SCOPED_TRACE ("3D feature code " + std::to_string ((int)fc));
		ASSERT_EQ (ibsi_vals[(int)fc][0], plain_vals[(int)fc][0]);
	}

	// and the tables under them, so what is equal is the matrix and not only sixteen sums over it
	ASSERT_EQ (f_ibsi.get_Ng(), f_plain.get_Ng());
	ASSERT_EQ (f_ibsi.get_Ns(), f_plain.get_Ns());
	ASSERT_EQ (f_ibsi.get_Nz(), f_plain.get_Nz());
	ASSERT_EQ (f_ibsi.get_Np(), f_plain.get_Np());
	ASSERT_EQ (f_ibsi.get_P(), f_plain.get_P());
}
