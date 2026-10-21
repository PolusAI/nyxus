#pragma once

// Invariants of the 3D GLRLM family: bounds its features hold by construction, whatever the
// configuration. Claims no oracle (SPEC 2).

#include "test_3d_glrlm_regression.h"	// gtest, the environment, run_3d_glrlm_pipeline() on the ut_ phantom

// Run Percentage is runs over the ROI's voxels, short-run emphasis averages 1/j^2, and the two
// normalised non-uniformities divide a sum of squares by the square of its sum, so all four lie in
// [0, 1]; long-run emphasis averages j^2 and is at least 1. Each holds per direction, so every one of
// the 13 directions is checked rather than their mean, which can stay in range while single
// directions leave it.
//
// The sweep crosses all three binning schemes: PyRadiomics bin counts, no binning, and MATLAB levels.
// The ut_ ROI leaves background inside its bounding box, and MATLAB binning sends intensity 0 to
// level 1, the level the ROI's darkest voxels carry, so the MATLAB depths are where a run counted
// outside the ROI would push RP past 1.
void test_3d_glrlm_bounded_features_invariant()
{
	using F = Nyxus::Feature3D;

	for (int depth : { -100, -20, 0, 32, 64, 128, 256 })
	{
		SCOPED_TRACE ("GLRLM_GREYDEPTH=" + std::to_string (depth));
		Environment e;
		int label = -1;
		run_3d_glrlm_pipeline (e, depth, depth, label);
		if (::testing::Test::HasFatalFailure())
			return;
		const auto& fv = e.roiData[label].fvals;

		for (F fc : { F::GLRLM_RP, F::GLRLM_SRE, F::GLRLM_RLNN, F::GLRLM_GLNN })
		{
			const std::vector<double>& v = fv[(int) fc];
			ASSERT_EQ (v.size(), 13u) << "feature " << (int) fc;
			for (size_t a = 0; a < v.size(); a++)
			{
				EXPECT_GE (v[a], 0.0) << "feature " << (int) fc << ", direction " << a;
				EXPECT_LE (v[a], 1.0) << "feature " << (int) fc << ", direction " << a;
			}
		}

		const std::vector<double>& lre = fv[(int) F::GLRLM_LRE];
		ASSERT_EQ (lre.size(), 13u);
		for (size_t a = 0; a < lre.size(); a++)
			EXPECT_GE (lre[a], 1.0) << "LRE, direction " << a;
	}
}
