#pragma once

#include <gtest/gtest.h>

#include "test_imq_common.h"                     // fixture: calc_imq_feature_on_spikes, imq_soft_nan, <vector>, <utility>, and FeatureIMQ via featureset.h
#include "../src/nyx/features/focus_score.h"     // FocusScoreFeature

// LOCAL_FOCUS_SCORE conventions Nyxus defines for itself (SPEC 2, mechanics): which pixels belong to
// no tile, and what an ROI with no room for a tile returns. Neither is a closed form or another
// tool's output, so these pin behaviour without vetting it. The tile scores themselves are vetted in
// test_imq_analytic.h and test_imq_opencv.h.
//
// get_local_focus_score() cuts the ROI into a scale x scale grid of (height/scale) x (width/scale)
// tiles; calculate() uses scale = 2.

static const double imq_mechanics_abs_tolerance = 0.0;

// The last width % scale columns and height % scale rows belong to no tile. On a 9 x 13 ROI the
// tiles are 4 x 6, so column 8 and row 12 are outside the grid; spikes only there leave every tile
// all zero, and the score is 0.
void test_imq_local_focus_score_remainder_mechanics()
{
	std::vector<std::pair<size_t, size_t>> spikes = { {8, 3}, {2, 12} };

	ASSERT_NEAR (calc_imq_feature_on_spikes<FocusScoreFeature> (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, 9, 13, spikes),
		0.0, imq_mechanics_abs_tolerance);
}

// An ROI one pixel thin on either side is shorter than scale = 2, so height / scale or width / scale
// is 0 and no tile fits. The score is undefined and reads as the soft-NaN setting, which the fixture
// sets to a value no score can take, so the assertion tells it from a computed 0. FOCUS_SCORE needs
// no tile and is still computed.
void test_imq_local_focus_score_thin_roi_mechanics()
{
	{
		SCOPED_TRACE ("MECHANICS__LOCAL_FOCUS_SCORE 5 x 1");
		ASSERT_EQ (calc_imq_feature_on_spikes<FocusScoreFeature> (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, 5, 1, {{2, 0}}), imq_soft_nan);
		ASSERT_GT (calc_imq_feature_on_spikes<FocusScoreFeature> (Nyxus::FeatureIMQ::FOCUS_SCORE, 5, 1, {{2, 0}}), 0.0);
	}
	{
		SCOPED_TRACE ("MECHANICS__LOCAL_FOCUS_SCORE 1 x 5");
		ASSERT_EQ (calc_imq_feature_on_spikes<FocusScoreFeature> (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, 1, 5, {{0, 2}}), imq_soft_nan);
		ASSERT_GT (calc_imq_feature_on_spikes<FocusScoreFeature> (Nyxus::FeatureIMQ::FOCUS_SCORE, 1, 5, {{0, 2}}), 0.0);
	}
}

// The other side of that guard: a 2 x 2 ROI is the smallest a 2 x 2 grid fits, with 1 x 1 tiles.
// A one-pixel tile's Laplacian is a single value, whose variance is 0, so the score is a defined 0
// and not the soft-NaN.
void test_imq_local_focus_score_smallest_tiled_roi_mechanics()
{
	ASSERT_NEAR (calc_imq_feature_on_spikes<FocusScoreFeature> (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, 2, 2, {{0, 0}}),
		0.0, imq_mechanics_abs_tolerance);
}
