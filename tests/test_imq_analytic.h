#pragma once

#include <gtest/gtest.h>

#include "test_imq_common.h"                     // fixture: calc_imq_feature_on, imq_all_ones_mask, <vector>, and FeatureIMQ via featureset.h
#include "../src/nyx/features/focus_score.h"     // FocusScoreFeature

// Closed-form checks of the focus-score tiling and kernel (SPEC 4, oracle=analytic). The opencv file
// vets the two scores on the im_quality fixture; what it cannot isolate is WHICH tiles
// LOCAL_FOCUS_SCORE averages, because every tile of that fixture carries texture. These ROIs put the
// texture in one place at a time, so each tile's contribution is asserted on its own.
//
// The probe is a single spike of height v in an otherwise-zero tile of P pixels, at least one pixel
// clear of the tile edge. Its ksize=1 Laplacian is -4v at the spike and +v at its four neighbours,
// 0 everywhere else, so the mean is exactly 0 and the population variance is
//
//     (16 v^2 + 4 v^2) / P = 20 v^2 / P.
//
// Every other tile is all zero and scores 0, so LOCAL_FOCUS_SCORE = 20 v^2 / (P * scale^2). On every
// ROI below, scale=2 makes the tiles 4 wide and 6 tall (P = 24), and v = 6 gives
// 20 * 36 / 24 / 4 = 7.5. All the arithmetic is on small integers, so the value is exact in double
// and the band is 0: any other result is a different tiling, not a rounding difference.
//
// The ROI sizes cover both parities on both axes, 8 or 9 wide by 12 or 13 tall: all four have the
// same 2 x 2 grid of 4 x 6 tiles, and the odd sides add one column or row that belongs to no tile. An
// even side is the one a loop bound of `y < height - M` cuts short - it stops after the first tile -
// while an odd side lets that bound reach the second tile anyway, so only an even side tells the
// two bounds apart.

static const double imq_analytic_abs_tolerance = 0.0;

static const std::vector<std::pair<size_t, size_t>> imq_tile_roi_sizes = { {8, 12}, {9, 12}, {8, 13}, {9, 13} };
static const size_t imq_tile_roi_w = 9,     // the odd-by-odd size: 2 tiles of 4 columns, plus 1 column in no tile
                    imq_tile_roi_h = 13;    // 2 tiles of 6 rows, plus 1 row in no tile
static const size_t imq_tile_w = 4,
                    imq_tile_h = 6;
static const unsigned int imq_spike = 6;

// 20 v^2 / (P * scale^2) with P = 4 * 6 and scale = 2
static const double imq_one_spike_local_focus_score = 20.0 * imq_spike * imq_spike / (imq_tile_w * imq_tile_h) / 4.0;

// A w x h ROI, 0 everywhere except the listed (x, y) cells, which hold imq_spike. The mask covers the
// whole rectangle, so the ROI image matrix is exactly this array.
static std::vector<NyxusPixel> imq_spikes_intensity (size_t w, size_t h, const std::vector<std::pair<size_t, size_t>>& spikes)
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++)
		{
			unsigned int v = 0;
			for (const auto& s : spikes)
				if (s.first == x && s.second == y)
					v = imq_spike;
			px.push_back (NyxusPixel {x, y, v});
		}
	return px;
}

// The feature is passed at the call site, on the assertion line, because
// tests/vetting/audit/scan_imq_coverage.py attributes coverage from the line that asserts
static double calc_imq_focus_on_spikes (Nyxus::FeatureIMQ feature, size_t w, size_t h,
	const std::vector<std::pair<size_t, size_t>>& spikes)
{
	std::vector<NyxusPixel> intensity = imq_spikes_intensity (w, h, spikes),
		mask = imq_all_ones_mask (w, h);
	return calc_imq_feature_on<FocusScoreFeature> (feature, intensity.data(), mask.data(), mask.size());
}

// Every one of the scale^2 = 4 tiles is averaged. The spike sits at tile-local (col 1, row 2), clear
// of every tile edge, and moves through the four tiles in turn; each placement must score the same
// 20 v^2 / (P * 4). A tiling that skips a tile scores that placement 0.
void test_imq_local_focus_score_each_tile_analytic()
{
	for (const auto& roi : imq_tile_roi_sizes)
		for (size_t ty = 0; ty < 2; ty++)
			for (size_t tx = 0; tx < 2; tx++)
			{
				size_t x = tx * imq_tile_w + 1,
					y = ty * imq_tile_h + 2;
				SCOPED_TRACE ("ANALYTIC__LOCAL_FOCUS_SCORE " + std::to_string(roi.first) + " x " + std::to_string(roi.second)
					+ " ROI, tile (" + std::to_string(tx) + ", " + std::to_string(ty) + ")");
				ASSERT_NEAR (calc_imq_focus_on_spikes (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, roi.first, roi.second, {{x, y}}),
					imq_one_spike_local_focus_score, imq_analytic_abs_tolerance);
			}
}

// Tiles average, they do not add up to more than one tile's worth: a spike in every tile scores
// 4 * 20 v^2 / (P * 4), i.e. one tile's full score.
void test_imq_local_focus_score_all_tiles_analytic()
{
	std::vector<std::pair<size_t, size_t>> spikes;
	for (size_t ty = 0; ty < 2; ty++)
		for (size_t tx = 0; tx < 2; tx++)
			spikes.push_back ({tx * imq_tile_w + 1, ty * imq_tile_h + 2});

	for (const auto& roi : imq_tile_roi_sizes)
	{
		SCOPED_TRACE ("ANALYTIC__LOCAL_FOCUS_SCORE " + std::to_string(roi.first) + " x " + std::to_string(roi.second) + " ROI");
		ASSERT_NEAR (calc_imq_focus_on_spikes (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, roi.first, roi.second, spikes),
			4.0 * imq_one_spike_local_focus_score, imq_analytic_abs_tolerance);
	}
}

// The last width % scale columns and height % scale rows belong to no tile. Spikes only there - in
// column 8 and in row 12 of the 9 x 13 ROI - leave every tile all zero, so the score is 0.
void test_imq_local_focus_score_remainder_analytic()
{
	std::vector<std::pair<size_t, size_t>> spikes = { {8, 3}, {2, 12} };

	ASSERT_NEAR (calc_imq_focus_on_spikes (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, imq_tile_roi_w, imq_tile_roi_h, spikes),
		0.0, imq_analytic_abs_tolerance);
}

// An ROI one pixel thin on either side is shorter than scale = 2, so height / scale or width / scale
// is 0 and no tile fits. The score is 0 - there is nothing to average - and the call returns.
void test_imq_local_focus_score_thin_roi_analytic()
{
	SCOPED_TRACE ("ANALYTIC__LOCAL_FOCUS_SCORE 5 x 1");
	ASSERT_NEAR (calc_imq_focus_on_spikes (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, 5, 1, {{2, 0}}), 0.0, imq_analytic_abs_tolerance);

	SCOPED_TRACE ("ANALYTIC__LOCAL_FOCUS_SCORE 1 x 5");
	ASSERT_NEAR (calc_imq_focus_on_spikes (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, 1, 5, {{0, 2}}), 0.0, imq_analytic_abs_tolerance);
}

// laplacian() of a spike is the kernel itself, centred on the spike. A call with ksize != 1 selects
// the {{2,0,2},{0,-8,0},{2,0,2}} kernel for that call only: the ksize=1 call after it must still see
// {{0,1,0},{1,-4,1},{0,1,0}}.
void test_imq_focus_score_kernel_per_call_analytic()
{
	const int n = 5;
	std::vector<PixIntens> spike (n * n, 0);
	spike[2 * n + 2] = 1;

	std::vector<double> wide (n * n, 0.);
	FocusScoreFeature::laplacian (spike, wide, n, n, 3);
	const double wide_expected[9] = { 2, 0, 2,  0, -8, 0,  2, 0, 2 };
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			ASSERT_EQ (wide[(r + 1) * n + (c + 1)], wide_expected[r * 3 + c]) << "ksize=3 at " << r << "," << c;

	std::vector<double> plain (n * n, 0.);
	FocusScoreFeature::laplacian (spike, plain, n, n, 1);
	const double plain_expected[9] = { 0, 1, 0,  1, -4, 1,  0, 1, 0 };
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			ASSERT_EQ (plain[(r + 1) * n + (c + 1)], plain_expected[r * 3 + c]) << "ksize=1 at " << r << "," << c;
}
