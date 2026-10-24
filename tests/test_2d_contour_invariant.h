#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <set>
#include <vector>
#include "test_2d_contour_common.h"

// The ROI pixels with a 4-neighbour outside the ROI: the edge every contour consumer is defined
// on, and the set skimage.segmentation.find_boundaries(connectivity=1, mode='inner') and
// CellProfiler's edge pixels are.
static std::set<std::pair<int, int>> contour_boundary_rule (const std::set<std::pair<int, int>>& S)
{
	std::set<std::pair<int, int>> B;
	for (const auto& p : S)
	{
		int x = p.first, y = p.second;
		if (!S.count({ x - 1, y }) || !S.count({ x + 1, y }) || !S.count({ x, y - 1 }) || !S.count({ x, y + 1 }))
			B.insert(p);
	}
	return B;
}

// Required of every ROI, however irregular: each contour is a closed walk in 8-connected steps
// over ROI pixels, and together the contours cover exactly the ROI's 4-adjacent edge pixels. Leaves
// in 'walks' the summed closed-walk length, which PERIMETER has to equal, and in 'edge_sum' the
// intensity summed over the edge pixels, each once, which EDGE_INTEGRATED_INTENSITY has to equal.
static void assert_contour_boundary_rule (const std::set<std::pair<int, int>>& S, LR& r, double& walks, double& edge_sum)
{
	contour_roi_from_pixels (r, S);
	contour_calculate (r);

	walks = 0;
	for (const auto& K : r.multicontour_)
	{
		ASSERT_TRUE (contour_is_closed_8_walk (K));
		for (const Pixel2& p : K)
			ASSERT_TRUE (S.count({ (int)p.x, (int)p.y })) << "contour pixel (" << p.x << "," << p.y << ") is not a ROI pixel";
		walks += contour_closed_walk_length (K);
	}

	const auto B = contour_boundary_rule (S);
	ASSERT_EQ (contour_pixel_set (r), B);

	edge_sum = 0;
	for (const auto& p : B)
		edge_sum += contour_fixture_intensity (p.first, p.second);
}

// 400 random masks, 3..39 pixels a side and 30..90 % filled, so they carry holes, fragments,
// one-pixel necks and isolated pixels. Seeded, so a failure names a reproducible case.
void test_2d_contour_noisy_rois_boundary_rule_invariant()
{
	std::mt19937 rng (20261022);
	std::uniform_int_distribution<int> side (3, 39), origin (0, 200);
	std::uniform_real_distribution<double> u (0.0, 1.0);
	int n_tested = 0;
	for (int i = 0; i < 400; i++)
	{
		const int w = side(rng), h = side(rng), x0 = origin(rng), y0 = origin(rng);
		const double fill = 0.3 + 0.6 * u(rng);
		std::set<std::pair<int, int>> S;
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
				if (u(rng) < fill)
					S.insert({ x0 + x, y0 + y });
		if (S.empty())
			continue;

		SCOPED_TRACE("noisy ROI #" + std::to_string(i));
		LR r (1);
		double walks, edge_sum;
		assert_contour_boundary_rule (S, r, walks, edge_sum);
		if (::testing::Test::HasFatalFailure())
			return;
		ASSERT_NEAR (r.fvals[(int)Nyxus::Feature2D::PERIMETER][0], walks, 1e-9);
		ASSERT_NEAR (r.fvals[(int)Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY][0], edge_sum, 1e-9);
		n_tested++;
	}
	ASSERT_GE (n_tested, 390);
}

// The filled 9066-pixel blob: one border covering its 495 edge pixels.
void test_2d_contour_blob97_boundary_rule_invariant()
{
	const auto S = contour_mask_pixels (bench_blob97_filled_rows, 113, 41);
	ASSERT_EQ (S.size(), size_t(9066));
	ASSERT_EQ (contour_boundary_rule (S).size(), size_t(495));

	LR r (1);
	double walks, edge_sum;
	assert_contour_boundary_rule (S, r, walks, edge_sum);
	if (::testing::Test::HasFatalFailure())
		return;
	ASSERT_EQ (r.multicontour_.size(), size_t(1));
	ASSERT_NEAR (r.fvals[(int)Nyxus::Feature2D::PERIMETER][0], walks, 1e-9);
	ASSERT_NEAR (r.fvals[(int)Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY][0], edge_sum, 1e-9);
}

// Pixel2::min_sqdist and max_sqdist read a contour as one ordered walk. A walk of up to 10 points is
// scanned whole, so on it both return exactly what an exhaustive scan does, at any pixel size. Every
// walk length from 1 to 10, on points scattered at random so that the distance along a walk has
// several extremes, each walk queried from every point of a grid around it. A longer walk is searched
// coarse to fine and may miss its extreme, which is not asserted here.
void test_2d_contour_distance_search_short_walk_invariant()
{
	std::mt19937 rng (20261023);
	std::uniform_int_distribution<int> coord (0, 12);
	for (size_t n = 1; n <= 10; n++)
		for (int trial = 0; trial < 40; trial++)
		{
			std::vector<Pixel2> W;
			for (size_t i = 0; i < n; i++)
				W.push_back (Pixel2 (coord(rng), coord(rng), 1u));

			for (int qy = -1; qy <= 13; qy += 2)
				for (int qx = -1; qx <= 13; qx += 2)
				{
					const Pixel2 q (qx, qy, 0u);
					for (const auto& [sx, sy] : { std::pair<double, double>{ 1.0, 1.0 }, { 2.0, 3.0 } })
					{
						double mn = q.sqdist (W[0], sx, sy), mx = mn;
						for (const Pixel2& p : W)
						{
							mn = std::min (mn, q.sqdist (p, sx, sy));
							mx = std::max (mx, q.sqdist (p, sx, sy));
						}
						SCOPED_TRACE ("walk of " + std::to_string(n) + " points, trial " + std::to_string(trial) + ", query (" + std::to_string(qx) + "," + std::to_string(qy) + "), pixel " + std::to_string(sx) + " x " + std::to_string(sy));
						ASSERT_EQ (q.min_sqdist (W, sx, sy), mn);
						ASSERT_EQ (q.max_sqdist (W, sx, sy), mx);
					}
				}
		}

	// no contour at all: both searches report 0
	const Pixel2 q (3, 4, 0u);
	ASSERT_EQ (q.min_sqdist (std::vector<Pixel2>()), 0.0);
	ASSERT_EQ (q.max_sqdist (std::vector<Pixel2>()), 0.0);
}

// Pixel2::min_sqdist and max_sqdist over a contour of several borders search each border on its own
// and skip a border whose bounding box cannot beat the extreme found so far. Skipping must not change
// the answer: it equals the smallest, or largest, of the searches of every border as a walk of its
// own. 100 random masks with holes and fragments, every pixel of each queried, at two pixel sizes.
void test_2d_contour_distance_search_borders_invariant()
{
	std::mt19937 rng (20261024);
	std::uniform_int_distribution<int> side (4, 30);
	std::uniform_real_distribution<double> u (0.0, 1.0);
	size_t n_multi = 0;
	for (int i = 0; i < 100; i++)
	{
		const int w = side(rng), h = side(rng);
		const double fill = 0.4 + 0.5 * u(rng);
		std::set<std::pair<int, int>> S;
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
				if (u(rng) < fill)
					S.insert({ 10 + x, 20 + y });
		if (S.empty())
			continue;

		LR r (1);
		contour_roi_from_pixels (r, S);
		contour_calculate (r);
		const BorderedContour Kb (r.contour_pixels_, r.contour_border_ends_);
		n_multi += Kb.borders.size() > 1;

		ASSERT_TRUE (border_ends_cover (r.contour_pixels_.size(), r.contour_border_ends_));
		std::vector<std::vector<Pixel2>> borders;
		size_t begin = 0;
		for (size_t end : r.contour_border_ends_)
		{
			borders.emplace_back (r.contour_pixels_.begin() + begin, r.contour_pixels_.begin() + end);
			begin = end;
		}
		ASSERT_EQ (borders.size(), Kb.borders.size());

		for (const auto& [sx, sy] : { std::pair<double, double>{ 1.0, 1.0 }, { 2.0, 3.0 } })
			for (const Pixel2& q : r.raw_pixels)
			{
				double mn = q.min_sqdist (borders[0], sx, sy), mx = q.max_sqdist (borders[0], sx, sy);
				for (const auto& B : borders)
				{
					mn = std::min (mn, q.min_sqdist (B, sx, sy));
					mx = std::max (mx, q.max_sqdist (B, sx, sy));
				}
				SCOPED_TRACE ("mask #" + std::to_string(i) + ", pixel (" + std::to_string(q.x) + "," + std::to_string(q.y) + ")");
				ASSERT_EQ (q.min_sqdist (Kb, sx, sy), mn);
				ASSERT_EQ (q.max_sqdist (Kb, sx, sy), mx);
			}
	}
	ASSERT_GE (n_multi, (size_t)50) << "too few masks had more than one border to test the skipping";
}
