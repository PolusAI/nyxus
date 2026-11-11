#pragma once

// CHARACTERIZATION of the 2D radial intensity distribution's plumbing: the coordinate frame the
// traced contour comes back in, the centre pixel the feature measures from, and the radius it
// normalises by. The first two cases assert that plumbing; the third asserts its effect on the
// feature outputs; the last three cover the one- and two-pixel ROIs, whose radii are 0 and 1, and a
// ROI of two fragments, whose borders are searched one at a time. The file is diagnostic
// characterization and claims no oracle.
//
// The first case and the last three are requirements: the contour is in the frame of the ROI's own
// pixels, the tiny ROIs land in the first and last bins with finite values, and the centre and
// radius of the two fragments are the ones exhaustive scans give. The second and third case
// pin behaviour that tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md §6 argues is wrong
// - the short radius^2 10 (defect 2) and the num_bins-1 bin scaling (defect 3) - so read their
// numbers as a record of what the tree does today, NOT as acceptance criteria. A correction should
// invalidate the assertions tied to the defect it fixes; the corrective work and its preconditions
// are tracked in tests/vetting/audit/radial_2d_golden_regen.md §5, which is the fix branch's
// checklist.
//
// The point of pinning them is that the 24 public values in test_2d_radial_regression.h are decided
// entirely by these three pieces of wiring and nothing else in the tree looks at them, so without
// these cases a fix - or an accident - moves the goldens with no statement of which of the three
// changed. That is a diagnostic role, which is why the registry credits this file for no feature:
// the three radial rows list test_2d_radial_{regression,invariant}.h and not this one, and
// tests/vetting/not_covered.md §A.1 records the omission as deliberate.

#include <gtest/gtest.h>

#include <algorithm>                          // std::max

#include "../src/nyx/featureset.h"             // Feature2D
#include "../src/nyx/features/pixel.h"         // Pixel2, StatsInt
#include "test_2d_radial_common.h"             // build_radial_2d_roi, LR, <vector>;
                                               // <cmath> for std::sqrt via test_main_nyxus.h

// The contour is in the frame of the ROI's own pixels: RadialDistributionFeature measures distances
// from raw_pixels to the contour, so every contour pixel has to be one of them. Here they are the
// 18 edge pixels of the outer border and the border of the one-pixel hole.
void test_2d_radial_contour_frame_mechanics()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);

	const std::vector<Pixel2>& K = roidata.contour_pixels_;
	ASSERT_EQ(roidata.raw_pixels.size(), (size_t)26);
	ASSERT_EQ(roidata.multicontour_.size(), (size_t)2);
	ASSERT_EQ(roidata.contour_border_ends_.size(), (size_t)2);
	ASSERT_EQ(K.size(), (size_t)18);

	auto in_roi = [&roidata](StatsInt x, StatsInt y) {
		for (const auto& p : roidata.raw_pixels)
			if (p.x == x && p.y == y)
				return true;
		return false;
	};

	for (const auto& c : K)
		ASSERT_TRUE(in_roi(c.x, c.y)) << "contour pixel (" << c.x << "," << c.y << ") is not a ROI pixel";
}

// Defect 2. Pixel2::find_center and Pixel2::max_sqdist search each border of the contour coarse to
// fine, and the coarse pass can step over a border's extreme. On this ROI the centre search lands
// on (2,3), the pixel the exact scan picks too, but the radius the feature normalises by is 23%
// short of the largest distance actually present: the outer border is a walk of 14 pixels, sampled
// every 5th from its first, and the farthest pixel, (4,6), lies between the samples.
// radius^2 10 below is the search's current wrong answer; the exact scan in the same case computes
// 13. A fix makes the two agree and fails this case.
void test_2d_radial_center_and_radius_mechanics()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);

	const std::vector<Pixel2>& K = roidata.contour_pixels_;
	const BorderedContour Kb(K, roidata.contour_border_ends_);

	int idxO = Pixel2::find_center(roidata.raw_pixels, Kb);
	const Pixel2& pxO = roidata.raw_pixels[idxO];
	ASSERT_EQ(pxO.x, 2) << "the exact scan below picks (2,3)";
	ASSERT_EQ(pxO.y, 3) << "the exact scan below picks (2,3)";

	// the searched maximum against a full linear scan of the same contour
	double exact_max = 0.0;
	for (const auto& p : K)
		exact_max = std::max(exact_max, pxO.sqdist(p));
	ASSERT_DOUBLE_EQ(pxO.max_sqdist(Kb), 10.0)
		<< "known-defect pin (report §6 defect 2): the exact scan of the same contour gives 13";
	ASSERT_DOUBLE_EQ(exact_max, 13.0);

	// the centre a full linear scan of the same criterion (min of max-min squared distance) picks
	int exact_idx = 0;
	double best = -1.0;
	for (size_t i = 0; i < roidata.raw_pixels.size(); i++)
	{
		const Pixel2& q = roidata.raw_pixels[i];
		double mn = -1.0, mx = 0.0;
		for (const auto& p : K)
		{
			double d = q.sqdist(p);
			mx = std::max(mx, d);
			mn = (mn < 0.0 || d < mn) ? d : mn;
		}
		if (best < 0.0 || mx - mn < best)
		{
			best = mx - mn;
			exact_idx = (int)i;
		}
	}
	ASSERT_EQ(roidata.raw_pixels[exact_idx].x, 2);
	ASSERT_EQ(roidata.raw_pixels[exact_idx].y, 3);
}

// Defect 3. The bin index is int(r / r_max * (num_bins - 1)), so the 8 bins are 7 equal-width rings
// plus a last bin that only r >= r_max reaches. Here r_max is the short radius pinned above, so the
// last bin holds the 3 pixels sitting exactly on it plus the 1 that lies beyond it - 4 of the 26.
void test_2d_radial_bin_index_mechanics()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);

	const BorderedContour Kb(roidata.contour_pixels_, roidata.contour_border_ends_);
	const Pixel2& pxO = roidata.raw_pixels[Pixel2::find_center(roidata.raw_pixels, Kb)];
	const double r_max = std::sqrt(pxO.max_sqdist(Kb));

	size_t at_r_max = 0, past_r_max = 0;
	for (const auto& p : roidata.raw_pixels)
	{
		const double rat = std::sqrt(p.sqdist(pxO)) / r_max;
		if (rat > 1.0)
			past_r_max++;
		else if (rat == 1.0)
			at_r_max++;
	}
	ASSERT_EQ(past_r_max, (size_t)1)
		<< "known-defect pin (report §6 defects 2-3): a correct radius leaves no pixel past r_max";
	ASSERT_EQ(at_r_max, (size_t)3)
		<< "known-defect pin (report §6 defects 2-3)";

	// and those 7 together are what the last bin of all three features is computed from
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];
	const auto& cv = roidata.fvals[(int)Nyxus::Feature2D::RADIAL_CV];
	const double last_bin_count = frac.back() * double(roidata.raw_pixels.size());
	ASSERT_NEAR(last_bin_count, double(at_r_max + past_r_max), 1e-6) << "FRAC_AT_D last bin";
	ASSERT_GT(mean.back(), 0.0) << "MEAN_FRAC last bin is fed by the clamped pixels";
	ASSERT_GT(cv.back(), 0.0) << "RADIAL_CV last bin is fed by the clamped pixels";
}

// A one-pixel ROI is its own contour, so the radius the feature normalises by is 0. The pixel is the
// centre, at distance 0, and lands in the first bin: FRAC_AT_D is 1 there and 0 elsewhere, and every
// value is finite.
void test_2d_radial_single_pixel_roi_mechanics()
{
	const NyxusPixel px[] = { { 5, 7, 42 } };
	LR roidata(102);
	load_masked_test_roi_data(roidata, px, px, 1);
	roidata.initialize_fvals();

	Fsettings s = make_shape2d_settings();
	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);
	ASSERT_EQ(roidata.multicontour_.size(), (size_t)1);
	ASSERT_EQ(roidata.multicontour_[0].size(), (size_t)1);

	RadialDistributionFeature radial;
	radial.calculate(roidata, s);
	radial.save_value(roidata.fvals);

	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];
	const auto& cv = roidata.fvals[(int)Nyxus::Feature2D::RADIAL_CV];
	ASSERT_EQ(frac.size(), (size_t)RadialDistributionFeature::num_bins);
	for (size_t i = 0; i < frac.size(); i++)
	{
		ASSERT_TRUE(std::isfinite(frac[i]) && std::isfinite(mean[i]) && std::isfinite(cv[i])) << "bin " << i;
		ASSERT_NEAR(frac[i], i == 0 ? 1.0 : 0.0, 1e-6) << "FRAC_AT_D[" << i << "]";
	}
	ASSERT_NEAR(mean[0], 42.0, 1e-6);
}

// A two-pixel ROI: both pixels are the contour, a border of two points. Each pixel is 1 from the
// other, so whichever is the centre, the radius is 1, not 0: the centre lands in the first bin and
// the other pixel, at r == r_max, in the last.
void test_2d_radial_two_pixel_roi_mechanics()
{
	const NyxusPixel px[] = { { 5, 7, 40 }, { 6, 7, 60 } };
	LR roidata(103);
	load_masked_test_roi_data(roidata, px, px, 2);
	roidata.initialize_fvals();

	Fsettings s = make_shape2d_settings();
	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);
	ASSERT_EQ(roidata.multicontour_.size(), (size_t)1);

	RadialDistributionFeature radial;
	radial.calculate(roidata, s);
	radial.save_value(roidata.fvals);

	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];
	const int n = RadialDistributionFeature::num_bins;
	ASSERT_EQ(frac.size(), (size_t)n);
	for (int i = 0; i < n; i++)
		ASSERT_NEAR(frac[i], i == 0 || i == n - 1 ? 0.5 : 0.0, 1e-6) << "FRAC_AT_D[" << i << "]";
	ASSERT_NEAR(mean[0] + mean[n - 1], 100.0, 1e-6) << "one pixel in each of the two bins";
	ASSERT_NEAR(std::abs(mean[0] - mean[n - 1]), 20.0, 1e-6);
}

// Two 3x3 squares 4 pixels apart under one label: two borders of 8 pixels each, the outer borders of
// the two fragments. Pixel2::find_center and Pixel2::max_sqdist search every border on its own, and
// a border of up to 10 pixels is scanned whole, so here both return exactly what exhaustive scans of
// the contour give: the centre (3,2), the middle of the left square's right column, and the squared
// radius 50, to the far corners of the right square. All 8 FRAC_AT_D entries are rebuilt from that
// centre and radius. Searched as one 16-pixel sequence, the contour gives the centre (8,1) and 49.
void test_2d_radial_two_fragments_mechanics()
{
	std::vector<NyxusPixel> px;
	for (StatsInt y = 1; y <= 3; y++)
		for (StatsInt x : { 1, 2, 3, 8, 9, 10 })
			px.push_back({ (size_t)x, (size_t)y, (unsigned int)(10 + x + 3 * y) });
	LR roidata(104);
	load_masked_test_roi_data(roidata, px.data(), px.data(), px.size());
	roidata.initialize_fvals();

	Fsettings s = make_shape2d_settings();
	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);
	ASSERT_EQ(roidata.multicontour_.size(), (size_t)2);
	ASSERT_EQ(roidata.contour_border_ends_, (std::vector<size_t>{ 8, 16 }));

	RadialDistributionFeature radial;
	radial.calculate(roidata, s);
	radial.save_value(roidata.fvals);

	// the centre by the feature's own criterion, the least (max - min) squared distance to the
	// contour, by exhaustive scans; the first such pixel wins a tie, as it does in the feature
	const auto& K = roidata.contour_pixels_;
	const auto& cloud = roidata.raw_pixels;
	size_t idxO = 0;
	double best = -1.0;
	for (size_t i = 0; i < cloud.size(); i++)
	{
		double mn = -1.0, mx = 0.0;
		for (const auto& p : K)
		{
			const double d = cloud[i].sqdist(p);
			mx = std::max(mx, d);
			mn = (mn < 0.0 || d < mn) ? d : mn;
		}
		if (best < 0.0 || mx - mn < best)
		{
			best = mx - mn;
			idxO = i;
		}
	}
	const Pixel2& pxO = cloud[idxO];
	double r_max2 = 0.0;
	for (const auto& p : K)
		r_max2 = std::max(r_max2, pxO.sqdist(p));
	ASSERT_EQ(pxO.x, 3);
	ASSERT_EQ(pxO.y, 2);
	ASSERT_EQ(r_max2, 50.0);

	// the searches the feature runs
	const BorderedContour Kb(K, roidata.contour_border_ends_);
	const Pixel2& pxS = cloud[Pixel2::find_center(cloud, Kb)];
	ASSERT_EQ(pxS.x, 3);
	ASSERT_EQ(pxS.y, 2);
	ASSERT_EQ(pxS.max_sqdist(Kb), 50.0);

	const int n = RadialDistributionFeature::num_bins;
	std::vector<double> counts(n, 0.0);
	for (const auto& p : cloud)
	{
		int bi = int(std::sqrt(p.sqdist(pxO)) / std::sqrt(r_max2) * (n - 1));
		counts[std::min(bi, n - 1)] += 1.0;
	}
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	ASSERT_EQ(frac.size(), (size_t)n);
	for (int i = 0; i < n; i++)
		ASSERT_NEAR(frac[i] * double(cloud.size()), counts[i], 1e-6) << "FRAC_AT_D[" << i << "]";
}
