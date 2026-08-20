#pragma once

// Plumbing behind the 2D radial intensity distribution: the coordinate frame the traced contour
// comes back in, the centre pixel the feature measures from, and the radius it normalises by. None
// of it is a feature value and none of it claims an oracle (SPEC 2) - these are drift guards on
// three pieces of wiring that the three pinned feature vectors depend on completely and that no
// other test in the tree looks at.
//
// Two of the three record behaviour that tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md
// argues is wrong. They are pinned rather than corrected because a correction moves public feature
// values and belongs on its own branch; pinning them here is what makes it impossible to land
// silently.

#include <gtest/gtest.h>

#include <algorithm>                          // std::max

#include "../src/nyx/featureset.h"             // Feature2D
#include "../src/nyx/features/pixel.h"         // Pixel2, StatsInt
#include "test_2d_radial_common.h"             // build_radial_2d_roi, LR, <vector>;
                                               // <cmath> for std::sqrt via test_main_nyxus.h

// ContourFeature traces on an image padded by one pixel on every side and adds the ROI's bounding-box
// origin back afterwards, but never subtracts that one-pixel pad. Every traced contour pixel is
// therefore reported one pixel right and one pixel down of where it is, which puts 7 of this ROI's 18
// contour pixels outside the ROI altogether. RadialDistributionFeature measures distances from
// raw_pixels (untranslated) to this contour, so it mixes the two frames.
void test_2d_radial_contour_frame_mechanics()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);

	std::vector<Pixel2> K;
	roidata.merge_multicontour(K);
	ASSERT_EQ(roidata.raw_pixels.size(), (size_t)26);
	ASSERT_EQ(K.size(), (size_t)18);

	auto in_roi = [&roidata](StatsInt x, StatsInt y) {
		for (const auto& p : roidata.raw_pixels)
			if (p.x == x && p.y == y)
				return true;
		return false;
	};

	size_t outside = 0, inside_after_unpadding = 0;
	for (const auto& c : K)
	{
		if (!in_roi(c.x, c.y))
			outside++;
		if (in_roi(c.x - 1, c.y - 1))
			inside_after_unpadding++;
	}

	// undoing the one-pixel pad puts every contour pixel back onto a ROI pixel
	ASSERT_EQ(inside_after_unpadding, K.size());
	ASSERT_EQ(outside, (size_t)7);
}

// Pixel2::find_center and Pixel2::max_sqdist are coarse-to-fine searches that assume an ordered
// contour; merge_multicontour concatenates the outer contour and the hole's, so the sequence they
// walk is not one. On this ROI both return a non-extremal answer, and the radius the feature
// normalises by is 23% short of the largest distance actually present.
void test_2d_radial_center_and_radius_mechanics()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);

	std::vector<Pixel2> K;
	roidata.merge_multicontour(K);

	int idxO = Pixel2::find_center(roidata.raw_pixels, K);
	const Pixel2& pxO = roidata.raw_pixels[idxO];
	ASSERT_EQ(pxO.x, 3);
	ASSERT_EQ(pxO.y, 4);

	// the searched maximum against a full linear scan of the same contour
	double exact_max = 0.0;
	for (const auto& p : K)
		exact_max = std::max(exact_max, pxO.sqdist(p));
	ASSERT_DOUBLE_EQ(pxO.max_sqdist(K), 10.0);
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
	ASSERT_EQ(roidata.raw_pixels[exact_idx].x, 4);
	ASSERT_EQ(roidata.raw_pixels[exact_idx].y, 4);
}

// The bin index is int(r / r_max * (num_bins - 1)), so the 8 bins are 7 equal-width rings plus a
// last bin that only r >= r_max reaches. Here r_max is the short radius pinned above, so the last
// bin holds the 3 pixels sitting exactly on it plus the 4 that lie beyond it - 7 of the 26, more
// than any of the 7 real rings.
void test_2d_radial_bin_index_mechanics()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);

	std::vector<Pixel2> K;
	roidata.merge_multicontour(K);
	const Pixel2& pxO = roidata.raw_pixels[Pixel2::find_center(roidata.raw_pixels, K)];
	const double r_max = std::sqrt(pxO.max_sqdist(K));

	size_t at_r_max = 0, past_r_max = 0;
	for (const auto& p : roidata.raw_pixels)
	{
		const double rat = std::sqrt(p.sqdist(pxO)) / r_max;
		if (rat > 1.0)
			past_r_max++;
		else if (rat == 1.0)
			at_r_max++;
	}
	ASSERT_EQ(past_r_max, (size_t)4);
	ASSERT_EQ(at_r_max, (size_t)3);

	// and those 7 together are what the last bin of all three features is computed from
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];
	const auto& cv = roidata.fvals[(int)Nyxus::Feature2D::RADIAL_CV];
	const double last_bin_count = frac.back() * double(roidata.raw_pixels.size());
	ASSERT_NEAR(last_bin_count, double(at_r_max + past_r_max), 1e-6) << "FRAC_AT_D last bin";
	ASSERT_GT(mean.back(), 0.0) << "MEAN_FRAC last bin is fed by the clamped pixels";
	ASSERT_GT(cv.back(), 0.0) << "RADIAL_CV last bin is fed by the clamped pixels";
}
