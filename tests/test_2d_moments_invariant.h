#pragma once

#include "test_2d_moments_common.h"

// Copies of one disk of radius 12, 50 pixels apart along x, under one label, with an intensity that
// depends only on a pixel's offset from its own disk's centre. Returns the feature values of the
// weighted shape and intensity moments.
static void calculate_2d_moments_disk_copies_values (int n_copies, std::vector<std::vector<double>>& fvals)
{
	const int R = 12, cx = 20, cy = 20, gap = 50;
	LR roidata(1402);
	for (int k = 0; k < n_copies; k++)
		for (int y = cy - R; y <= cy + R; y++)
			for (int x = cx - R + k * gap; x <= cx + R + k * gap; x++)
			{
				const int dx = x - (cx + k * gap), dy = y - cy;
				if (dx * dx + dy * dy > R * R)
					continue;
				const auto intensity = static_cast<PixIntens>(10 + (dx + R) + 2 * (dy + R));
				if (roidata.aux_area == 0)
					Nyxus::init_label_record_3(roidata, x, y, intensity);
				else
					Nyxus::update_label_record_3(roidata, x, y, intensity);
				roidata.raw_pixels.push_back(Pixel2(x, y, intensity));
			}
	roidata.make_nonanisotropic_aabb();
	roidata.aux_image_matrix.allocate(roidata.aabb.get_width(), roidata.aabb.get_height());
	roidata.aux_image_matrix.calculate_from_pixelcloud(roidata.raw_pixels, roidata.aabb);
	roidata.initialize_fvals();

	Fsettings s = make_2d_geomoment_settings();
	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);
	ASSERT_EQ(roidata.multicontour_.size(), (size_t)n_copies);

	Smoms2D_feature shape_moments;
	shape_moments.calculate(roidata, s);
	shape_moments.save_value(roidata.fvals);

	Imoms2D_feature intensity_moments;
	intensity_moments.calculate(roidata, s);
	intensity_moments.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

// The weighted moments weigh each pixel by log(d + epsilon), d its distance to the ROI's contour
// (2d_geomoments_basic.cpp). For two copies of a disk far apart, the nearest border to every pixel
// is its own disk's and lies where it does in a lone disk, so every pixel carries the weight it
// carries in one copy, and the zeroth weighted moments -- the summed weights -- of the pair are twice
// those of one copy. The distance is searched along each border on its own; a search that ran
// along the two borders as one sequence would give the copies different weights.
void test_2d_moments_weighted_disk_copies_invariant()
{
	std::vector<std::vector<double>> one, two;
	calculate_2d_moments_disk_copies_values(1, one);
	if (::testing::Test::HasFatalFailure())
		return;
	calculate_2d_moments_disk_copies_values(2, two);
	if (::testing::Test::HasFatalFailure())
		return;

	// the two sums add the same terms in a different order
	const double w1 = one[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00][0], i1 = one[(int)Nyxus::Feature2D::IMOM_WRM_00][0];
	ASSERT_NE(w1, 0.0);
	ASSERT_NE(i1, 0.0);
	ASSERT_NEAR(two[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00][0], 2.0 * w1, 1e-12 * std::abs(2.0 * w1));
	ASSERT_NEAR(two[(int)Nyxus::Feature2D::IMOM_WRM_00][0], 2.0 * i1, 1e-12 * std::abs(2.0 * i1));
}
