#pragma once

// The geometry ZernikeFeature measures in. None of it is a feature value and none of it claims an
// oracle (SPEC 2) - these are drift guards on the three quantities that fix every one of the 30
// magnitudes, and that nothing else in the tree looks at.
//
// mb_zernike2D builds its unit disk from the ROI's bounding box: the radius is min(width, height) in
// pixels and the centre is the ROI's intensity centroid. Both are read off the same aux image matrix
// the moments are summed over, so pinning them is what tells a reader which of the three inputs
// moved when all 30 values move together.

#include <gtest/gtest.h>

#include <algorithm>                          // std::min, std::max

#include "../src/nyx/features/image_matrix.h" // ImageMatrix, readOnlyPixels
#include "test_2d_zernike_common.h"           // build_zernike_2d_roi, LR
#include "test_main_nyxus.h"                  // <cmath> for std::hypot

// The bounding box the moments are summed over, the disk radius derived from it, and the intensity
// centroid the disk is centred on.
void test_2d_zernike_geometry_mechanics()
{
	LR roidata(101);
	build_zernike_2d_roi(roidata);

	const ImageMatrix& im = roidata.aux_image_matrix;
	const int cols = im.width, rows = im.height;
	ASSERT_EQ(cols, 6) << "ZERNIKE2D bounding box";
	ASSERT_EQ(rows, 7) << "ZERNIKE2D bounding box";

	const double rad = double(std::min(cols, rows));
	ASSERT_DOUBLE_EQ(rad, 6.0) << "ZERNIKE2D disk radius";

	readOnlyPixels pix = im.ReadablePixels();
	double total = 0.0, m10 = 0.0, m01 = 0.0;
	for (int i = 0; i < cols; i++)
		for (int j = 0; j < rows; j++)
		{
			const double v = (double)pix.yx(j, i);
			total += v;
			m10 += (i + 1) * v;
			m01 += (j + 1) * v;
		}
	ASSERT_DOUBLE_EQ(total, 1048.0) << "ZERNIKE2D total intensity";
	ASSERT_NEAR(m10 / total, 3.8416030534351144, 1e-12) << "ZERNIKE2D centroid x";
	ASSERT_NEAR(m01 / total, 4.4389312977099236, 1e-12) << "ZERNIKE2D centroid y";
}

// Every pixel of the bounding box lies inside the unit disk, which is why the weights the moments
// are summed with add to exactly 1 and Z(0,0) is exactly 1/pi. If a future fixture put pixels
// outside the disk, mb_zernike2D would drop them and that identity would no longer hold.
void test_2d_zernike_every_pixel_is_inside_the_unit_disk_mechanics()
{
	LR roidata(101);
	build_zernike_2d_roi(roidata);

	const ImageMatrix& im = roidata.aux_image_matrix;
	const int cols = im.width, rows = im.height;
	const double rad = double(std::min(cols, rows));

	readOnlyPixels pix = im.ReadablePixels();
	double total = 0.0, m10 = 0.0, m01 = 0.0;
	for (int i = 0; i < cols; i++)
		for (int j = 0; j < rows; j++)
		{
			const double v = (double)pix.yx(j, i);
			total += v; m10 += (i + 1) * v; m01 += (j + 1) * v;
		}
	const double cx = m10 / total, cy = m01 / total;

	double r_max = 0.0;
	for (int i = 0; i < cols; i++)
		for (int j = 0; j < rows; j++)
			r_max = std::max(r_max, std::hypot((i + 1 - cx) / rad, (j + 1 - cy) / rad));

	ASSERT_LT(r_max, 1.0) << "ZERNIKE2D unit disk covers every bounding-box pixel";
}
