#pragma once

#include <gtest/gtest.h>
#include <string>
#include "test_2d_contour_common.h"
#include "test_ref_vals.h"

// The ROI contour and PERIMETER against OpenCV's Suzuki-Abe border following,
// cv2.findContours(RETR_LIST, CHAIN_APPROX_NONE): every outer and hole border as a closed walk of
// pixels, PERIMETER being the summed closed-walk length (accumulated in double from OpenCV's integer
// points). Same algorithm on both sides, so the agreement is exact up to summation order.
// Tool: OpenCV 4.13.0 (5.0.0 gives the same pins). Generator:
// tests/vetting/oracles/gen_contour_opencv.py, which reads the fixtures from their headers and
// re-verifies every pin below.
static const ref_vals_map<double> contour_2d_opencv_ref_vals{
	// bench_blob97_filled: 9066 ROI pixels, no holes
	{"NUM_CONTOURS_BLOB97", 1},
	{"CONTOUR_PIXELS_BLOB97", 495},
	{"PERIMETER_BLOB97", 537.6639969244295},
	// shape2d_morphology_mask: 26 ROI pixels around a one-pixel hole
	{"NUM_CONTOURS_SHAPE2D", 2},
	{"CONTOUR_PIXELS_SHAPE2D", 18},
	{"PERIMETER_SHAPE2D", 22.970562748477146},
};

// The number of borders and the pixels they cover.
static void assert_contour_borders_opencv (const LR& r, const std::string& tag)
{
	SCOPED_TRACE("OPENCV_ORACLE__" + tag);
	ASSERT_EQ ((double)r.multicontour_.size(), contour_2d_opencv_ref_vals.at("NUM_CONTOURS_" + tag));
	ASSERT_EQ ((double)contour_pixel_set(r).size(), contour_2d_opencv_ref_vals.at("CONTOUR_PIXELS_" + tag));
}

// The filled 9066-pixel blob on which the contour once collapsed to two pixels (PERIMETER 2).
void test_2d_contour_blob97_opencv()
{
	LR r (1);
	contour_roi_from_pixels (r, contour_mask_pixels (bench_blob97_filled_rows, 113, 41));
	contour_calculate (r);
	assert_contour_borders_opencv (r, "BLOB97");
	ASSERT_NEAR (r.fvals[(int)Nyxus::Feature2D::PERIMETER][0], contour_2d_opencv_ref_vals.at("PERIMETER_BLOB97"), 1e-9);
}

// The 8x8 shape2d mask: an outer border and the border of its one-pixel hole.
void test_2d_contour_shape2d_opencv()
{
	LR r (1);
	Nyxus::load_masked_test_roi_data (r, shape2d_morphology_intensity, shape2d_morphology_mask,
		sizeof(shape2d_morphology_mask) / sizeof(NyxusPixel));
	contour_calculate (r);
	assert_contour_borders_opencv (r, "SHAPE2D");
	ASSERT_NEAR (r.fvals[(int)Nyxus::Feature2D::PERIMETER][0], contour_2d_opencv_ref_vals.at("PERIMETER_SHAPE2D"), 1e-9);
}
