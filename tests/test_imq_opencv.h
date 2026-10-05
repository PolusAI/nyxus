#pragma once

#include <gtest/gtest.h>

#include "test_imq_common.h"                   // fixture: calc_imq_feature, and FeatureIMQ via featureset.h
#include "test_ref_vals.h"                     // ref_vals_map, and <string> for the helper
#include "../src/nyx/features/focus_score.h"   // FocusScoreFeature

// OpenCV oracle for the image-quality focus-score features FOCUS_SCORE and LOCAL_FOCUS_SCORE
// (SPEC 2 / 6.1: correctness claims live in oracle files, never in _regression files).
// tool=OpenCV 4.13.0 (opencv-python), cv2.Laplacian(roi, cv2.CV_64F, ksize=1,
// borderType=cv2.BORDER_CONSTANT) then ndarray.var(); env=nyxus_mirp (conda);
// recipe=imq.laplacian_ksize1_zeropad; generator=tests/vetting/oracles/gen_imq_opencv.py.
//
// FOCUS_SCORE is the Pech-Pacheco et al. (2000) variance-of-the-Laplacian focus measure. The
// generator asserts cv2's filtered image equals Nyxus' hand-rolled laplacian() cell for cell before
// it compares any scalar, so the convolution is proved and only the variance step is being checked
// here. The pins carry OpenCV's own digits; Nyxus reproduces them to 7.1e-15 absolute (2.0e-16
// relative) on FOCUS_SCORE and 7.1e-15 / 2.5e-16 on LOCAL_FOCUS_SCORE - agreement, not bit
// identity, and the residual is the variance summation order alone. See
// tests/vetting/audit/imq_opencv_vetting_report.md.
//
// Scope of the claim:
//   * The two scores are vetted at ksize=1, the only kernel calculate() selects. laplacian()'s
//     ksize>1 kernel {{2,0,2},{0,-8,0},{2,0,2}} is cv2.Laplacian's ksize=3 aperture exactly; both
//     stencils are pinned below from cv2's response to a unit spike, and the generator also
//     asserts the ksize=3 filtered fixture equal to cv2's cell for cell.
//   * LOCAL_FOCUS_SCORE is the mean of var(Laplacian(tile)) over the 2x2 grid of 4x6 tiles at
//     scale=2. cv2 computes each tile's score; the grid is Nyxus' definition, reproduced in the
//     generator. Which tiles are averaged is asserted tile by tile in test_imq_analytic.h.
//   * The out-of-core path shares the in-RAM scoring code and is held equal to it by
//     test_imq_invariant.h, not vetted here.
//
// CellProfiler also publishes features named FocusScore / LocalFocusScore, but those are a
// different statistic (normalized variance of the raw image), so CellProfiler is not an oracle for
// these two -- see test_imq_cellprofiler.h.
static const ref_vals_map<double> imq_opencv_ref_vals {
	{"FOCUS_SCORE", 34.956597222222221},
	{"LOCAL_FOCUS_SCORE", 28.341145833333336}
};

// SPEC 7's exact tier verbatim: an absolute band, so ASSERT_NEAR rather than the relative agrees_gt
// the looser-tiered files use. The tier applies for the reason the SPEC gives it -- the two sides
// filter the image identically and differ only in the order the variance is summed. The band is
// what SPEC 7 sets rather than what the measurement needs: the worst residual it covers is 7.1e-15.
static const double imq_opencv_abs_tolerance = 1e-9;

static void assert_imq_opencv (Nyxus::FeatureIMQ feature, const std::string& feature_name)
{
	SCOPED_TRACE (std::string("OPENCV__") + feature_name);

	// .at() on a const table: operator[] would default-insert a missing key and compare against
	// the zero it just created, so a missing pin would read as a golden of 0
	ASSERT_TRUE (imq_opencv_ref_vals.count(feature_name) > 0) << feature_name;

	ASSERT_NEAR (calc_imq_feature<FocusScoreFeature>(feature),
		imq_opencv_ref_vals.at(feature_name), imq_opencv_abs_tolerance) << feature_name;
}

void test_imq_focus_score_opencv()
{
	assert_imq_opencv (Nyxus::FeatureIMQ::FOCUS_SCORE, "FOCUS_SCORE");
}

void test_imq_local_focus_score_opencv()
{
	assert_imq_opencv (Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, "LOCAL_FOCUS_SCORE");
}

// cv2.Laplacian(spike, CV_64F, ksize=k, borderType=cv2.BORDER_CONSTANT) of a 5x5 image holding a
// single 1 at its centre, rows 1..3 and columns 1..3: the filter's own stencil. A filter's response
// to a unit spike is its kernel, so these are cv2's kernels, re-verified by gen_imq_opencv.py.
static const double imq_opencv_laplacian_ksize1[9] = {  0,  1,  0,
                                                        1, -4,  1,
                                                        0,  1,  0 };
static const double imq_opencv_laplacian_ksize3[9] = {  2,  0,  2,
                                                        0, -8,  0,
                                                        2,  0,  2 };

// laplacian() at ksize=3 and then ksize=1 on the unit spike reproduces cv2's two stencils. The
// order matters: each call chooses its kernel from its own ksize, so the ksize=1 call after a
// ksize=3 one must still return the ksize=1 stencil. Exact equality: both sides multiply 1 by small
// integers.
void test_imq_focus_score_kernel_per_call_opencv()
{
	const int n = 5;
	std::vector<PixIntens> spike (n * n, 0);
	spike[2 * n + 2] = 1;

	std::vector<double> wide (n * n, 0.);
	FocusScoreFeature::laplacian (spike, wide, n, n, 3);
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			ASSERT_EQ (wide[(r + 1) * n + (c + 1)], imq_opencv_laplacian_ksize3[r * 3 + c]) << "ksize=3 at " << r << "," << c;

	std::vector<double> plain (n * n, 0.);
	FocusScoreFeature::laplacian (spike, plain, n, n, 1);
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			ASSERT_EQ (plain[(r + 1) * n + (c + 1)], imq_opencv_laplacian_ksize1[r * 3 + c]) << "ksize=1 at " << r << "," << c;
}
