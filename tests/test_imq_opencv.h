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
// relative) on FOCUS_SCORE and 3.6e-15 / 4.7e-16 on LOCAL_FOCUS_SCORE - agreement, not bit
// identity, and the residual is the variance summation order alone. See
// tests/vetting/audit/imq_opencv_vetting_report.md.
//
// Scope of the claim:
//   * ksize=1 only -- the ksize>1 kernel {{2,0,2},{0,-8,0},{2,0,2}} has no cv2.Laplacian
//     counterpart and calculate() never selects it.
//   * LOCAL_FOCUS_SCORE covers the top-left tile only. get_local_focus_score() loops
//     y < height - M with M = height/scale, so at scale=2 exactly one 4x6 tile is visited while
//     the divisor stays scale^2 = 4; the golden is var(Laplacian(that tile)) / 4. The loop bound is
//     pinned as-is, not endorsed -- tests/vetting/matrix/imq.md records it as an open defect.
//   * The out-of-core path (get_focus_score_NT) is not covered.
//
// CellProfiler also publishes features named FocusScore / LocalFocusScore, but those are a
// different statistic (normalized variance of the raw image), so CellProfiler is not an oracle for
// these two -- see test_imq_cellprofiler.h.
static const ref_vals_map<double> imq_opencv_ref_vals {
	{"FOCUS_SCORE", 34.956597222222221},
	{"LOCAL_FOCUS_SCORE", 7.5763888888888902}
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
