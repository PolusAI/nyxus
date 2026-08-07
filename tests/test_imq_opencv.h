#pragma once

// OpenCV oracle for the image-quality focus-score features FOCUS_SCORE and
// LOCAL_FOCUS_SCORE (SPEC 2 / 6.1: correctness claims live in oracle files, never in
// _regression files).
//
// FOCUS_SCORE is the Pech-Pacheco et al. (2000) variance-of-the-Laplacian focus measure,
// i.e. cv2.Laplacian(img, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT).var().
// The real cv2.Laplacian was run on this fixture: its filtered image is bit-identical to
// Nyxus' hand-rolled laplacian() (max abs diff 0.0 -- same ksize=1 kernel, same zero
// padding), so only the variance step had to agree, and it now does. Goldens + the
// offline OpenCV run are in tests/vetting/oracles/gen_imq_opencv.py.
//
// Scope of the claim:
//   * ksize=1 only -- the ksize>1 kernel {{2,0,2},{0,-8,0},{2,0,2}} has no cv2.Laplacian
//     counterpart and calculate() never selects it.
//   * LOCAL_FOCUS_SCORE covers the top-left tile only. get_local_focus_score() loops
//     y < height - M with M = height/scale, so with scale=2 exactly one tile is visited,
//     not scale^2; the golden is var(Laplacian(top-left h/2 x w/2 tile)) / scale^2. The
//     loop bound is pinned as-is, not endorsed.
//   * The out-of-core path (get_focus_score_NT) is not covered.
//
// CellProfiler also publishes features named FocusScore / LocalFocusScore, but those are
// a different statistic (normalized variance of the raw image), so CellProfiler is not
// an oracle for these two -- see tests/vetting/oracles/gen_imq_cellprofiler.py.

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_feature_calculation.h"

#include "../src/nyx/features/focus_score.h"

void test_imq_focus_score_opencv() {

    FocusScoreFeature f;
    // cv2 4.13.0: cv2.Laplacian(roi, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT).var()
    // gen_imq_opencv.py; tolerance rel=1e-3 (same definition, different summation order).
    double truth_value = 34.95659722222222;

    test_feature(f, Nyxus::FeatureIMQ::FOCUS_SCORE, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};

void test_imq_local_focus_score_opencv() {

    FocusScoreFeature f;
    // cv2 4.13.0: the same Laplacian call on the top-left h/2 x w/2 tile, / scale^2.
    // gen_imq_opencv.py; tolerance rel=1e-3.
    double truth_value = 7.57638888888889;

    test_feature(f, Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};
