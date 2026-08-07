#pragma once

// 3COVERED_IMAGE_INTENSITY_RANGE is the only 3D first-order feature the registry marks
// status=regression (target_test=test_3d_firstorder_regression.h): it is a fraction of the image's
// own dynamic range, a Nyxus convention with no MATLAB counterpart, so unlike its 36 siblings it is a
// drift guard rather than an oracle assertion. Split out here per SPEC 2 (one kind per file); the
// golden map and the shared helper still live in test_3d_firstorder_matlab.h, which this includes.
//
// NOT WIRED IN: like the file it comes from, this header is not #include'd by test_all.cc, so the
// assertion does not run (not_covered.md section B.1).

#include "test_3d_firstorder_matlab.h"

void test_3d_firstorder_ciir_regression() {
	assert_3d_firstorder_feature_matlab ("3COVERED_IMAGE_INTENSITY_RANGE", Nyxus::Feature3D::COVERED_IMAGE_INTENSITY_RANGE);
}
