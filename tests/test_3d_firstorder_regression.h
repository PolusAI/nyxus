#pragma once

#include "test_3d_firstorder_common.h"	// assert_3d_firstorder_feature, ref_vals_map

// 3COVERED_IMAGE_INTENSITY_RANGE is the only 3D first-order feature the registry marks
// status=regression: it is a fraction of the slide's own dynamic range, a Nyxus convention with no
// counterpart in any listed oracle, so it is a drift guard and establishes no vetting (SPEC 1).
// Kept in its own file per SPEC 2, one kind per file.
//
// The pin is Nyxus' own output at full precision on the fixture
// tests/data/nifti/phantoms/ut_inten.nii + ut_mask57.nii, label 57, at default settings.
//
// It is greater than 1, which the feature's definition does not allow: the value is the ROI's
// intensity range over the slide's, and an ROI cannot span more than the slide that contains it.
// The numerator comes from the ROI in the loader's truncated integer domain (3024 - 1024 = 2000)
// while the denominator comes from SlideProps in the raw stored domain (2000 - 0.40855798 =
// 1999.5914), so the ratio is measuring two different scales against each other.
//
// The pin below is a drift guard on what the code produces, not an endorsement of it. The bound
// itself is deliberately not asserted here: an assertion of it would fail, and a bound is an
// invariant rather than a snapshot (SPEC 4.4), so it belongs in an _invariant file once the ratio
// is computed in one domain. The registry carries flag=bound-violation and
// tests/vetting/not_covered.md section E records the defect.
static ref_vals_map<double> firstorder_3d_regression_ref_vals
{
	{ "3COVERED_IMAGE_INTENSITY_RANGE",	1.0002043207290587 },
};

void test_3d_firstorder_covered_image_intensity_range_regression() {

	assert_3d_firstorder_feature (
		"3COVERED_IMAGE_INTENSITY_RANGE",
		Nyxus::Feature3D::COVERED_IMAGE_INTENSITY_RANGE,
		firstorder_3d_regression_ref_vals["3COVERED_IMAGE_INTENSITY_RANGE"],
		1.e9);

}
