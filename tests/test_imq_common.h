#pragma once

// The fixture the IMQ oracle and snapshot files share. gtest is deliberately absent: this file
// builds one ROI and asserts nothing - the files that include it bring gtest in themselves, and
// SPEC 6.3.1 keeps every golden table with the assertions that read it.

#include "../src/nyx/feature_settings.h"   // Fsettings
#include "../src/nyx/featureset.h"         // FeatureIMQ
#include "../src/nyx/roi_cache.h"          // LR
#include "test_data.h"                     // NyxusPixel, im_quality_intensity, im_quality_mask
#include "test_main_nyxus.h"               // load_masked_test_roi_data

// One IMQ feature computed on an arbitrary {intensity, mask} pixel pair. The im_quality fixture is
// one caller among several - the matrix cells that live outside it (a constant ROI, a mask narrower
// than the bounding box, an ROI wide enough to clear the power-spectrum guard) are built by their
// own assertions and fed through here.
//
// Templated on the feature class so this header needs none of the feature headers - each including
// file brings its own. No settings are passed beyond a default-constructed Fsettings: none of the
// four IMQ feature classes reads a NyxSetting.
template <class F>
static double calc_imq_feature_on (Nyxus::FeatureIMQ feature, const NyxusPixel* intensity,
	const NyxusPixel* mask, size_t count)
{
	LR roidata;
	F f;
	Fsettings s;

	Nyxus::load_masked_test_roi_data (roidata, intensity, mask, count);
	f.calculate (roidata, s);
	roidata.initialize_fvals();
	f.save_value (roidata.fvals);

	return roidata.fvals[(int)feature][0];
}

// The im_quality fixture. The mask covers the whole bounding box, so the ROI image matrix is the
// full 8 x 12 rectangle; rows 7..9 of the intensity literal repeat the coordinates of rows 1..3,
// which leaves x=3..8 there unassigned and therefore 0. That 0 is the ROI's observed minimum and it
// is what MIN_SATURATION counts.
template <class F>
static double calc_imq_feature (Nyxus::FeatureIMQ feature)
{
	return calc_imq_feature_on<F> (feature, im_quality_intensity, im_quality_mask,
		sizeof(im_quality_mask) / sizeof(NyxusPixel));
}
