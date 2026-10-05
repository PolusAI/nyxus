#pragma once

// The fixture the IMQ oracle and snapshot files share. gtest is deliberately absent: this file
// builds one ROI and asserts nothing - the files that include it bring gtest in themselves, and
// SPEC 6.3.1 keeps every golden table with the assertions that read it.

#include <utility>                         // std::pair, the spike cells
#include <vector>                          // imq_all_ones_mask, imq_spikes_intensity

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
// file brings its own.

// The soft-NaN the IMQ tests run with. FocusScoreFeature returns it for a LOCAL_FOCUS_SCORE no tile
// fits. It is a value no focus score can take - a population variance is never negative - so an
// assertion expecting it cannot be met by a computed score, and it compares exactly.
static const double imq_soft_nan = -1234.5;

// Settings as Environment::compile_feature_settings() lays them out, one slot per NyxSetting, with
// the soft-NaN above. Only FocusScoreFeature reads any of them.
static Fsettings imq_settings()
{
	Fsettings s ((int)NyxSetting::__COUNT__);
	s[(int)NyxSetting::SOFTNAN].rval = imq_soft_nan;
	return s;
}

template <class F>
static double calc_imq_feature_on (Nyxus::FeatureIMQ feature, const NyxusPixel* intensity,
	const NyxusPixel* mask, size_t count)
{
	LR roidata;
	F f;
	Fsettings s = imq_settings();

	Nyxus::load_masked_test_roi_data (roidata, intensity, mask, count);
	f.calculate (roidata, s);
	roidata.initialize_fvals();
	f.save_value (roidata.fvals);

	return roidata.fvals[(int)feature][0];
}

// A mask covering the whole w x h rectangle, so the ROI image matrix is the full intensity array
static std::vector<NyxusPixel> imq_all_ones_mask (size_t w, size_t h)
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++)
			px.push_back (NyxusPixel {x, y, 1u});
	return px;
}

// The spike height of the focus-score probes: a w x h ROI that is 0 except at a few listed cells
static const unsigned int imq_spike = 6;

// A w x h ROI, 0 everywhere except the listed (x, y) cells, which hold imq_spike. Paired with
// imq_all_ones_mask, the ROI image matrix is exactly this array.
static std::vector<NyxusPixel> imq_spikes_intensity (size_t w, size_t h, const std::vector<std::pair<size_t, size_t>>& spikes)
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++)
		{
			unsigned int v = 0;
			for (const auto& s : spikes)
				if (s.first == x && s.second == y)
					v = imq_spike;
			px.push_back (NyxusPixel {x, y, v});
		}
	return px;
}

// One IMQ feature on a spike ROI. The feature is passed at the call site, on the assertion line,
// because tests/vetting/audit/scan_imq_coverage.py attributes coverage from the line that asserts.
template <class F>
static double calc_imq_feature_on_spikes (Nyxus::FeatureIMQ feature, size_t w, size_t h,
	const std::vector<std::pair<size_t, size_t>>& spikes)
{
	std::vector<NyxusPixel> intensity = imq_spikes_intensity (w, h, spikes),
		mask = imq_all_ones_mask (w, h);
	return calc_imq_feature_on<F> (feature, intensity.data(), mask.data(), mask.size());
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
