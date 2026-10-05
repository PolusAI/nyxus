#pragma once

#include <gtest/gtest.h>

#include <cstdint>                               // the pseudo-random ROIs' generator state
#include <string>

#include "test_imq_common.h"                     // fixture: imq_settings, imq_all_ones_mask, imq_spikes_intensity, the im_quality arrays, load_masked_test_roi_data
#include "../src/nyx/features/focus_score.h"     // FocusScoreFeature, and ImageLoader via environment.h

// Out-of-core equals in-RAM for FOCUS_SCORE and LOCAL_FOCUS_SCORE (SPEC 2, invariant). An ROI past
// the RAM limit is scored by osized_calculate() from a disk-backed copy of its pixels instead of by
// calculate() from the in-RAM image matrix. Both read the same bounding-box image and hand it to the
// same two scoring templates, so the scores must be EQUAL, not close: any difference means the two
// paths no longer compute the same thing. This establishes no vetting of its own - the in-RAM
// scores are vetted in test_imq_opencv.h and test_imq_analytic.h.
//
// The ROIs cover the shapes the out-of-core path has to get right: the im_quality fixture (masked
// zeros inside the box), pseudo-random textures wider and taller than 30 px on each side and on one
// side only, a constant ROI, and a 1 px thin one whose LOCAL_FOCUS_SCORE is the soft-NaN. They run
// through ONE FocusScoreFeature instance, as processNontrivialRois() runs every oversized ROI of a
// slide, and the constant and thin ROIs come after textured ones, so a score left over from the
// previous ROI cannot pass for this one's.

struct ImqFocusScores
{
	double focus, local;
};

static ImqFocusScores imq_focus_in_ram (const NyxusPixel* intensity, const NyxusPixel* mask, size_t count)
{
	LR r (1);
	FocusScoreFeature f;
	Fsettings s = imq_settings();

	Nyxus::load_masked_test_roi_data (r, intensity, mask, count);
	f.calculate (r, s);
	r.initialize_fvals();
	f.save_value (r.fvals);

	return { r.fvals[(int)Nyxus::FeatureIMQ::FOCUS_SCORE][0], r.fvals[(int)Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE][0] };
}

static ImqFocusScores imq_focus_out_of_core (FocusScoreFeature& f, const NyxusPixel* intensity, const NyxusPixel* mask, size_t count)
{
	LR r (1);
	Fsettings s = imq_settings();

	// The same bounding box and pixels the in-RAM side gets, then streamed into the disk-backed cloud
	Nyxus::load_masked_test_roi_data (r, intensity, mask, count);
	r.raw_pixels_NT.init (r.label, "nyxus_ut_imq_focus_ooc");
	for (const auto& p : r.raw_pixels)
		r.raw_pixels_NT.add_pixel (p);
	r.initialize_fvals();

	ImageLoader dummy;   // osized_calculate reads its pixels from the cloud, not the loader
	f.osized_calculate (r, s, dummy);
	ImqFocusScores out = { r.fvals[(int)Nyxus::FeatureIMQ::FOCUS_SCORE][0], r.fvals[(int)Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE][0] };

	r.raw_pixels_NT.close();
	return out;
}

// A w x h ROI of pseudo-random 16-bit intensities, the same for a given seed on every platform
static std::vector<NyxusPixel> imq_random_intensity (size_t w, size_t h, uint32_t seed)
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++)
		{
			seed = seed * 1664525u + 1013904223u;
			px.push_back (NyxusPixel {x, y, (seed >> 16) & 0xFFFFu});
		}
	return px;
}

// A w x h ROI holding v everywhere
static std::vector<NyxusPixel> imq_constant_intensity (size_t w, size_t h, unsigned int v)
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++)
			px.push_back (NyxusPixel {x, y, v});
	return px;
}

void test_imq_focus_score_out_of_core_invariant()
{
	struct Roi
	{
		std::string name;
		std::vector<NyxusPixel> intensity, mask;
	};
	std::vector<Roi> rois = {
		{ "70 x 45 random",         imq_random_intensity (70, 45, 1u),  imq_all_ones_mask (70, 45) },
		{ "50 x 20 random",         imq_random_intensity (50, 20, 2u),  imq_all_ones_mask (50, 20) },
		{ "6 x 6 constant",         imq_constant_intensity (6, 6, 7u),  imq_all_ones_mask (6, 6) },
		{ "20 x 50 random",         imq_random_intensity (20, 50, 3u),  imq_all_ones_mask (20, 50) },
		{ "5 x 1 spike",            imq_spikes_intensity (5, 1, {{2, 0}}), imq_all_ones_mask (5, 1) },
		{ "9 x 13 random",          imq_random_intensity (9, 13, 4u),   imq_all_ones_mask (9, 13) },
	};
	const size_t n_fixture = sizeof(im_quality_mask) / sizeof(NyxusPixel);
	rois.push_back ({ "im_quality fixture",
		std::vector<NyxusPixel> (im_quality_intensity, im_quality_intensity + n_fixture),
		std::vector<NyxusPixel> (im_quality_mask, im_quality_mask + n_fixture) });

	FocusScoreFeature persistent;
	for (const auto& roi : rois)
	{
		SCOPED_TRACE ("INVARIANT__FOCUS_SCORE out of core vs in RAM, " + roi.name + " ROI");
		ImqFocusScores ram = imq_focus_in_ram (roi.intensity.data(), roi.mask.data(), roi.mask.size()),
			ooc = imq_focus_out_of_core (persistent, roi.intensity.data(), roi.mask.data(), roi.mask.size());

		ASSERT_EQ (ooc.focus, ram.focus) << "FOCUS_SCORE";
		ASSERT_EQ (ooc.local, ram.local) << "LOCAL_FOCUS_SCORE";
	}
}
