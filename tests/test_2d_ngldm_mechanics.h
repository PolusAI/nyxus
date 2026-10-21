#pragma once

// Mechanics of the 2D NGLDM family: what a setting does to the grey levels the matrix is built on,
// rather than the values against a reference. Claims no oracle (SPEC 2).

// Nothing of its own: gtest, the settings recipe, NGLDMfeature and the ROI loader with its fixtures
// all arrive with the common header.
#include "test_2d_ngldm_common.h"

static std::vector<std::vector<double>> run_2d_ngldm_square12 (int greydepth, bool ibsi)
{
	std::vector<NyxusPixel> intensity, mask;
	Nyxus::make_2d_square12_graded_phantom (intensity, mask);

	Fsettings s = make_ngldm2d_settings (ibsi);
	s[(int)NyxSetting::GREYDEPTH].ival = greydepth;

	LR roidata;
	Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
	NGLDMfeature f;
	f.calculate (roidata, s);
	roidata.initialize_fvals();
	f.save_value (roidata.fvals);
	return roidata.fvals;
}

// A coarse grey depth of 0 bins nothing, so NGLDM reads the raw levels -- the same grey levels IBSI
// mode reads -- and every feature equals the IBSI run bit for bit. bench_square12_graded has 36 raw
// levels; a 0 that collapsed them would leave one grey level and move every feature.
void test_2d_ngldm_zero_greydepth_equals_ibsi_mechanics()
{
	const auto none = run_2d_ngldm_square12 (0, false),
		ibsi = run_2d_ngldm_square12 (64, true);
	for (auto fc : NGLDMfeature::featureset)
	{
		SCOPED_TRACE ("2D feature code " + std::to_string ((int)fc));
		ASSERT_EQ (none[(int)fc], ibsi[(int)fc]);
	}
}
