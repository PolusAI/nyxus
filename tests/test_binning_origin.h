#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/features/texture_feature.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/ngtdm.h"
#include "../src/nyx/roi_cache.h"
#include "test_data.h"
#include "test_main_nyxus.h"

namespace Nyxus
{
	// "zero" origin (positive greybin_info), MATLAB-style:
	//   slope = n_levels / max, intercept = 1
	//   bin = floor(slope * x + intercept), clipped to [1, n_levels]
	static void test_bin_pixel_zero_origin()
	{
		int n_bins = 10;
		PixIntens min_I = 50, max_I = 200, x = 100;

		// slope = 10 / 200 = 0.05, intercept = 1 - 0.05*0 = 1
		// bin = floor(0.05 * 100 + 1) = floor(6) = 6
		auto result = TextureFeature::bin_pixel(x, min_I, max_I, n_bins);
		ASSERT_EQ(result, 6);
	}

	// "min" origin (negative greybin_info), PyRadiomics-style:
	//   binWidth = (max - min) / binCount
	//   bin = floor((x - min) / binWidth) + 1
	static void test_bin_pixel_min_origin()
	{
		int n_bins = 10;
		PixIntens min_I = 50, max_I = 200, x = 100;

		// binWidth = (200 - 50) / 10 = 15
		// bin = floor((100 - 50) / 15) + 1 = floor(3.33) + 1 = 4
		auto result = TextureFeature::bin_pixel(x, min_I, max_I, -n_bins);
		ASSERT_EQ(result, 4);
	}

	// End-to-end: GLCM_ASM computed with "zero" vs "min" binning origin
	// must differ, and each must match its known ground truth.
	static void test_glcm_binning_origin_divergence()
	{
		Fsettings s;
		s.resize((int)NyxSetting::__COUNT__);
		s[(int)NyxSetting::SOFTNAN].rval = 0.0;
		s[(int)NyxSetting::TINY].rval = 0.0;
		s[(int)NyxSetting::SINGLEROI].bval = false;
		s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
		s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
		s[(int)NyxSetting::USEGPU].bval = false;
		s[(int)NyxSetting::VERBOSLVL].ival = 0;
		s[(int)NyxSetting::IBSI].bval = false;
		s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
		GLCMFeature::symmetric_glcm = false;
		GLCMFeature::angles = { 0, 45, 90, 135 };

		int feature = int(Feature2D::GLCM_ASM);

		// --- "zero" origin (MATLAB) ---
		s[(int)NyxSetting::GREYDEPTH].ival = 100;
		s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 100;

		LR roi_zero;
		GLCMFeature f_zero;
		load_masked_test_roi_data(roi_zero, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
			sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
		ASSERT_NO_THROW(f_zero.calculate(roi_zero, s));
		roi_zero.initialize_fvals();
		f_zero.save_value(roi_zero.fvals);
		double val_zero = roi_zero.fvals[feature][0];

		// --- "min" origin (PyRadiomics) ---
		s[(int)NyxSetting::GREYDEPTH].ival = -100;
		s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -100;

		LR roi_min;
		GLCMFeature f_min;
		load_masked_test_roi_data(roi_min, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
			sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
		ASSERT_NO_THROW(f_min.calculate(roi_min, s));
		roi_min.initialize_fvals();
		f_min.save_value(roi_min.fvals);
		double val_min = roi_min.fvals[feature][0];

		// The two binning origins must produce different GLCM values
		ASSERT_NE(val_zero, val_min)
			<< "zero_origin=" << val_zero << " min_origin=" << val_min;

		// Ground truth for GLCM_ASM angle-0 on ibsi_phantom_z1 at 100 bins
		ASSERT_TRUE(agrees_gt(val_zero, 0.148438, 100.))
			<< "zero_origin GLCM_ASM=" << val_zero;
		ASSERT_TRUE(agrees_gt(val_min, 0.140625, 100.))
			<< "min_origin GLCM_ASM=" << val_min;
	}

	// End-to-end: NGTDM_COARSENESS computed with "zero" vs "min" binning origin
	// must differ — exercises the non-GLCM texture feature path (bin_intensities).
	static void test_ngtdm_binning_origin_divergence()
	{
		Fsettings s;
		s.resize((int)NyxSetting::__COUNT__);
		s[(int)NyxSetting::SOFTNAN].rval = 0.0;
		s[(int)NyxSetting::TINY].rval = 0.0;
		s[(int)NyxSetting::SINGLEROI].bval = false;
		s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
		s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
		s[(int)NyxSetting::USEGPU].bval = false;
		s[(int)NyxSetting::VERBOSLVL].ival = 0;
		s[(int)NyxSetting::IBSI].bval = false;

		int feature = int(Feature2D::NGTDM_COARSENESS);

		// --- "zero" origin (MATLAB) ---
		s[(int)NyxSetting::GREYDEPTH].ival = 100;
		NGTDMFeature::n_levels = 0;	// let it use GREYDEPTH

		LR roi_zero;
		NGTDMFeature f_zero;
		load_masked_test_roi_data(roi_zero, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
			sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
		ASSERT_NO_THROW(f_zero.calculate(roi_zero, s));
		roi_zero.initialize_fvals();
		f_zero.save_value(roi_zero.fvals);
		double val_zero = roi_zero.fvals[feature][0];

		// --- "min" origin (PyRadiomics) ---
		s[(int)NyxSetting::GREYDEPTH].ival = -100;
		NGTDMFeature::n_levels = 0;

		LR roi_min;
		NGTDMFeature f_min;
		load_masked_test_roi_data(roi_min, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
			sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
		ASSERT_NO_THROW(f_min.calculate(roi_min, s));
		roi_min.initialize_fvals();
		f_min.save_value(roi_min.fvals);
		double val_min = roi_min.fvals[feature][0];

		// The two binning origins must produce different NGTDM values
		ASSERT_NE(val_zero, val_min)
			<< "zero_origin=" << val_zero << " min_origin=" << val_min;
	}
}
