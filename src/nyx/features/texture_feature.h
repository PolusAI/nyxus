#pragma once

#include "image_matrix.h"
#include "image_cube.h"
#include "../feature_settings.h"

struct AngleShift
{
	int dz, dy, dx;
};

class TextureFeature
{
public:

	TextureFeature() {}

	static inline int cast_to_range(PixIntens orig_I, PixIntens min_orig_I, PixIntens max_orig_I, int min_target_I, int max_target_I)
	{
		int target_I = (int)(double(orig_I - min_orig_I) / double(max_orig_I - min_orig_I) * double(max_target_I - min_target_I) + min_target_I);
		return target_I;
	}

	// Bin intensities using the explicit BinningOrigin enum.
	// Replaces the former tri-state integer encoding (greybin_info > 0 = matlab,
	// < 0 = radiomics, == 0 = IBSI) with clearer dispatch: n_levels==0 means IBSI
	// (no binning), then BinningOrigin selects min-based (PyRadiomics) vs zero-based
	// (Nyxus/MATLAB) binning. Dispatch order: IBSI first because n_levels==0 is
	// unambiguous, then min_based, then zero as the default.
	void bin_intensities(
		pixData& S,
		const pixData& I,
		PixIntens min_I_inten,
		PixIntens max_I_inten,
		int n_levels,
		BinningOrigin bo)
	{
		if (n_levels == 0)
		{
			// no binning (IBSI)
			std::copy(I.begin(), I.end(), S.begin());
			return;
		}

		auto out = S.begin();

		if (bo == BinningOrigin::min_based)
		{
			// radiomics binning
			for (auto v : I)
				*out++ = to_grayscale_radiomix(v, min_I_inten, max_I_inten, n_levels);
			return;
		}

		// matlab binning (BinningOrigin::zero)
		prep_bin_array_matlab(max_I_inten, n_levels);
		for (auto v : I)
			*out++ = bin_array_matlab(v);
	}

	void bin_intensities_3d(
		SimpleCube<PixIntens>& S,
		const SimpleCube<PixIntens>& I,
		PixIntens min_I_inten,
		PixIntens max_I_inten,
		int n_levels,
		BinningOrigin bo)
	{
		if (n_levels == 0)
		{
			// no binning (IBSI)
			S.assign(I.begin(), I.end());
			return;
		}

		auto out = S.begin();

		if (bo == BinningOrigin::min_based)
		{
			// radiomics binning
			for (auto v : I)
				*out++ = to_grayscale_radiomix(v, min_I_inten, max_I_inten, n_levels);
			return;
		}

		// matlab binning (BinningOrigin::zero)
		prep_bin_array_matlab(max_I_inten, n_levels);
		for (auto v : I)
			*out++ = bin_array_matlab(v);
	}

	static PixIntens bin_pixel (PixIntens x, PixIntens min_I_inten, PixIntens max_I_inten, int n_levels, BinningOrigin bo)
	{
		if (n_levels == 0)
		{
			// no binning (IBSI)
			return x;
		}
		if (bo == BinningOrigin::min_based)
		{
			// radiomics binning
			return to_grayscale_radiomix (x, min_I_inten, max_I_inten, n_levels);
		}
		// matlab binning (BinningOrigin::zero)
		return bin_pixel_matlab(x, max_I_inten, n_levels);
	}

	// returns 1-based bin indices
	static inline PixIntens to_grayscale_radiomix (PixIntens x, PixIntens min__, PixIntens max__, int binCount)
	{
		if (x)
		{
			double binW = double(max__ - min__) / double(binCount);
			PixIntens y = (PixIntens) (double(x - min__) / binW + 1);
			if (y > binCount)
				y = binCount;	// the last bin is +1 unit wider
			return y;
		}
		else
			return 0;
	}

	// 'afv' is angled feature values
	double calc_ave (const std::vector<double>& afv)
	{
		if (afv.empty())
			return 0;

		double n = static_cast<double> (afv.size()),
			ave = std::reduce(afv.begin(), afv.end()) / n;

		return ave;
	}

private:

	// Matlab binning 
	unsigned int cached_n_levels;	// initialized in prep_grayscale_binning(), referenced in to_grayscale_2024_v2()
	double slope = 0.;
	double intercept = 0.;
	inline void prep_bin_array_matlab (unsigned int max_i, unsigned int target_n_levels)
	{
		cached_n_levels = target_n_levels;

		double min_i = 0.;
		slope = double(target_n_levels) / (double(max_i) - min_i);
		intercept = 1. - slope * min_i;
	}

	//
	//!!! Need to route it via scale_pixel_matlab_imp (unsigned int x, double slope_, double intercept_, int n_levels)
	//
	inline PixIntens bin_array_matlab (PixIntens i)
	{
		// 0 is trivially 1
		if (i == 0)
			return 1;

		// scale
		double scaled_real_i = std::floor(slope * double(i) + intercept);
		PixIntens scaled_i = (PixIntens) scaled_real_i;

		// clip values outside the range
		if (scaled_i > cached_n_levels)
			scaled_i = cached_n_levels;
		if (scaled_i < 1)
			scaled_i = 1;

		return scaled_i;
	}

	static inline unsigned int scale_pixel_matlab_imp (unsigned int x, double slope_, double intercept_, int n_levels)
	{
		// 0 is trivially 1
		if (x == 0)
			return 1;

		// scale
		double scaled_real_i = std::floor(slope_ * double(x) + intercept_);
		unsigned int scaled_i = (unsigned int)scaled_real_i;

		// clip values outside the range
		if (scaled_i > (PixIntens) n_levels)
			scaled_i = n_levels;
		if (scaled_i < 1)
			scaled_i = 1;

		return scaled_i;
	}

	static inline PixIntens bin_pixel_matlab(PixIntens x, PixIntens max_i, int greybin_info)
	{
		auto target_n_levels = greybin_info;
		double min_i = 0.;
		double slope = double(target_n_levels) / (double(max_i) - min_i);
		double intercept = 1. - slope * min_i;
		auto y = scale_pixel_matlab_imp(x, slope, intercept, target_n_levels);
		return y;
	}
};