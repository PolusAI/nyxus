#pragma once

#include "image_matrix.h"
#include "image_cube.h"

class TextureFeature
{
public:

	TextureFeature() {}

	static inline int cast_to_range(PixIntens orig_I, PixIntens min_orig_I, PixIntens max_orig_I, int min_target_I, int max_target_I)
	{
		int target_I = (int)(double(orig_I - min_orig_I) / double(max_orig_I - min_orig_I) * double(max_target_I - min_target_I) + min_target_I);
		return target_I;
	}

	void bin_intensities (pixData& S, const pixData& I, PixIntens min_I_inten, PixIntens max_I_inten, int greybin_info)
	{
		if (radiomics_grey_binning(greybin_info))
		{
			// radiomics binning
			auto n = I.size();
			for (size_t i = 0; i < n; i++)
				S[i] = to_grayscale_radiomix (I[i], min_I_inten);
			return;
		}
		if (matlab_grey_binning(greybin_info))
		{
			// matlab binning
			auto n = I.size();
			int n_matlab_levels = greybin_info;

			prep_bin_array_matlab (max_I_inten, n_matlab_levels);
			for (size_t i = 0; i < n; i++)
				S[i] = bin_array_matlab (I[i]);
		}
		else
		{
			// no binning (IBSI)
			auto n = I.size();
			for (size_t i = 0; i < n; i++)
				S[i] = I[i];
		}
	}

	void bin_intensities_3d (SimpleCube<PixIntens> & S, const SimpleCube<PixIntens> & I, PixIntens min_I_inten, PixIntens max_I_inten, int greybin_info)
	{
		if (radiomics_grey_binning(greybin_info))
		{
			// radiomics binning
			auto n = I.size();
			for (size_t i = 0; i < n; i++)
				S[i] = to_grayscale_radiomix(I[i], min_I_inten);
			return;
		}
		if (matlab_grey_binning(greybin_info))
		{
			// matlab binning
			auto n = I.size();
			int n_matlab_levels = greybin_info;

			prep_bin_array_matlab(max_I_inten, n_matlab_levels);
			for (size_t i = 0; i < n; i++)
				S[i] = bin_array_matlab(I[i]);
		}
		else
		{
			// no binning (IBSI)
			auto n = I.size();
			for (size_t i = 0; i < n; i++)
				S[i] = I[i];
		}
	}

	static PixIntens bin_pixel (PixIntens x, PixIntens min_I_inten, PixIntens max_I_inten, int greybin_info)
	{
		if (radiomics_grey_binning(greybin_info))
		{
			// radiomics binning
			auto y = to_grayscale_radiomix (x, min_I_inten);
			return y;
		}
		else
		if (matlab_grey_binning(greybin_info))
		{
			// matlab binning
			int n_matlab_levels = greybin_info;
			auto y = bin_pixel_matlab(x, max_I_inten, n_matlab_levels); //to_grayscale_matlab (x, n_matlab_levels);
			return y;
		}
		else
		{
			// no binning (IBSI)
			return x;
		}
	}

	static inline bool matlab_grey_binning (int greybinning_info) { return greybinning_info > 0; }
	static inline bool radiomics_grey_binning (int greybinning_info) { return greybinning_info < 0; }
	static inline bool ibsi_grey_binning (int greybinning_info) { return greybinning_info == 0; }

	static double radiomics_bin_width;	// default: 25

	//---------------------- binning by Leijenaar RTH, Nalbantov G, Carvalho et al. (PyRadiomics)
	static inline PixIntens to_grayscale_radiomix (PixIntens x, PixIntens min__)
	{
		if (x)
		{
			PixIntens y = (unsigned int)(double(x - min__) / TextureFeature::radiomics_bin_width) + 1;
			return y;
		}
		else
			return 0;
	}

private:

	//---------------------- Matlab binning --------------------------------
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