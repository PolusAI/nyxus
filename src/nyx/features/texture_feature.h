#pragma once

#include <cmath>
#include "image_matrix.h"
#include "image_cube.h"

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

	void bin_intensities (pixData& S, const pixData& I, PixIntens min_I_inten, PixIntens max_I_inten, int greybin_info)
	{
		if (radiomics_grey_binning(greybin_info))
		{
			// radiomics binning
			auto n = I.size();
			for (size_t i = 0; i < n; i++)
				S[i] = to_grayscale_radiomix (I[i], min_I_inten, max_I_inten, std::abs(greybin_info));
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
				S[i] = to_grayscale_radiomix(I[i], min_I_inten, max_I_inten, std::abs(greybin_info));
		}
		else 
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
				S.assign (I.begin(), I.end());
			}
	}

	static PixIntens bin_pixel (PixIntens x, PixIntens min_I_inten, PixIntens max_I_inten, int greybin_info)
	{
		if (radiomics_grey_binning(greybin_info))
		{
			// radiomics binning
			auto y = to_grayscale_radiomix (x, min_I_inten, max_I_inten, std::abs(greybin_info));
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

	// NGLDM's grey level of intensity 'x' of a ROI spanning [min_I_inten, max_I_inten]. The sign of
	// 'greybin_info' selects the kind of scheme as it does for bin_pixel(), 0 keeping the raw level and
	// a negative value taking abs() PyRadiomics bins over [min, max]; IBSI keeps the raw level. A
	// positive value n is NGLDM's own scheme, not bin_pixel()'s MATLAB one: Nyxus::to_grayscale()
	// truncates n x / max, so [0, max] maps onto the n + 1 levels 0..n, where bin_pixel() gives 1..n.
	static PixIntens ngldm_grey_level (PixIntens x, PixIntens min_I_inten, PixIntens max_I_inten, int greybin_info, bool ibsi)
	{
		if (ibsi || ibsi_grey_binning(greybin_info))
			return x;
		if (radiomics_grey_binning(greybin_info))
			return to_grayscale_radiomix (x, min_I_inten, max_I_inten, std::abs(greybin_info));
		return Nyxus::to_grayscale (x, 0, max_I_inten, greybin_info);
	}

	static inline bool matlab_grey_binning (int greybinning_info) { return greybinning_info > 0; }
	static inline bool radiomics_grey_binning (int greybinning_info) { return greybinning_info < 0; }
	static inline bool ibsi_grey_binning (int greybinning_info) { return greybinning_info == 0; }

	// returns 1-based bin indices: IBSI's fixed bin number, floor(n (x - min) / (max - min)) + 1. The
	// product is taken before the division so that a level on a bin edge -- n (x - min) a multiple of
	// (max - min), exact in double for integer levels -- lands in the bin above it, as it does in
	// PyRadiomics and MIRP. A ROI of one intensity has a range of zero width, which is one bin.
	static inline PixIntens to_grayscale_radiomix (PixIntens x, PixIntens min__, PixIntens max__, int binCount)
	{
		if (x)
		{
			if (max__ <= min__)
				return 1;
			PixIntens y = (PixIntens) (std::floor (double(binCount) * double(x - min__) / double(max__ - min__)) + 1);
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