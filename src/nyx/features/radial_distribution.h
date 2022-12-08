#pragma once

#include <vector>
#include <unordered_map>
#include "../feature_method.h"
#include "../roi_cache.h"
#include "pixel.h"

/// @brief Features describing the radial intensity distribution within a ROI - fraction of total stain in an object at a given radius, mean fractional intensity at a given radius, coefficient of variation of intensity within a ring.
class RadialDistributionFeature: public FeatureMethod
{
public:
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled({ FRAC_AT_D, MEAN_FRAC, RADIAL_CV });
	}

	RadialDistributionFeature(); 
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Constants used in the output
	const static int num_bins = 8,
		num_features_FracAtD = 8,
		num_features_MeanFrac = 8,
		num_features_RadialCV = 8;

private:
	// Fraction of total stain in an object at a given radius
	void get_FracAtD();

	// Mean fractional intensity at a given radius (Fraction of total intensity normalized by fraction of pixels at a given radius)
	void get_MeanFrac();

	// Coefficient of variation of intensity within a ring, calculated over 8 slices
	void get_RadialCV();

	// Returns the index of the pixel in parameter 'cloud' having maximum distance from 'contour'
	size_t find_center_NT (const OutOfRamPixelCloud& cloud, const std::vector<Pixel2>& contour);

	// Zeros the counters
	void reset_buffers();

	// Return-ready feature values
	std::vector<double> values_FracAtD,
		values_MeanFrac,
		values_RadialCV;

	// Counters
	std::vector<int> radial_count_bins;
	std::vector<double> radial_intensity_bins;
	std::vector<std::vector<size_t>> banded_wedges;

	// Helpers
	int cached_center_x = -1, 
		cached_center_y = -1;
	int cached_num_pixels = 0;
	const double epsilon = 0.000000001;
};