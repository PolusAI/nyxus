#pragma once

#include <vector>
#include <unordered_map>
#include "../roi_cache.h"
#include "convex_hull.h"
#include "pixel.h"

/// @brief Features describing the radial intensity distribution within a ROI - fraction of total stain in an object at a given radius, mean fractional intensity at a given radius, coefficient of variation of intensity within a ring.
class RadialDistribution_features
{
public:
	static bool required(const FeatureSet& fs) {
		return fs.anyEnabled({ FRAC_AT_D, MEAN_FRAC, RADIAL_CV });
	}

	RadialDistribution_features (const std::vector<Pixel2>& raw_pixels, const std::vector<Pixel2>& contour_pixels);

	// Fraction of total stain in an object at a given radius
	const std::vector<double>& get_FracAtD();

	// Mean fractional intensity at a given radius (Fraction of total intensity normalized by fraction of pixels at a given radius)
	const std::vector<double>& get_MeanFrac();

	// Coefficient of variation of intensity within a ring, calculated over 8 slices
	const std::vector<double> & get_RadialCV();

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Used in the output
	const static int num_bins = 8,
		num_features_FracAtD = 8,
		num_features_MeanFrac = 8,
		num_features_RadialCV = 8;

private:
	std::vector<double> values_FracAtD,
		values_MeanFrac,
		values_RadialCV;

	std::vector<int> radial_count_bins;
	std::vector<double> radial_intensity_bins;
	std::vector<int> angular_bins;
	std::vector<std::vector<Pixel2>> band_pixels;
	int cached_center_x = -1, 
		cached_center_y = -1;

	int cached_num_pixels = 0;
};