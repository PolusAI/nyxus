#pragma once

#include <vector>
#include "f_convex_hull.h"
#include "pixel.h"

class RadialDistribution
{
public:

	RadialDistribution() 
	{ 
		radial_count_bins.resize (RadialDistribution::num_bins, 0); 
		radial_intensity_bins.resize (RadialDistribution::num_bins, 0.0);
		angular_bins.resize (RadialDistribution::num_bins, 0);
		band_pixels.resize (RadialDistribution::num_bins);

		values_FracAtD.resize (RadialDistribution::num_bins, 0);
		values_MeanFrac.resize (RadialDistribution::num_bins, 0);
		values_RadialCV.resize (RadialDistribution::num_bins, 0);
	}

	void initialize (const std::vector<Pixel2>& raw_pixels, const std::vector<Pixel2>& contour_pixels);

	// Fraction of total stain in an object at a given radius
	const std::vector<double>& get_FracAtD();

	// Mean fractional intensity at a given radius (Fraction of total intensity normalized by fraction of pixels at a given radius)
	const std::vector<double>& get_MeanFrac();

	// Coefficient of variation of intensity within a ring, calculated over 8 slices
	const std::vector<double> & get_RadialCV();

	const static int num_bins = 8,
		num_features_FracAtD = 8, 
		num_features_MeanFrac = 8, 
		num_features_RadialCV = 8;

protected:

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