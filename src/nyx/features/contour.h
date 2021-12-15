#pragma once

#include "image_matrix.h"

class Contour
{
public:
	Contour();
	void calculate (const ImageMatrix& im);	// Leaves result in 'contour_pixels'
	StatsInt get_roi_perimeter();
	StatsReal get_diameter_equal_perimeter();
	std::tuple<StatsReal, StatsReal, StatsReal, StatsReal> get_min_max_mean_stddev_intensity();
	
	std::vector<Pixel2> contour_pixels;
};

