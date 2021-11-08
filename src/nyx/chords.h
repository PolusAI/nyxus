#pragma once

#include <tuple>
#include <vector>
#include "aabb.h"
#include "pixel.h"

class Chords
{
public:
	Chords() {}

	void initialize(const std::vector<Pixel2> & raw_pixels, const AABB & bb, const double cenx, const double ceny);

	// Returns
	// --------
	//	max
	//	min
	//	median
	//	mean
	//	mode
	//	std
	//	min_angle
	//	max_angle
	//
	std::tuple<double, double, double, double, double, double, double, double> get_maxchords_stats();

	// Returns
	// --------
	//	max
	//	min
	//	median
	//	mean
	//	mode
	//	std
	//	min_angle
	//	max_angle
	//
	std::tuple<double, double, double, double, double, double, double, double> get_allchords_stats();

protected:

	double
		allchords_max = 0,
		allchords_min = 0,
		allchords_median = 0,
		allchords_mean = 0,
		allchords_mode = 0,
		allchords_stddev = 0,
		allchords_min_angle = 0,
		allchords_max_angle = 0;

	double
		maxchords_max = 0,
		maxchords_min = 0,
		maxchords_median = 0,
		maxchords_mean = 0,
		maxchords_mode = 0,
		maxchords_stddev = 0,
		maxchords_min_angle = 0,
		maxchords_max_angle = 0;
};