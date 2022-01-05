#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include <tuple>
#include "pixel.h"

/// @brief The geodetic lengths and thickness are approximated by a rectangle with the same areaand perimeter: area = geodeticlength * thickness; perimeter = 2 * (geodetic_length + thickness).
class GeodeticLength_and_Thickness_features
{
public:
	static bool required(const FeatureSet& fs) { return fs.anyEnabled({ GEODETIC_LENGTH, THICKNESS }); }
	GeodeticLength_and_Thickness_features (size_t roiArea, StatsInt roiPerimeter);
	double get_geodetic_length();
	double get_thickness();
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	double geodetic_length, thickness;
};

