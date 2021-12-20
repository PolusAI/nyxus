#pragma once

#include <unordered_map>
#include "../roi_data.h"
#include <tuple>
#include "pixel.h"

class GeodeticLength_and_Thickness
{
public:
	static bool required(const FeatureSet& fs) { return fs.anyEnabled({ GEODETIC_LENGTH, THICKNESS }); }
	GeodeticLength_and_Thickness (size_t roiArea, StatsInt roiPerimeter);
	double get_geodetic_length();
	double get_thickness();
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	double geodetic_length, thickness;
};

