#pragma once
#include <unordered_map>
#include "../roi_data.h"
#include <tuple>
#include "pixel.h"

class GeodeticLength_and_Thickness
{
public:
	GeodeticLength_and_Thickness() {}
	std::tuple<double, double> calculate(StatsInt roiArea, StatsInt roiPerimeter);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
};

