#pragma once
#include <vector>
#include <unordered_map>
#include "../roi_data.h"
#include "histogram.h"
#include "moments.h"
#include "pixel.h"

class RoiRadius
{
public:
	RoiRadius();
	void initialize (const std::vector<Pixel2>& cloud, const std::vector<Pixel2>& contour);
	std::tuple<double, double, double> get_min_max_median_radius();
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

protected:
	double max_r = 0, mean_r = 0, median_r = 0;
};