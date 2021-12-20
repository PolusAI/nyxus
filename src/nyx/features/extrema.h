#pragma once

#include <unordered_map>
#include "../roi_data.h"
#include <tuple>
#include <vector>
#include "pixel.h"

class ExtremaFeatures
{
public:
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled({
			EXTREMA_P1_Y,
			EXTREMA_P1_X,
			EXTREMA_P2_Y,
			EXTREMA_P2_X,
			EXTREMA_P3_Y,
			EXTREMA_P3_X,
			EXTREMA_P4_Y,
			EXTREMA_P4_X,
			EXTREMA_P5_Y,
			EXTREMA_P5_X,
			EXTREMA_P6_Y,
			EXTREMA_P6_X,
			EXTREMA_P7_Y,
			EXTREMA_P7_X,
			EXTREMA_P8_Y,
			EXTREMA_P8_X });
	}
	ExtremaFeatures (const std::vector<Pixel2> & roi_pixels);
	std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> get_values();
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8;
};