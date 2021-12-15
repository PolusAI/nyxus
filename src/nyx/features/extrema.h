#pragma once

#include <unordered_map>
#include "../roi_data.h"
#include <tuple>
#include <vector>
#include "pixel.h"

class ExtremaFeatures
{
public:
	ExtremaFeatures();
	void initialize (const std::vector<Pixel2> & roi_pixels);
	std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> get_values();
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8;
};