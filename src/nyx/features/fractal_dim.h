#pragma once

#include <vector>
#include <unordered_map>
#include "../roi_cache.h"
#include "aabb.h"
#include "pixel.h"

/// @brief Fractal dimension determined by the box counting and the perimeter methods according to DIN ISO 9276-6 (evenly structured gait).
class FractalDimension_feature
{
public:
	static bool required(const FeatureSet& fs) { return fs.anyEnabled({ FRACT_DIM_BOXCOUNT, FRACT_DIM_PERIMETER }); }
	FractalDimension_feature (const std::vector<Pixel2>& cloud, const AABB& aabb);
	double get_box_count_fd();
	double get_perimeter_fd();

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	double box_count_fd = 0, perim_fd = 0;
};