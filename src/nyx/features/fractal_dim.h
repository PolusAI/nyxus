#pragma once

#include <vector>
#include <unordered_map>
#include "../roi_data.h"
#include "aabb.h"
#include "pixel.h"

class FractalDimension
{
public:

	FractalDimension() {}

	void initialize (const std::vector<Pixel2>& cloud, const AABB& aabb);
	double get_box_count_fd() { return box_count_fd; }
	double get_perimeter_fd() { return perim_fd; }

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	double box_count_fd = 0, perim_fd = 0;
};