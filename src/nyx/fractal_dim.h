#pragma once

#include <vector>
#include "aabb.h"
#include "pixel.h"

class FractalDimension
{
public:

	FractalDimension() {}

	void initialize (const std::vector<Pixel2>& cloud, const AABB& aabb);
	double get_box_count_fd() { return box_count_fd; }
	double get_perimeter_fd() { return perim_fd; }

protected:
	double box_count_fd = 0, perim_fd = 0;
};