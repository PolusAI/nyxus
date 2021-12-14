#pragma once

#include <vector>
#include "aabb.h"
#include "pixel.h"

void zernike2D(
	// in
	std::vector <Pixel2>& nonzero_intensity_pixels,
	AABB& aabb,
	int order,
	// out
	std::vector<double>& Z_values);
