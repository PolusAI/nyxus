#pragma once

#include <vector>
#include "aabb.h"
#include "pixel.h"

/// @brief 
/// @param nonzero_intensity_pixels 
/// @param aabb 
/// @param order 
/// @param Z_values 
void zernike2D(
	// in
	std::vector <Pixel2>& nonzero_intensity_pixels,
	AABB& aabb,
	int order,
	// out
	std::vector<double>& Z_values);

void calcRoiZernike (LR& r);
void parallelReduceZernike2D (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

class ZernikeFeatures
{
public:
	static bool required(const FeatureSet& fs) { return fs.isEnabled(ZERNIKE2D);  }
};
