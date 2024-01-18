#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include <tuple>
#include "pixel.h"
#include "../feature_method.h"

/// @brief The geodetic lengths and thickness are approximated by a rectangle with the same area and perimeter: area = geodeticlength * thickness; perimeter = 2 * (geodetic_length + thickness).
class GeodeticLengthThicknessFeature:public FeatureMethod
{
public:
	GeodeticLengthThicknessFeature();
	
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	static bool required(const FeatureSet& fs);
private:
	double geodetic_length = 0, thickness = 0;
};

