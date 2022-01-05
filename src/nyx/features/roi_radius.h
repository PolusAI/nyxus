#pragma once

#include <vector>
#include <unordered_map>
#include "../roi_cache.h"
#include "histogram.h"
#include "moments.h"
#include "pixel.h"

/// @brief Statistics of ROI pixels' distance to the edge.
class RoiRadius_features
{
public:
	static bool required(const FeatureSet& fs) {
		return fs.anyEnabled({
			ROI_RADIUS_MEAN,
			ROI_RADIUS_MAX,
			ROI_RADIUS_MEDIAN
			});
	}
	RoiRadius_features (const std::vector<Pixel2>& cloud, const std::vector<Pixel2>& contour);
	double get_mean_radius();
	double get_max_radius();
	double get_median_radius();
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	double max_r = 0, mean_r = 0, median_r = 0;
};