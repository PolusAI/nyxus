#pragma once

#include <vector>
#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include "histogram.h"
#include "moments.h"
#include "pixel.h"
#include "../feature_method.h"
#include "../feature_settings.h"

/// @brief Statistics of ROI pixels' distance to the edge.
class RoiRadiusFeature: public FeatureMethod
{
public:
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::ROI_RADIUS_MEAN,
		Nyxus::Feature2D::ROI_RADIUS_MAX,
		Nyxus::Feature2D::ROI_RADIUS_MEDIAN
	};

	RoiRadiusFeature();
	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void extract (LR& roi, const Fsettings& s);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);

	// Compatibility with manual reduce
	static bool required (const FeatureSet& fs) 
	{
		return fs.anyEnabled (RoiRadiusFeature::featureset);
	}

private:
	double max_r = 0, mean_r = 0, median_r = 0;
};