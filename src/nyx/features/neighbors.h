#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>
#include "aabb.h"
#include "pixel.h"
#include "../roi_cache.h"
#include "../feature_method.h"

/// @brief Number of neighboring ROIs within specified radius, distance to the closest neighboring ROI, distance to the second closest neighboring ROI.
class NeighborsFeature : public FeatureMethod
{
public:
	NeighborsFeature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);

	// Compatibility with manual reduce
	static void manual_reduce();
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled ({
			NUM_NEIGHBORS,
			PERCENT_TOUCHING,
			CLOSEST_NEIGHBOR1_DIST,
			CLOSEST_NEIGHBOR1_ANG,
			CLOSEST_NEIGHBOR2_DIST,
			CLOSEST_NEIGHBOR2_ANG,
			ANG_BW_NEIGHBORS_MEAN,
			ANG_BW_NEIGHBORS_STDDEV,
			ANG_BW_NEIGHBORS_MODE
			});
	}
private:
	int collision_radius = 0;
	static bool aabbNoOverlap(
		StatsInt xmin1, StatsInt xmax1, StatsInt ymin1, StatsInt ymax1,
		StatsInt xmin2, StatsInt xmax2, StatsInt ymin2, StatsInt ymax2,
		int R);
	static bool aabbNoOverlap(LR& r1, LR& r2, int radius);
	static unsigned long spat_hash_2d(StatsInt x, StatsInt y, int m);
};
