#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>
#include "aabb.h"
#include "../globals.h"
#include "pixel.h"
#include "../roi_cache.h"
#include "../feature_method.h"

/// @brief Number of neighboring ROIs within specified radius, distance to the closest neighboring ROI, distance to the second closest neighboring ROI.
class NeighborsFeature : public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset = {
		Nyxus::Feature2D::NUM_NEIGHBORS,
		Nyxus::Feature2D::PERCENT_TOUCHING,
		Nyxus::Feature2D::CLOSEST_NEIGHBOR1_DIST,
		Nyxus::Feature2D::CLOSEST_NEIGHBOR1_ANG,
		Nyxus::Feature2D::CLOSEST_NEIGHBOR2_DIST,
		Nyxus::Feature2D::CLOSEST_NEIGHBOR2_ANG,
		Nyxus::Feature2D::ANG_BW_NEIGHBORS_MEAN,
		Nyxus::Feature2D::ANG_BW_NEIGHBORS_STDDEV,
		Nyxus::Feature2D::ANG_BW_NEIGHBORS_MODE
	};

	NeighborsFeature();

	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings& s);

	// Compatibility with manual reduce
	static void manual_reduce (	
		// out
		Nyxus::Roidata& roiData,
		// in
		const Fsettings& s,
		const std::unordered_set<int>& uniqueLabels);
	static bool required(const FeatureSet& fs);

private:
	int collision_radius = 0;
	static bool aabbNoOverlap(
		StatsInt xmin1, StatsInt xmax1, StatsInt ymin1, StatsInt ymax1,
		StatsInt xmin2, StatsInt xmax2, StatsInt ymin2, StatsInt ymax2,
		int R);
	static bool aabbNoOverlap(LR& r1, LR& r2, int radius);
	static unsigned long spat_hash_2d(StatsInt x, StatsInt y, int m);
};