#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>
#include "aabb.h"
#include "pixel.h"
#include "../roi_data.h"

class NeighborFeatures
{
public:
	static bool required(const FeatureSet& fs) { return fs.anyEnabled({ NUM_NEIGHBORS, CLOSEST_NEIGHBOR1_DIST, CLOSEST_NEIGHBOR2_DIST }); }
	NeighborFeatures();
	static void reduce(int radius);

private:
	int collision_radius = 0;
	static bool aabbNoOverlap(
		StatsInt xmin1, StatsInt xmax1, StatsInt ymin1, StatsInt ymax1,
		StatsInt xmin2, StatsInt xmax2, StatsInt ymin2, StatsInt ymax2,
		int R);
	static bool aabbNoOverlap(LR& r1, LR& r2, int radius);
	static unsigned long spat_hash_2d(StatsInt x, StatsInt y, int m);
};