#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include "sensemaker.h"

// Spatial hashing
inline bool aabbNoOverlap(
	StatsInt xmin1, StatsInt xmax1, StatsInt ymin1, StatsInt ymax1,
	StatsInt xmin2, StatsInt xmax2, StatsInt ymin2, StatsInt ymax2,
	int R)
{
	bool retval = xmin2 - R > xmax1 + R || xmax2 + R < xmin1 - R
		|| ymin2 - R > ymax1 + R || ymax2 + R < ymin1 - R;
	return retval;
}

inline bool aabbNoOverlap(LR& r1, LR& r2, int radius)
{
	bool retval = aabbNoOverlap(r1.aabb.get_xmin(), r1.aabb.get_xmax(), r1.aabb.get_ymin(), r1.aabb.get_ymax(),
		r2.aabb.get_xmin(), r2.aabb.get_xmax(), r2.aabb.get_ymin(), r2.aabb.get_ymax(), radius);
	return retval;
}

inline unsigned long spat_hash_2d(StatsInt x, StatsInt y, int m)
{
	unsigned long h = x * 73856093;
	h = h ^ y * 19349663;
	// hash   hash  z × 83492791	// For the future
	// hash   hash  l × 67867979
	unsigned long retval = h % m;
	return retval;
}

void reduce_neighbors (int radius)
{
#if 0	// Leaving the greedy implementation just for the record and the time when we want to run it on a GPU
	//==== Collision detection, method 1 (best with GPGPU)
	//  Calculate collisions into a triangular matrix
	int nul = uniqueLabels.size();
	std::vector <char> CM(nul * nul, false);	// collision matrix
	// --this loop can be parallel
	for (auto l1 : uniqueLabels)
	{
		LR& r1 = labelData[l1];
		for (auto l2 : uniqueLabels)
		{
			if (l1 == l2)
				continue;	// Trivial - diagonal element
			if (l1 > l2)
				continue;	// No need to check the upper triangle

			LR& r2 = labelData[l2];
			bool noOverlap = r2.aabb.get_xmin() > r1.aabb.get_xmax() + radius || r2.aabb.get_xmax() < r1.aabb.get_xmin() - radius
				|| r2.aabb.get_ymin() > r1.aabb.get_ymax() + radius || r2.aabb.get_ymax() < r1.aabb.get_ymin() - radius;
			if (!noOverlap)
			{
				unsigned int idx = l1 * nul + l2;
				CM[idx] = true;
			}
		}
	}

	// Harvest collision pairs
	for (auto l1 : uniqueLabels)
	{
		LR& r1 = labelData[l1];
		for (auto l2 : uniqueLabels)
		{
			if (l1 == l2)
				continue;	// Trivial - diagonal element
			if (l1 > l2)
				continue;	// No need to check the upper triangle

			unsigned int idx = l1 * nul + l2;
			if (CM[idx] == true)
			{
				LR& r2 = labelData[l2];
				r1.num_neighbors++;
				r2.num_neighbors++;
			}
		}
	}
#endif

	//==== Collision detection, method 2
	int m = 10000;
	std::vector <std::vector<int>> HT(m);	// hash table
	for (auto l : uniqueLabels)
	{
		LR& r = labelData[l];
		auto h1 = spat_hash_2d(r.aabb.get_xmin(), r.aabb.get_ymin(), m),
			h2 = spat_hash_2d(r.aabb.get_xmin(), r.aabb.get_ymax(), m),
			h3 = spat_hash_2d(r.aabb.get_xmax(), r.aabb.get_ymin(), m),
			h4 = spat_hash_2d(r.aabb.get_xmax(), r.aabb.get_ymax(), m);
		HT[h1].push_back(l);
		HT[h2].push_back(l);
		HT[h3].push_back(l);
		HT[h4].push_back(l);
	}

	// Broad phase of collision detection
	for (auto& bin : HT)
	{
		// No need to bother about not colliding labels
		if (bin.size() <= 1)
			continue;

		// Perform the N^2 check
		for (auto& l1 : bin)
		{
			LR& r1 = labelData[l1];

			for (auto& l2 : bin)
			{
				if (l1 < l2)	// Lower triangle 
				{
					LR& r2 = labelData[l2];
					bool overlap = !aabbNoOverlap(r1, r2, radius);
					if (overlap)
					{
						r1.num_neighbors++;
						r2.num_neighbors++;
					}
				}
			}
		}
	}
}

