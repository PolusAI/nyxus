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
#include "../globals.h"
#include "../environment.h"
#include "neighbors.h"

NeighborsFeature::NeighborsFeature(): FeatureMethod("NeighborsFeature")
{
	provide_features({
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

void NeighborsFeature::calculate(LR& r)
{}

void NeighborsFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} 

void NeighborsFeature::osized_calculate(LR& r, ImageLoader& imloader)
{
	calculate(r);
}

/// @brief All the logic is in parallel_process()
/// @param feature_vals 
void NeighborsFeature::save_value(std::vector<std::vector<double>>& feature_vals) {}

/// @brief All the logic is in parallel_process()
/// @param start 
/// @param end 
/// @param ptrLabels 
/// @param ptrLabelData 
void NeighborsFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData) {}

// Calculates the features using spatial hashing approach (indirectly)
void NeighborsFeature::parallel_process (std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	manual_reduce();
}

// Calculates the features using spatial hashing approach
void NeighborsFeature::manual_reduce()
{
	int radius = theEnvironment.get_pixel_distance();

	// Keeping the commented out greedy implementation just for the record and the time when we want to run it on a GPU
	//==== Collision detection, method 1 (best with GPGPU)
	//  Calculate collisions into a triangular matrix
	int nul = uniqueLabels.size();
	
	std::vector <int> LabsVec;
	LabsVec.reserve (uniqueLabels.size());
	LabsVec.insert (LabsVec.end(), uniqueLabels.begin(), uniqueLabels.end());

	std::vector <char> CM ((nul+1) * (nul+1), false);	// collision matrix
	// --this loop can be parallel
	for (size_t i1 = 0; i1 < nul; i1++) // for (auto l1 : uniqueLabels)
	{
		auto l1 = LabsVec[i1];
		LR& r1 = roiData[l1];

		for (size_t i2 = 0; i2 < nul; i2++) // for (auto l2 : uniqueLabels)
		{
			auto l2 = LabsVec[i1];
			if (l1 == l2)
				continue;	// Trivial - diagonal element
			if (l1 > l2)
				continue;	// No need to check the upper triangle

			LR& r2 = roiData[l2];
			bool noOverlap = r2.aabb.get_xmin() > r1.aabb.get_xmax() || r2.aabb.get_xmax() < r1.aabb.get_xmin() 
				|| r2.aabb.get_ymin() > r1.aabb.get_ymax() || r2.aabb.get_ymax() < r1.aabb.get_ymin() ;
			if (! noOverlap)
			{
				unsigned int idx = l1 * nul + l2;
				CM[idx] = true;
			}
		}
	}

	// Harvest collision pairs
	size_t radius2 = radius * radius;
	for (size_t i1 = 0; i1 < nul; i1++) // for (auto l1 : uniqueLabels)
	{
		auto l1 = LabsVec[i1];
		LR& r1 = roiData[l1];

		for (size_t i2 = 0; i2 < nul; i2++) // for (auto l2 : uniqueLabels)
		{
			auto l2 = LabsVec[i1];			
			if (l1 == l2)
				continue;	// Trivial - diagonal element
			if (l1 > l2)
				continue;	// No need to check the upper triangle

			unsigned int idx = l1 * nul + l2;
			if (CM[idx] == true)
			{
				// Check if these labels are close enough
				LR& r2 = roiData[l2];

				// Iterate r1's contour pixels
				auto [mind, maxd] = r1.contour[0].min_max_sqdist(r2.contour);
				size_t n_touchingContourPixels = 0;
				for (auto& cp : r1.contour)
				{
					auto [minD, maxD] = cp.min_max_sqdist(r2.contour);
					mind = std::min(mind, minD);

					//--We aren't interested in max distance-->	maxd = std::max(maxd, maxD);

					// Maintain touching pixels stats
					if (minD == 0) // (minD <= radius2)
						n_touchingContourPixels++;
				}

				// Check versus the radius
				if (mind > radius2)
					continue;

				// Save partial statis of r1's touching pixel stats
				r1.fvals[PERCENT_TOUCHING][0] += n_touchingContourPixels;

				// Definitely neigbors
				r1.fvals[NUM_NEIGHBORS][0]++;
				r2.fvals[NUM_NEIGHBORS][0]++;
				r1.aux_neighboring_labels.push_back (l2);
				r2.aux_neighboring_labels.push_back (l1);
			}
		}

		// Finalize the % touching calculation
		r1.fvals[PERCENT_TOUCHING][0] = double(r1.fvals[PERCENT_TOUCHING][0]) / double(r1.contour.size());
	}

#if 0
	//==== Collision detection, method 2

	// Hash table
	int m = 100;
	std::vector <std::vector<int>> HT(m);

	for (auto l : Nyxus::uniqueLabels)
	{
		LR& r = Nyxus::roiData[l];

		/*
		auto h1 = spat_hash_2d(r.aabb.get_xmin(), r.aabb.get_ymin(), m),
			h2 = spat_hash_2d(r.aabb.get_xmin(), r.aabb.get_ymax(), m),
			h3 = spat_hash_2d(r.aabb.get_xmax(), r.aabb.get_ymin(), m),
			h4 = spat_hash_2d(r.aabb.get_xmax(), r.aabb.get_ymax(), m);
		HT[h1].push_back(l);
		HT[h2].push_back(l);
		HT[h3].push_back(l);
		HT[h4].push_back(l);
		*/

		long x, y, x1, x2, y1, y2, ns = 10, step;
		// horizontal 1
		y = r.aabb.get_ymin();
		x1 = r.aabb.get_xmin();
		x2 = r.aabb.get_xmax();
		step = (x2 - x1) / ns;
		for (int i = 0; i < ns; i++)
		{
			x = x1 + i * step;
			auto h = spat_hash_2d(x, y, m);
			HT [h].push_back(l);
		}

		// horizontal 2
		y = r.aabb.get_ymax();
		x1 = r.aabb.get_xmin();
		x2 = r.aabb.get_xmax();
		step = (x2 - x1) / ns;
		for (int i = 0; i < ns; i++)
		{
			x = x1 + i * step;
			auto h = spat_hash_2d(x, y, m);
			HT[h].push_back(l);
		}

		// vertical 1
		x = r.aabb.get_xmin();
		y1 = r.aabb.get_ymin(), y2 = r.aabb.get_ymax();
		ns = 10;
		step = (y2 - y1) / ns;
		for (int i = 0; i < ns; i++)
		{
			y = y1 + i * step;
			auto h = spat_hash_2d (x, y, m);
			HT[h].push_back(l);
		}

		// vertical 2
		x = r.aabb.get_xmax();
		y1 = r.aabb.get_ymin(), y2 = r.aabb.get_ymax();
		step = (y2 - y1) / ns;
		for (int i = 0; i < ns; i++)
		{
			y = y1 + i * step;
			auto h = spat_hash_2d(x, y, m);
			HT[h].push_back(l);
		}
	}

	// Broad phase of collision detection
	for (auto& bin : HT)
	{
		// No need to bother about not colliding ROIs
		if (bin.size() <= 1)
			continue;

		// Perform the N^2 check
		for (auto& l1 : bin)
		{
			LR& r1 = Nyxus::roiData[l1];

			for (auto& l2 : bin)
			{
				if (l1 < l2)	// Lower triangle 
				{
					LR& r2 = Nyxus::roiData[l2];
					bool overlap = !aabbNoOverlap(r1, r2, radius);
					if (overlap)
					{
						// l1's neighbors
						r1.fvals[NUM_NEIGHBORS][0]++;
						r1.aux_neighboring_labels.push_back(l2);

						// l2's neighbors
						r2.fvals[NUM_NEIGHBORS][0]++;
						r2.aux_neighboring_labels.push_back(l1);
					}
				}
			}
		}
	}
#endif

	// Closest neighbors
	for (auto l : Nyxus::uniqueLabels)
	{
		LR& r = Nyxus::roiData[l];
		int n_neigs = int(r.fvals[NUM_NEIGHBORS][0]);

		// Any neighbors of this ROI ?
		if (n_neigs == 0)
			continue;

		double cenx = r.fvals[CENTROID_X][0],
			ceny = r.fvals[CENTROID_Y][0];

		std::vector<double> dists;
		for (auto l_neig : r.aux_neighboring_labels)
		{
			LR& r_neig = Nyxus::roiData[l_neig];
			double cenx_n = r_neig.fvals[CENTROID_X][0],
				ceny_n = r_neig.fvals[CENTROID_Y][0],
				dx = cenx - cenx_n,
				dy = ceny - ceny_n,
				dist = dx * dx + dy * dy;
			dists.push_back(dist);
		}

		// Find idx of minimum
		auto ite1st = std::min_element(dists.begin(), dists.end());
		auto closest_1_idx = std::distance(dists.begin(), ite1st);
		auto closest1label = r.aux_neighboring_labels[closest_1_idx];

		// Save distance to neighbor #1
		r.fvals[CLOSEST_NEIGHBOR1_DIST][0] = dists[closest_1_idx];

		// Save angle with neighbor #1
		LR& r1 = Nyxus::roiData[closest1label];
		r.fvals[CLOSEST_NEIGHBOR1_ANG][0] = angle(cenx, ceny, r1.fvals[CENTROID_X][0], r1.fvals[CENTROID_X][0]);

		// Find idx of 2nd minimum
		if (n_neigs > 1)
		{
			auto lambSkip1st = [&ite1st](double a, double b)
			{
				return ((b != (*ite1st)) && (a > b));
			};
			auto ite2nd = std::min_element(dists.begin(), dists.end(), lambSkip1st);
			auto closest_2_idx = std::distance(dists.begin(), ite2nd);
			auto closest2label = r.aux_neighboring_labels[closest_2_idx];

			// Save distance to neighbor #2
			r.fvals[CLOSEST_NEIGHBOR2_DIST][0] = dists[closest_2_idx];

			// Save angle with neighbor #2
			LR& r2 = Nyxus::roiData[closest2label];
			r.fvals[CLOSEST_NEIGHBOR2_ANG][0] = angle(cenx, ceny, r2.fvals[CENTROID_X][0], r2.fvals[CENTROID_X][0]);
		}
	}

	// Angle between neigbors
	Moments2 mom2;
	std::vector<int> anglesRounded;
	for (auto l : Nyxus::uniqueLabels)
	{
		LR& r = Nyxus::roiData[l];
		int n_neigs = int(r.fvals[NUM_NEIGHBORS][0]);

		// Any neighbors of this ROI ?
		if (n_neigs == 0)
			continue;

		double cenx = r.fvals[CENTROID_X][0],
			ceny = r.fvals[CENTROID_Y][0];

		// Iterate all the neighbors
		for (auto l_neig : r.aux_neighboring_labels)
		{
			LR& r_neig = Nyxus::roiData[l_neig];
			double cenx_n = r_neig.fvals[CENTROID_X][0],
				ceny_n = r_neig.fvals[CENTROID_Y][0];

			double ang = angle (cenx, ceny, cenx_n, ceny_n);	// radians
			ang = ang * 180.0;	// degrees because we will later need angles' mode
			mom2.add(ang);
			anglesRounded.push_back(ang);
		}

		r.fvals[ANG_BW_NEIGHBORS_MEAN][0] = mom2.mean();
		r.fvals[ANG_BW_NEIGHBORS_STDDEV][0] = mom2.std();
		r.fvals[ANG_BW_NEIGHBORS_MODE][0] = mode(anglesRounded);
	}
}

// Spatial hashing
inline bool NeighborsFeature::aabbNoOverlap(
	StatsInt xmin1, StatsInt xmax1, StatsInt ymin1, StatsInt ymax1,
	StatsInt xmin2, StatsInt xmax2, StatsInt ymin2, StatsInt ymax2,
	int R)
{
	bool retval = xmin2 - R > xmax1 + R || xmax2 + R < xmin1 - R
		|| ymin2 - R > ymax1 + R || ymax2 + R < ymin1 - R;
	return retval;
}

inline bool NeighborsFeature::aabbNoOverlap (LR& r1, LR& r2, int radius)
{
	bool retval = aabbNoOverlap(r1.aabb.get_xmin(), r1.aabb.get_xmax(), r1.aabb.get_ymin(), r1.aabb.get_ymax(),
		r2.aabb.get_xmin(), r2.aabb.get_xmax(), r2.aabb.get_ymin(), r2.aabb.get_ymax(), radius);
	return retval;
}

inline unsigned long NeighborsFeature::spat_hash_2d(StatsInt x, StatsInt y, int m)
{
	unsigned long h = x * 73856093;
	h = h ^ y * 19349663;
	// hash   hash  z × 83492791	// For the future
	// hash   hash  l × 67867979
	unsigned long retval = h % m;
	return retval;
}


