#pragma once

#include <algorithm>
#include <vector>
#include "pixel.h"

/// @brief Growable union-find over connected-component labels, shared by the out-of-core
/// GLSZM/GLDZM streaming connected-component labeling (see docs/13-3d-ooc-glszm-gldzm-design.md).
///
/// A label is created for each newly-discovered zone (make_label) and merged with another when a
/// streaming scan discovers two previously-separate labels actually belong to the same zone
/// (union_sum/union_min). Each label carries a running metric -- the zone's voxel count for GLSZM
/// (combined by SUM on union) or its minimum distance-to-border for GLDZM (combined by MIN) -- so
/// after the whole volume streams once, each root label directly holds its zone's final metric; no
/// second pass over voxels is needed.
class LabelUnionFind
{
public:
	int make_label (PixIntens intensity, long long initial_metric)
	{
		parent_.push_back ((int) parent_.size());
		rank_.push_back (0);
		metric_.push_back (initial_metric);
		intensity_.push_back (intensity);
		return (int) parent_.size() - 1;
	}

	int find (int x)
	{
		while (parent_[x] != x)
		{
			parent_[x] = parent_[parent_[x]];	// path halving
			x = parent_[x];
		}
		return x;
	}

	// Merge labels a and b, combining their metric as a SUM (GLSZM zone size)
	int union_sum (int a, int b) { return union_impl (a, b, true); }
	// Merge labels a and b, combining their metric as a MIN (GLDZM zone distance)
	int union_min (int a, int b) { return union_impl (a, b, false); }

	// Add this voxel to an existing zone's running size (GLSZM)
	void add_sum (int label, long long delta) { int r = find (label); metric_[r] += delta; }
	// Fold this voxel's distance into an existing zone's running minimum (GLDZM)
	void update_min (int label, long long val) { int r = find (label); metric_[r] = (std::min)(metric_[r], val); }

	long long metric (int label) { return metric_[find (label)]; }
	PixIntens intensity_of (int label) const { return intensity_[label]; }
	size_t num_labels() const { return parent_.size(); }

private:
	int union_impl (int a, int b, bool sumMode)
	{
		int ra = find (a), rb = find (b);
		if (ra == rb) return ra;
		if (rank_[ra] < rank_[rb]) std::swap (ra, rb);
		parent_[rb] = ra;
		if (rank_[ra] == rank_[rb]) rank_[ra]++;
		metric_[ra] = sumMode ? (metric_[ra] + metric_[rb]) : (std::min)(metric_[ra], metric_[rb]);
		return ra;
	}

	std::vector<int> parent_, rank_;
	std::vector<long long> metric_;
	std::vector<PixIntens> intensity_;
};
