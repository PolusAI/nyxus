#include <algorithm>
#include <cmath>
#include "roi_radius.h"

using namespace Nyxus;

namespace
{
	// Mean, max and median of the per-pixel distances to the ROI's contour. Shared by the in-RAM and
	// out-of-core paths so the two cannot report different statistics of the same distances.
	void radius_stats (std::vector<double>& dists, double& mean_r, double& max_r, double& median_r)
	{
		if (dists.empty())
		{
			mean_r = max_r = median_r = 0.0;
			return;
		}

		Moments2 mom2;
		for (auto d : dists)
			mom2.add (d);
		mean_r = mom2.mean();
		max_r = mom2.max__();

		// Median over the distances directly. They are real-valued, so this cannot go through
		// TrivialHistogram, whose item type is 'unsigned int' and would round every radius to a
		// whole pixel.
		size_t n = dists.size();
		std::sort (dists.begin(), dists.end());
		median_r = n % 2 ? dists[n / 2] : (dists[n / 2 - 1] + dists[n / 2]) / 2.0;
	}
}

RoiRadiusFeature::RoiRadiusFeature() : FeatureMethod("RoiRadiusFeature")
{
	provide_features ({Feature2D::ROI_RADIUS_MEAN, Feature2D::ROI_RADIUS_MAX, Feature2D::ROI_RADIUS_MEDIAN});
	add_dependencies ({ Feature2D::PERIMETER});
}

void RoiRadiusFeature::calculate (LR& r, const Fsettings& s)
{
	const std::vector<Pixel2>& cloud = r.raw_pixels;

	std::vector<Pixel2> K;
	r.merge_multicontour (K);

	std::vector<double> dists;
	dists.reserve (cloud.size());
	for (auto& pxA : cloud)
		// The pixel's distance to the contour. exact_min_sqdist() rather than min_sqdist(): the
		// latter is an approximate hill-descent that assumes a locally-unimodal ordered contour and
		// can settle in the wrong basin on a closed one, returning more than the true minimum. The
		// exact scan is O(contour) per pixel against the descent's O(log contour), which on a filled
		// disk is 1.1x the time at 8k pixels and 8.2x at 500k; a seeded distance transform would be
		// exact at O(pixels) and is where to go if a caller needs that back.
		dists.push_back (std::sqrt (pxA.exact_min_sqdist (K)));

	radius_stats (dists, mean_r, max_r, median_r);
}

void RoiRadiusFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void RoiRadiusFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	const auto& cloud = r.raw_pixels_NT; 

	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	std::vector<double> dists;
	dists.reserve (cloud.size());
	for (size_t i=0; i<cloud.size(); i++) 
	{
		Pixel2 pxA = cloud.get_at(i);
		// The same quantity calculate() takes, by the same call: the two paths are required to
		// return identical values.
		dists.push_back (std::sqrt (pxA.exact_min_sqdist (K)));
	}

	radius_stats (dists, mean_r, max_r, median_r);
}

void RoiRadiusFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::ROI_RADIUS_MEAN][0] = mean_r;
	fvals[(int)Feature2D::ROI_RADIUS_MAX][0] = max_r;
	fvals[(int)Feature2D::ROI_RADIUS_MEDIAN][0] = median_r;
}

void RoiRadiusFeature::extract (LR& r, const Fsettings& s)
{
	RoiRadiusFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void RoiRadiusFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		extract (r, s);
	}
}
