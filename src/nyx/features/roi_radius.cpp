#include <algorithm>
#include <cmath>
#include "edt.h"
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

	// The largest raster the distance transform is allowed to allocate, in cells of 8 bytes: a 16 MB
	// ceiling, held per ROI being measured and so once per worker thread. A bounding box past it
	// takes the exhaustive scan, which needs no raster at all.
	constexpr size_t EDT_MAX_CELLS = 2 * 1024 * 1024;

	// Where the transform stops paying. It costs a pass over the bounding box, the scan costs a pass
	// over the contour per ROI pixel, and the two run at measured rates of ~1.3e5 cells/ms and
	// ~1.5e6 pixel-contour pairs/ms in an optimised build. That puts the crossover near
	// cells = pixels * contour / 12; 16 leaves the transform a margin on the shapes where it loses.
	constexpr size_t EDT_WORK_RATIO = 16;

}

// A compact ROI fills its bounding box and the transform wins by more the larger it gets, while a
// sparse one - a thin diagonal is the extreme, at one pixel per row and column - carries a box far
// larger than its pixel count and the scan wins. Fill ratio alone does not separate the two: a thin
// ring is nearly as sparse as a diagonal and the transform still wins on it, because its contour is
// long and that is what the scan pays for. Hence pixels * contour rather than pixels.
bool RoiRadiusFeature::edt_is_cheaper (size_t cells, size_t n_pixels, size_t n_contour)
{
	if (cells > EDT_MAX_CELLS)
		return false;

	// cells is under the ceiling by now, and a 64-bit size_t holds the pixel-contour product for
	// any ROI a slide can carry.
	return cells * EDT_WORK_RATIO <= n_pixels * n_contour;
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

	if (cloud.empty() || K.empty())
	{
		radius_stats (dists, mean_r, max_r, median_r);
		return;
	}

	// The raster a distance transform would need spans the ROI's pixels and its contour both. Both
	// extents are measured here rather than read from r.aabb: that member is accumulated while the
	// ROI is scanned, not derived from raw_pixels, so a caller that fills the cloud by another route
	// would leave it disagreeing with the pixels and the lookups below would run off the raster.
	// The contour is traced on a padded image and comes back offset from the pixel cloud, so its
	// extent is not the cloud's either, and taking the union is what keeps every site inside.
	StatsInt xmin = cloud[0].x, xmax = cloud[0].x, ymin = cloud[0].y, ymax = cloud[0].y;
	for (const auto& p : cloud)
	{
		xmin = std::min (xmin, p.x);
		xmax = std::max (xmax, p.x);
		ymin = std::min (ymin, p.y);
		ymax = std::max (ymax, p.y);
	}
	for (const auto& p : K)
	{
		xmin = std::min (xmin, p.x);
		xmax = std::max (xmax, p.x);
		ymin = std::min (ymin, p.y);
		ymax = std::max (ymax, p.y);
	}

	size_t W = (size_t)(xmax - xmin) + 1,
		H = (size_t)(ymax - ymin) + 1,
		cells = W * H;

	if (edt_is_cheaper (cells, cloud.size(), K.size()))
	{
		// Each pixel's distance to the contour, read off a distance transform seeded at the contour.
		// The transform is exact - a cell holds the same integer squared distance the scan below
		// returns - so which branch a ROI takes does not change the values it reports.
		std::vector<int64_t> sqdist;
		Nyxus::exact_sqedt (K, xmin, ymin, W, H, sqdist);

		for (auto& pxA : cloud)
			dists.push_back (std::sqrt ((double) sqdist [(size_t)(pxA.y - ymin) * W + (size_t)(pxA.x - xmin)]));
	}
	else
		for (auto& pxA : cloud)
			// exact_min_sqdist() rather than min_sqdist(): the latter is an approximate hill-descent
			// that assumes a locally-unimodal ordered contour and can settle in the wrong basin on a
			// closed one, returning more than the true minimum.
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

	// The same quantity calculate() takes, by an exhaustive scan over the contour per pixel. A
	// distance transform is the faster way to it, and this path does not take it: a ROI reaches here
	// by its bounding box raster in Pixel2 exceeding the RAM limit, so a transform's raster of that
	// same box, at two thirds the bytes, is on the order of the whole budget that sent it here.
	// Both are exact and both work in integer squared distances, so the choice does not move a value.
	for (size_t i=0; i<cloud.size(); i++)
	{
		Pixel2 pxA = cloud.get_at(i);
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
