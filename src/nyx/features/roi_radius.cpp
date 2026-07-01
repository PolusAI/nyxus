#include <algorithm>
#include <cmath>
#include "roi_radius.h"

using namespace Nyxus;

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

	Moments2 mom2;
	std::vector<double> dists;
	for (auto& pxA : cloud)
	{
		// min_sqdist returns the SQUARED distance to the nearest contour pixel; the ROI radius
		// is a length, so take the square root (otherwise ROI_RADIUS_MAX is r^2 and can exceed
		// the image diagonal, e.g. 25 == 5^2).
		double minSD = std::sqrt((double)pxA.min_sqdist(K));
		mom2.add(minSD);
		dists.push_back(minSD);
	}

	// Mean
	mean_r = mom2.mean();

	// Max
	max_r = mom2.max__();

	// Median (interpolated, on the fractional radii - an integer histogram would truncate them)
	median_r = median_of(dists);
}

double RoiRadiusFeature::median_of (std::vector<double>& v)
{
	if (v.empty())
		return 0.0;
	std::sort (v.begin(), v.end());
	size_t m = v.size();
	return (m % 2) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
}

void RoiRadiusFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void RoiRadiusFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	const auto& cloud = r.raw_pixels_NT; 

	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	Moments2 mom2;
	std::vector<double> dists;
	for (size_t i=0; i<cloud.size(); i++)
	{
		Pixel2 pxA = cloud.get_at(i);
		auto [minSD, maxSD] = pxA.min_max_sqdist(K);
		double r_ = std::sqrt((double)minSD);   // radius is a length, not a squared distance
		mom2.add(r_);
		dists.push_back(r_);
	}

	// Mean
	mean_r = mom2.mean();

	// Max
	max_r = mom2.max__();

	// Median (interpolated, on the fractional radii)
	median_r = median_of(dists);
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
