#include "roi_radius.h"

RoiRadius::RoiRadius (const std::vector<Pixel2>& cloud, const std::vector<Pixel2>& contour)
{
	Moments2 mom2;
	std::vector<HistoItem> dists;
	for (auto& pxA : cloud)
	{
		auto [minSD, maxSD] = pxA.min_max_sqdist(contour);
		mom2.add(minSD);
		dists.push_back(minSD);
	}

	// Mean
	mean_r = mom2.mean();

	// Max
	max_r = mom2.max__();

	// Median
	TrivialHistogram H;
	H.initialize(mom2.min__(), mom2.max__(), dists);
	auto [median_, mode_, p01_, p10_, p25_, p75_, p90_, p99_, iqr_, rmad_, entropy_, uniformity_] = H.get_stats();
	median_r = median_;
}

double RoiRadius::get_mean_radius()
{
	return mean_r;
}

double RoiRadius::get_max_radius()
{
	return max_r;
}

double RoiRadius::get_median_radius()
{
	return median_r;
}

void RoiRadius::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Prepare the contour if necessary
		if (r.contour.contour_pixels.size() == 0)
			r.contour.calculate(r.aux_image_matrix);

		RoiRadius roir (r.raw_pixels, r.contour.contour_pixels);
		r.fvals[ROI_RADIUS_MEAN][0] = roir.get_mean_radius();
		r.fvals[ROI_RADIUS_MAX][0] = roir.get_max_radius();
		r.fvals[ROI_RADIUS_MEDIAN][0] = roir.get_median_radius();
	}
}
