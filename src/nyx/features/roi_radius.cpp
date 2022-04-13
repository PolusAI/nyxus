#include "roi_radius.h"

RoiRadiusFeature::RoiRadiusFeature() : FeatureMethod("RoiRadiusFeature")
{
	provide_features ({ROI_RADIUS_MEAN, ROI_RADIUS_MAX, ROI_RADIUS_MEDIAN});
	add_dependencies ({PERIMETER});
}

void RoiRadiusFeature::calculate (LR& r)
{
	const std::vector<Pixel2>& cloud = r.raw_pixels;
	const std::vector<Pixel2>& contour = r.contour;

	Moments2 mom2;
	std::vector<HistoItem> dists;
	for (auto& pxA : cloud)
	{
		auto minSD = pxA.min_sqdist(contour);
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

void RoiRadiusFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void RoiRadiusFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	const auto& cloud = r.osized_pixel_cloud; 
	const std::vector<Pixel2>& contour = r.contour;

	Moments2 mom2;
	std::vector<HistoItem> dists;
	for (size_t i=0; i<cloud.get_size(); i++) 
	{
		Pixel2 pxA = cloud.get_at(i);
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

void RoiRadiusFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[ROI_RADIUS_MEAN][0] = mean_r;
	fvals[ROI_RADIUS_MAX][0] = max_r;
	fvals[ROI_RADIUS_MEDIAN][0] = median_r; 
}

void RoiRadiusFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		RoiRadiusFeature rrf;
		rrf.calculate(r);
		rrf.save_value(r.fvals);
	}
}
