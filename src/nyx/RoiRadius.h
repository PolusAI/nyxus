#pragma once

#include <vector>
#include "histogram.h"
#include "moments.h"
#include "pixel.h"

class RoiRadius
{
public:
	RoiRadius() {}

	void initialize (const std::vector<Pixel2>& cloud, const std::vector<Pixel2>& contour)
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
		H.initialize (mom2.min__(), mom2.max__(), dists);
		auto [median_, mode_, p01_, p10_, p25_, p75_, p90_, p99_, iqr_, rmad_, entropy_, uniformity_] = H.get_stats();
		median_r = median_;
	}

	std::tuple<double, double, double> get_min_max_median_radius() 
	{
		return {mean_r, max_r, median_r};
	}

protected:
	double max_r = 0, mean_r = 0, median_r = 0;
};