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
#include <array>
#include "../environment.h"
#include "histogram.h"
#include "moments.h"
#include "contour.h"

// Required by the reduction function
#include "../roi_data.h"

namespace Nyxus
{
	void parallelReduceIntensityStats(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
	{
		for (auto i = start; i < end; i++)
		{
			int lab = (*ptrLabels)[i];
			LR& lr = (*ptrLabelData)[lab];

			//==== Reduce pixel intensity #1, including MIN and MAX
			//XXX--not using online approach any more--		lr.reduce_pixel_intensity_features();
			// --MIN, MAX
			lr.fvals[MIN][0] = lr.aux_min;
			lr.fvals[MAX][0] = lr.aux_max;

			double n = lr.raw_pixels.size();

			// --AREA
			lr.fvals[AREA_PIXELS_COUNT][0] = n;
			if (theEnvironment.xyRes > 0.0)
				lr.fvals[AREA_UM2][0] = n * std::pow(theEnvironment.pixelSizeUm, 2);

			// --MEAN, ENERGY, CENTROID_XY
			double mean_ = 0.0;
			double energy = 0.0;
			double cen_x = 0.0,
				cen_y = 0.0,
				integInten = 0.0;
			for (auto& px : lr.raw_pixels)
			{
				mean_ += px.inten;
				energy += px.inten * px.inten;
				cen_x += px.x;
				cen_y += px.y;
				integInten += px.inten;
			}
			mean_ /= n;
			lr.fvals[MEAN][0] = mean_;
			lr.fvals[ENERGY][0] = energy;
			lr.fvals[ROOT_MEAN_SQUARED][0] = sqrt(lr.fvals[ENERGY][0] / n);
			lr.fvals[INTEGRATED_INTENSITY][0] = integInten;

			// --Centroid and Compactness
			lr.fvals[CENTROID_X][0] = cen_x;
			lr.fvals[CENTROID_Y][0] = cen_y;

			Moments2 mom2;
			for (auto& px : lr.raw_pixels)
			{
				double dst = std::sqrt(px.sqdist(cen_x, cen_y));
				mom2.add(dst);
			}
			double compa = mom2.std() / n;
			lr.fvals[COMPACTNESS][0] = compa;

			// --MAD, VARIANCE, STDDEV
			double mad = 0.0,
				var = 0.0;
			for (auto& px : lr.raw_pixels)
			{
				mad += std::abs(px.inten - mean_);
				var += (px.inten - mean_) * (px.inten - mean_);
			}
			lr.fvals[MEAN_ABSOLUTE_DEVIATION][0] = mad / n;
			var /= n;
			double stddev = sqrt(var);
			lr.fvals[STANDARD_DEVIATION][0] = stddev;

			// --Standard error
			lr.fvals[STANDARD_ERROR][0] = stddev / sqrt(n);

			//==== Do not calculate features of all-blank intensities (to avoid NANs)
			if (lr.intensitiesAllZero())
				continue;

			// P10, 25, 75, 90, IQR, RMAD, entropy, uniformity
#if 0	// Replaced with a faster version (class TrivialHistogram) 
			auto ptrH = lr.aux_Histogram;
			ptrH->build_histogram();
			auto [mean_, mode_, p10_, p25_, p75_, p90_, iqr_, rmad_, entropy_, uniformity_] = ptrH->get_stats();
			ptrH->reset();
#endif

			// Faster version
			TrivialHistogram H;
			H.initialize(lr.fvals[MIN][0], lr.fvals[MAX][0], lr.raw_pixels);
			auto [median_, mode_, p01_, p10_, p25_, p75_, p90_, p99_, iqr_, rmad_, entropy_, uniformity_] = H.get_stats();

			lr.fvals[MEDIAN][0] = median_;
			lr.fvals[P01][0] = p01_;
			lr.fvals[P10][0] = p10_;
			lr.fvals[P25][0] = p25_;
			lr.fvals[P75][0] = p75_;
			lr.fvals[P90][0] = p90_;
			lr.fvals[P99][0] = p99_;
			lr.fvals[INTERQUARTILE_RANGE][0] = iqr_;
			lr.fvals[ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = rmad_;
			lr.fvals[ENTROPY][0] = entropy_;
			lr.fvals[MODE][0] = mode_;
			lr.fvals[UNIFORMITY][0] = uniformity_;

			// --Uniformity calculated as PIU, percent image uniformity - see "A comparison of five standard methods for evaluating image intensity uniformity in partially parallel imaging MRI" [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/] and https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.2241606
			double piu = (1.0 - double(lr.aux_max - lr.aux_min) / double(lr.aux_max + lr.aux_min)) * 100.0;
			lr.fvals[UNIFORMITY_PIU][0] = piu;

			// Skewness
			//--Formula 1--	lr.fvals[SKEWNESS][0] = std::sqrt(n) * lr.aux_M3 / std::pow(lr.aux_M2, 1.5);
			//--Formula 2-- skewness = 3 * (mean - median) / stddev
			Moments4 mom;
			for (auto& px : lr.raw_pixels)
				mom.add(px.inten);
			lr.fvals[SKEWNESS][0] = mom.skewness();

			// Kurtosis
			//--Formula-- k1 = mean((x - mean(x)). ^ 4) / std(x). ^ 4
			lr.fvals[KURTOSIS][0] = mom.kurtosis();

			// Hyperskewness hs = E[x-mean].^5 / std(x).^5
			lr.fvals[HYPERSKEWNESS][0] = mom.hyperskewness();

			// Hyperflatness hf = E[x-mean].^6 / std(x).^6
			lr.fvals[HYPERFLATNESS][0] = mom.hyperflatness();

			//==== Basic morphology :: Bounding box
			// --
			lr.fvals[BBOX_XMIN][0] = lr.aabb.get_xmin();
			lr.fvals[BBOX_YMIN][0] = lr.aabb.get_ymin();
			lr.fvals[BBOX_WIDTH][0] = lr.aabb.get_width();
			lr.fvals[BBOX_HEIGHT][0] = lr.aabb.get_height();

			//==== Basic morphology :: Centroids
			lr.fvals[CENTROID_X][0] = lr.fvals[CENTROID_Y][0] = 0.0;
			for (auto& px : lr.raw_pixels)
			{
				lr.fvals[CENTROID_X][0] += px.x;
				lr.fvals[CENTROID_Y][0] += px.y;
			}
			lr.fvals[CENTROID_X][0] /= n;
			lr.fvals[CENTROID_Y][0] /= n;

			//==== Basic morphology :: Weighted centroids
			double x_mass = 0, y_mass = 0, mass = 0;

			for (auto& px : lr.raw_pixels)
			{
				x_mass = x_mass + (px.x + 1) * px.inten;    // the "+1" is only for compatability with matlab code (where index starts from 1) 
				y_mass = y_mass + (px.y + 1) * px.inten;    // the "+1" is only for compatability with matlab code (where index starts from 1) 
				mass += px.inten;
			}

			if (mass > 0)
			{
				lr.fvals[WEIGHTED_CENTROID_X][0] = x_mass / mass;
				lr.fvals[WEIGHTED_CENTROID_Y][0] = y_mass / mass;
			}
			else
			{
				lr.fvals[WEIGHTED_CENTROID_X][0] = 0.0;
				lr.fvals[WEIGHTED_CENTROID_Y][0] = 0.0;
			}

			// --Mass displacement (The distance between the centers of gravity in the gray-level representation of the object and the binary representation of the object.)
			double dx = lr.fvals[WEIGHTED_CENTROID_X][0] - lr.fvals[CENTROID_X][0],
				dy = lr.fvals[WEIGHTED_CENTROID_Y][0] - lr.fvals[CENTROID_Y][0],
				dist = std::sqrt(dx * dx + dy * dy);
			lr.fvals[MASS_DISPLACEMENT][0] = dist;

			//==== Basic morphology :: Extent
			lr.fvals[EXTENT][0] = n / lr.aabb.get_area();

			//==== Basic morphology :: Aspect ratio
			lr.fvals[ASPECT_RATIO][0] = lr.aabb.get_width() / lr.aabb.get_height();
		}
	}
}