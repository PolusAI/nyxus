#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <sstream>
#include "environment.h"
#include "sensemaker.h"
#include "f_erosion_pixels.h"
#include "f_radial_distribution.h"
#include "gabor.h"
#include "glrlm.h"
#include "glszm.h"
#include "gldm.h"
#include "ngtdm.h"
#include "hu.h"
#include "timing.h"
#include "moments.h"
#include "RoiRadius.h"


constexpr int N2R = 100 * 1000;
constexpr int N2R_2 = 100 * 1000;

// Preallocates the intensely accessed main containers
void init_feature_buffers()
{
	uniqueLabels.reserve(N2R);
	labelData.reserve(N2R);
	labelMutexes.reserve(N2R);
}

// Resets the main containers
void clearLabelStats()
{
	uniqueLabels.clear();
	labelData.clear();
	labelMutexes.clear();
}

// Label Record (structure 'LR') is where the state of label's pixels scanning and feature calculations is maintained. This function initializes an LR instance for the 1st pixel.
void init_label_record (LR& r, const std::string & segFile, const std::string & intFile, int x, int y, int label, PixIntens intensity)
{
	r.segFname = segFile;
	r.intFname = intFile;

	// Allocate the value matrix
	for (int i = 0; i < AvailableFeatures::_COUNT_; i++)
	{
		std::vector<StatsReal> row{0.0};	// One value initialy. More values can be added for Haralick and Zernike type methods
		r.fvals.push_back(row);
	}

	r.label = label;

	// Save the pixel
	r.raw_pixels.push_back(Pixel2(x, y, intensity));

	r.fvals[AREA_PIXELS_COUNT][0] = 1;
	r.aux_PrevCount = 0;
	// Min
	r.fvals[MIN][0] = r.aux_min = intensity; // r.min = intensity;
	// Max
	r.fvals[MAX][0] = r.aux_max = intensity; // r.max = intensity;
	// Moments
	r.fvals[MEAN][0] = intensity; // r.mean = intensity;
	r.aux_M2 = 0;
	r.aux_M3 = 0;
	r.aux_M4 = 0;
	r.fvals[ENERGY][0] = intensity * intensity; // r.massEnergy = intensity * intensity;
	r.aux_variance = 0.0;
	r.fvals[MEAN_ABSOLUTE_DEVIATION][0] = 0.0; // r.MAD = 0;
	r.aux_PrevIntens = intensity; // Previous intensity
	// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
	r.fvals[CENTROID_X][0] = StatsReal(x) + 1; // r.centroid_x = StatsReal(x) + 1;
	r.fvals[CENTROID_Y][0] = StatsReal(y) + 1; // r.centroid_y = StatsReal(y) + 1;
	
	#if 0 // Replaced with a faster version (class TrivialHistogram)
	// Histogram
	std::shared_ptr<Histo> ptrH = std::make_shared <Histo>();
	ptrH->add_observation(intensity);
	r.aux_Histogram = ptrH;
	#endif

	// Other fields
	r.fvals[MEDIAN][0] = 0; 
	r.fvals[STANDARD_DEVIATION][0] = 0; 
	r.fvals[SKEWNESS][0] = 0; 
	r.fvals[KURTOSIS][0] = 0; 
	r.fvals[ROOT_MEAN_SQUARED][0] = 0;
	r.fvals[P10][0] = r.fvals[P25][0] = r.fvals[P75][0] = r.fvals[90][0] = 0; 
	r.fvals[INTERQUARTILE_RANGE][0] = 0; 
	r.fvals[ENTROPY][0] = 0;
	r.fvals[MODE][0] = 0; 
	r.fvals[UNIFORMITY][0] = 0; 
	r.fvals[ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = 0; 
	// CellProfiler	
	r.fvals[CELLPROFILER_INTENSITY_INTEGRATEDINTENSITYEDGE][0] = 
	r.fvals[CELLPROFILER_INTENSITY_MAXINTENSITYEDGE][0] = 
	r.fvals[CELLPROFILER_INTENSITY_MEANINTENSITYEDGE][0] = 
	r.fvals[CELLPROFILER_INTENSITY_MININTENSITYEDGE][0] = 
	r.fvals[CELLPROFILER_INTENSITY_STDDEVINTENSITYEDGE][0] = 0; 
	
	r.init_aabb (x, y);

	#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
	// Dump intensities for testing
	if (label == SANITY_CHECK_INTENSITIES_FOR_LABEL)	// Put your label code of interest
		r.raw_intensities.push_back(intensity);
	#endif
}

// This function 'digests' the 2nd and the following pixel of a label and updates the label's feature calculation state - the instance of structure 'LR'
void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity)
{
	// Save the pixel
	lr.raw_pixels.push_back(Pixel2(x, y, intensity));

	// Update ROI intensity range
	lr.aux_min = std::min (lr.aux_min, intensity);
	lr.aux_max = std::max (lr.aux_max, intensity);

	// Update ROI bounds
	lr.update_aabb(x, y);
}

// The root function of handling a pixel being scanned
void update_label (int x, int y, int label, PixIntens intensity)
{
	auto it = uniqueLabels.find(label);
	if (it == uniqueLabels.end())
	{
		// Remember this label
		uniqueLabels.insert(label);

		// Initialize the label record
		LR lr;
		init_label_record (lr, theSegFname, theIntFname, x, y, label, intensity);
		labelData[label] = lr;
	}
	else
	{
		#ifdef SIMULATE_WORKLOAD_FACTOR
		// Simulate a chunk of processing. 1K iterations cost ~300 mks
		for (long tmp = 0; tmp < SIMULATE_WORKLOAD_FACTOR * 1000; tmp++)
			auto start = std::chrono::system_clock::now();
		#endif

		// Update label's stats
		LR& lr = labelData[label];
		update_label_record (lr, x, y, label, intensity);
	}
}

void parallelReduceIntensityStats (size_t start, size_t end, std::vector<int> * ptrLabels, std::unordered_map <int,LR> * ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels) [i];
		LR& lr = (*ptrLabelData) [lab];

		#if 0	// Condition is too strict
		if (lr.raw_pixels.size() < smallestROI)
		{
			lr.roi_disabled = true;
			continue;
		}
		#endif

		//==== Reduce pixel intensity #1, including MIN and MAX
		//XXX--not using online approach any more--		lr.reduce_pixel_intensity_features();
		// --MIN, MAX
		lr.fvals[MIN][0] = lr.aux_min;
		lr.fvals[MAX][0] = lr.aux_max;

		double n = lr.raw_pixels.size();

		// --AREA
		lr.fvals[AREA_PIXELS_COUNT][0] = n;

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
		lr.fvals[ROOT_MEAN_SQUARED][0] = sqrt (lr.fvals[ENERGY][0] / n);
		lr.fvals[CENTROID_X][0] = cen_x;
		lr.fvals[CENTROID_Y][0] = cen_y;
		lr.fvals[INTEGRATED_INTENSITY][0] = integInten;

		// --Compactness
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
#ifdef TESTING
		std::cout << "\n---Test data---\nraw_pixels_label_" << lab << " = [";
		for (int ix=0; ix<lr.raw_pixels.size(); ix++)
		{
			if (ix > 0)
				std::cout << ", ";
			std::cout << lr.raw_pixels[ix].inten;
		}
		std::cout << "]\n\n";
#endif
		TrivialHistogram H;
		H.initialize (lr.fvals[MIN][0], lr.fvals[MAX][0], lr.raw_pixels);
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
		lr.fvals[HYPERSKEWNESS][0] = mom.hyperskewness ();

		// Hyperflatness hf = E[x-mean].^6 / std(x).^6
		lr.fvals[HYPERFLATNESS][0] = mom.hyperflatness ();

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
			x_mass = x_mass + (px.x + 1) * px.inten;    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
			y_mass = y_mass + (px.y + 1) * px.inten;    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
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
			dist = std::sqrt (dx*dx + dy*dy);
		lr.fvals[MASS_DISPLACEMENT][0] = dist;

		//==== Basic morphology :: Extent
		lr.fvals[EXTENT][0] = n / lr.aabb.get_area();

		//==== Basic morphology :: Aspect ratio
		lr.fvals[ASPECT_RATIO][0] = lr.aabb.get_width() / lr.aabb.get_height();	
	}
}

void parallelReduceContour (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.roi_disabled)
			continue;

		//==== Contour, ROI perimeter, equivalent circle diameter
		ImageMatrix im (r.raw_pixels, r.aabb);
		r.contour.calculate (im);
		r.fvals[PERIMETER][0] = r.contour.get_roi_perimeter();	
		r.fvals[EQUIVALENT_DIAMETER][0] = r.contour.get_diameter_equal_perimeter();	
		auto [cmin, cmax, cmean, cstddev] = r.contour.get_min_max_mean_stddev_intensity();
		r.fvals[EDGE_MEAN_INTENSITY][0] = cmean;
		r.fvals[EDGE_STDDEV_INTENSITY][0] = cstddev;
		r.fvals[EDGE_MAX_INTENSITY][0] = cmax;
		r.fvals[EDGE_MIN_INTENSITY][0] = cmin;
	}
}

void parallelReduceConvHull (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{	
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		//==== Convex hull and solidity
		r.convHull.calculate(r.raw_pixels);
		r.fvals[CONVEX_HULL_AREA][0] = r.convHull.getArea();	// .convHullArea
		r.fvals[SOLIDITY][0] = r.raw_pixels.size() / r.fvals[CONVEX_HULL_AREA][0];	// .solidity

		//==== Circularity
		r.fvals[CIRCULARITY][0] = 4.0 * M_PI * r.raw_pixels.size() / (r.fvals[PERIMETER][0] * r.fvals[PERIMETER][0]); // r.circularity = 4.0 * M_PI * r.pixelCountRoiArea / (r.roiPerimeter * r.roiPerimeter);

		//==== IntegratedIntensityEdge, MaxIntensityEdge, MinIntensityEdge, etc
		r.reduce_edge_intensity_features();
	}
}

void parallelReduceErosionPixels (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR & r = (*ptrLabelData)[lab];

		// Skip calculation in case of bad data
		if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
			continue;

		// Calculate feature
		ImageMatrix im(r.raw_pixels, r.aabb);
		ErosionPixels epix;
		r.fvals[EROSION_PIXELS][0] = epix.calc_feature(im);
	}
}

void parallelReduceFeret (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		ParticleMetrics pm(r.convHull.CH);
		std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
		pm.calc_ferret(
			r.fvals[MAX_FERET_DIAMETER][0],	// .maxFeretDiameter
			r.fvals[MAX_FERET_ANGLE][0], // .maxFeretAngle
			r.fvals[MIN_FERET_DIAMETER][0], // .minFeretDiameter
			r.fvals[MIN_FERET_ANGLE][0], // .minFeretAngle
			allD
		);

		auto structStat = ComputeCommonStatistics2(allD);
		r.fvals[STAT_FERET_DIAM_MIN][0] // .feretStats_minD 
			= (double)structStat.min;	// ratios[59]
		r.fvals[STAT_FERET_DIAM_MAX][0] // .feretStats_maxD 
			= (double)structStat.max;	// ratios[60]
		r.fvals[STAT_FERET_DIAM_MEAN][0] // .feretStats_meanD 
			= structStat.mean;	// ratios[61]
		r.fvals[STAT_FERET_DIAM_MEDIAN][0] // .feretStats_medianD 
			= structStat.median;	// ratios[62]
		r.fvals[STAT_FERET_DIAM_STDDEV][0] // .feretStats_stddevD 
			= structStat.stdev;	// ratios[63]
		r.fvals[STAT_FERET_DIAM_MODE][0] // .feretStats_modeD 
			= (double)structStat.mode;	// ratios[64]		
	}
}

void parallelReduceMartin (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR & r = (*ptrLabelData)[lab];

		ParticleMetrics pm(r.convHull.CH);
		std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
		pm.calc_martin(allD);
		auto structStat = ComputeCommonStatistics2(allD);
		r.fvals[STAT_MARTIN_DIAM_MIN][0] = (double)structStat.min;	// .martinStats_minD, martinStats_maxD, martinStats_meanD, martinStats_medianD, martinStats_stddevD, martinStats_modeD
		r.fvals[STAT_MARTIN_DIAM_MAX][0] = (double)structStat.max;	
		r.fvals[STAT_MARTIN_DIAM_MEAN][0] = structStat.mean;	
		r.fvals[STAT_MARTIN_DIAM_MEDIAN][0] = structStat.median;	
		r.fvals[STAT_MARTIN_DIAM_STDDEV][0]	= structStat.stdev;	
		r.fvals[STAT_MARTIN_DIAM_MODE][0] = (double)structStat.mode;	
	}
}

void parallelReduceNassenstein (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR & r = (*ptrLabelData)[lab];

		ParticleMetrics pm(r.convHull.CH);
		std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
		pm.calc_nassenstein(allD);
		auto s = ComputeCommonStatistics2(allD);
		r.fvals[STAT_NASSENSTEIN_DIAM_MIN][0] = (double)s.min;	// nassStats_minD, nassStats_maxD, nassStats_meanD, nassStats_medianD, nassStats_stddevD, nassStats_modeD
		r.fvals[STAT_NASSENSTEIN_DIAM_MAX][0] = (double)s.max;
		r.fvals[STAT_NASSENSTEIN_DIAM_MEAN][0] = s.mean;
		r.fvals[STAT_NASSENSTEIN_DIAM_MEDIAN][0] = s.median;
		r.fvals[STAT_NASSENSTEIN_DIAM_STDDEV][0] = s.stdev;
		r.fvals[STAT_NASSENSTEIN_DIAM_MODE][0] = (double)s.mode;
	}
}

void parallelReduceHaralick2D (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR & r = (*ptrLabelData)[lab];

		std::vector<double> texture_Feature_Angles;
		haralick2D(
			// in
			r.raw_pixels,	// nonzero_intensity_pixels,
			r.aabb,			// AABB info not to calculate it again from 'raw_pixels' in the function
			0.0,			// distance,
			// out
			texture_Feature_Angles,
			r.fvals[TEXTURE_ANGULAR2NDMOMENT], // .texture_AngularSecondMoments,
			r.fvals[TEXTURE_CONTRAST], // .texture_Contrast,
			r.fvals[TEXTURE_CORRELATION], // .texture_Correlation,
			r.fvals[TEXTURE_VARIANCE], // .texture_Variance,
			r.fvals[TEXTURE_INVERSEDIFFERENCEMOMENT], // .texture_InverseDifferenceMoment,
			r.fvals[TEXTURE_SUMAVERAGE], // .texture_SumAverage,
			r.fvals[TEXTURE_SUMVARIANCE], // .texture_SumVariance,
			r.fvals[TEXTURE_SUMENTROPY], // .texture_SumEntropy,
			r.fvals[TEXTURE_ENTROPY], // .texture_Entropy,
			r.fvals[TEXTURE_DIFFERENCEVARIANCE], // .texture_DifferenceVariance,
			r.fvals[TEXTURE_DIFFERENCEENTROPY], // .texture_DifferenceEntropy,
			r.fvals[TEXTURE_INFOMEAS1], // .texture_InfoMeas1,
			r.fvals[TEXTURE_INFOMEAS2]); // .texture_InfoMeas2);

		// Fix calculated feature values due to all-0 intensity labels to avoid NANs in the output
		if (r.intensitiesAllZero())
		{
			for (int i = 0; i < texture_Feature_Angles.size(); i++)
			{
				r.fvals[TEXTURE_ANGULAR2NDMOMENT][i] =
					r.fvals[TEXTURE_CONTRAST][i] =
					r.fvals[TEXTURE_CORRELATION][i] =
					r.fvals[TEXTURE_VARIANCE][i] =
					r.fvals[TEXTURE_INVERSEDIFFERENCEMOMENT][i] =
					r.fvals[TEXTURE_SUMAVERAGE][i] =
					r.fvals[TEXTURE_SUMVARIANCE][i] =
					r.fvals[TEXTURE_SUMENTROPY][i] =
					r.fvals[TEXTURE_ENTROPY][i] =
					r.fvals[TEXTURE_DIFFERENCEVARIANCE][i] =
					r.fvals[TEXTURE_DIFFERENCEENTROPY][i] =
					r.fvals[TEXTURE_INFOMEAS1][i] =
					r.fvals[TEXTURE_INFOMEAS2][i] = 0.0;
			}
		}
	}
}

void parallelReduceZernike2D (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR & r = (*ptrLabelData)[lab];

		zernike2D(
			// in
			r.raw_pixels,	// nonzero_intensity_pixels,
			r.aabb,			// AABB info not to calculate it again from 'raw_pixels' in the function
			r.aux_ZERNIKE2D_ORDER,
			// out
			r.fvals[ZERNIKE2D]);	// .Zernike2D

		// Fix calculated feature values due to all-0 intensity labels to avoid NANs in the output
		if (r.intensitiesAllZero())
		{
			for (int i = 0; i < r.fvals[ZERNIKE2D].size(); i++)
				r.fvals[ZERNIKE2D][i] = 0.0;
		}
	}
}

void parallelGabor (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR & r = (*ptrLabelData)[lab];

		GaborFeatures gf;

		// Skip calculation in case of bad data
		if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
			continue;
		
		// Calculate Gabor
		ImageMatrix im(r.raw_pixels, r.aabb);
		gf.calc_GaborTextureFilters2D(im, r.fvals[GABOR]);	// r.fvals[GABOR] will contain GaborFeatures::num_features items upon return

	}
}

typedef void (*functype) (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

void runParallel (functype f, int nThr, size_t workPerThread, size_t datasetSize, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	std::vector<std::future<void>> T;
	for (int t = 0; t < nThr; t++)
	{
		size_t idxS = t * workPerThread,
			idxE = idxS + workPerThread;
		if (t == nThr - 1)
			idxE = datasetSize; // include the tail
		// Example:	T.push_back(std::async(std::launch::async, parallelReduceIntensityStats, idxS, idxE, &sortedUniqueLabels, &labelData));
		T.push_back (std::async(std::launch::async, f, idxS, idxE, &sortedUniqueLabels, &labelData));
	}
}


// This function should be called once after a file pair processing is finished.
void reduce (int nThr, int min_online_roi_size)
{
	// Build ROI size histogram
	if (theEnvironment.verbosity_level & VERBOSITY_ROI_INFO)
	{
		OnlineHistogram hist;
		for (auto& ld : labelData) // for (auto& lv : labelUniqueIntensityValues)
		{
			auto l = ld.first;		// Label code
			auto& lr = ld.second;	// Label record

			hist.add_observation((HistoItem)lr.raw_pixels.size());
		}
		hist.build_histogram();
		hist.print(true, "\nHistogram of ROI size:");
	}

	//=== Sort the labels 
	// We do this for 2 purposes: (1) being able to iterate them in equal chunks that in turn requires [indexed] access; (2) later, output the results by sorted labels
	// Implementing the following --> std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
	sortedUniqueLabels.clear();
	for (auto l : uniqueLabels)
		sortedUniqueLabels.push_back(l);
	//std::sort(sortedUniqueLabels.begin(), sortedUniqueLabels.end());
	
	//==== 	Parallel execution parameters 
	size_t tileSize = sortedUniqueLabels.size(),
		workPerThread = tileSize / nThr;

	//==== Pixel intensity stats 
	#if 0
	if (theFeatureSet.anyEnabled({ 
		// directly
		MEAN,
		MEDIAN,
		MIN,
		MAX,
		RANGE,
		STANDARD_DEVIATION,
		SKEWNESS,
		KURTOSIS,
		MEAN_ABSOLUTE_DEVIATION,
		ENERGY,
		ROOT_MEAN_SQUARED,
		ENTROPY,
		MODE,
		UNIFORMITY,
		P10, P25, P75, P90,
		INTERQUARTILE_RANGE,
		ROBUST_MEAN_ABSOLUTE_DEVIATION,
		WEIGHTED_CENTROID_Y,
		WEIGHTED_CENTROID_X,
		// indirectly
		NUM_NEIGHBORS, // aabb
		MAJOR_AXIS_LENGTH, MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION, // centroid_x|y, pixelCountArea
		}))
	#endif
	{
		STOPWATCH("Intensity stats ...", "\tReduced intensity stats");
		runParallel (parallelReduceIntensityStats, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Neighbors
	if (theFeatureSet.anyEnabled({ NUM_NEIGHBORS, CLOSEST_NEIGHBOR1_DIST, CLOSEST_NEIGHBOR2_DIST }))
	{
		STOPWATCH("Neighbors ...", "\tReduced neighbors");
		reduce_neighbors (5 /* collision radius [pixels] */);
	}

	//==== Fitting an ellipse
	if (theFeatureSet.anyEnabled({ MAJOR_AXIS_LENGTH, MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION, ROUNDNESS } ))
	{
		STOPWATCH("Ellipticity et al ...", "\tReduced ellipticity - MAJOR_AXIS_LENGTH, MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION");
		// 
		//Reference: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/19028/versions/1/previews/regiondata.m/index.html
		//		

		// Calculate normalized second central moments for the region.
		// 1/12 is the normalized second central moment of a pixel with unit length.
		for (auto& ld : labelData) // for (auto& lv : labelUniqueIntensityValues)
		{
			auto& r = ld.second;

			if (r.roi_disabled)
				continue;

			double XSquaredTmp = 0, YSquaredTmp = 0, XYSquaredTmp = 0;
			for (auto& pix : r.raw_pixels)
			{
				auto diffX = r.fvals[CENTROID_X][0] - pix.x,
					diffY = r.fvals[CENTROID_Y][0] - pix.y;
				XSquaredTmp += diffX * diffX; //(double(x) - (xCentroid - Im.ROIWidthBeg)) * (double(x) - (xCentroid - Im.ROIWidthBeg));
				YSquaredTmp += diffY * diffY; //(-double(y) + (yCentroid - Im.ROIHeightBeg)) * (-double(y) + (yCentroid - Im.ROIHeightBeg));
				XYSquaredTmp += diffX * diffY; //(double(x) - (xCentroid - Im.ROIWidthBeg)) * (-double(y) + (yCentroid - Im.ROIHeightBeg));
			}

			double uxx = XSquaredTmp / r.raw_pixels.size() + 1.0 / 12.0;
			double uyy = YSquaredTmp / r.raw_pixels.size() + 1.0 / 12.0;
			double uxy = XYSquaredTmp / r.raw_pixels.size();

			// Calculate major axis length, minor axis length, and eccentricity.
			double common = sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
			double MajorAxisLength = 2 * sqrt(2) * sqrt(uxx + uyy + common);
			double MinorAxisLength = 2 * sqrt(2) * sqrt(uxx + uyy - common);
			double Eccentricity = 2 * sqrt((MajorAxisLength / 2) * (MajorAxisLength / 2) - (MinorAxisLength / 2) * (MinorAxisLength / 2)) / MajorAxisLength;

			// Calculate orientation [-90,90]
			double num, den, Orientation;
			if (uyy > uxx) {
				num = uyy - uxx + sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
				den = 2 * uxy;
			}
			else {
				num = 2 * uxy;
				den = uxx - uyy + sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
			}
			if (num == 0 && den == 0)
				Orientation = 0;
			else
				Orientation = (180.0 / M_PI) * atan(num / den);

			r.fvals[MAJOR_AXIS_LENGTH][0] = MajorAxisLength;
			r.fvals[MINOR_AXIS_LENGTH][0] = MinorAxisLength;
			r.fvals[ECCENTRICITY][0] = Eccentricity;
			r.fvals[ORIENTATION][0] = Orientation;
			r.fvals[ROUNDNESS][0] = (4 * r.fvals[AREA_PIXELS_COUNT][0]) / (M_PI * MajorAxisLength);
		}
	}

	//==== Contour-related ROI perimeter, equivalent circle diameter
	if (theFeatureSet.anyEnabled({ EQUIVALENT_DIAMETER, PERIMETER,
			CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY, // depend on PERIMETER
			CELLPROFILER_INTENSITY_INTEGRATEDINTENSITYEDGE,
			CELLPROFILER_INTENSITY_MAXINTENSITYEDGE,
			CELLPROFILER_INTENSITY_MININTENSITYEDGE,
			CELLPROFILER_INTENSITY_MEANINTENSITYEDGE,
			CELLPROFILER_INTENSITY_STDDEVINTENSITYEDGE // depend on contour and PERIMETER
		}))
	{
		{
			STOPWATCH("Contour, ROI perimeter, equivalent circle diameter ...", "\tReduced contour, ROI perimeter, equivalent circle diameter");
			runParallel (parallelReduceContour, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
		}
		//==== Convex hull related Solidity, Circularity, IntegratedIntensityEdge, MaxIntensityEdge, MinIntensityEdge, etc
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
			STOPWATCH("Hulls, and related (circularity, solidity, etc) ...", "\tReduced hulls, and related (circularity, solidity, etc)");
			runParallel (parallelReduceConvHull, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
		}	
	}

	//==== Extrema and Euler number
	if (theFeatureSet.anyEnabled({ 
		EXTREMA_P1_Y,
		EXTREMA_P1_X,
		EXTREMA_P2_Y,
		EXTREMA_P2_X,
		EXTREMA_P3_Y,
		EXTREMA_P3_X,
		EXTREMA_P4_Y,
		EXTREMA_P4_X,
		EXTREMA_P5_Y,
		EXTREMA_P5_X,
		EXTREMA_P6_Y,
		EXTREMA_P6_X,
		EXTREMA_P7_Y,
		EXTREMA_P7_X,
		EXTREMA_P8_Y,
		EXTREMA_P8_X, 
		EULER_NUMBER }))
		
	{
		STOPWATCH("Extrema and Euler ...", "\tReduced extrema and Euler");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;

			if (r.roi_disabled)
				continue;

			//==== Extrema
			int TopMostIndex = -1;
			int LowestIndex = -1;
			int LeftMostIndex = -1;
			int RightMostIndex = -1;

			for (auto& pix : r.raw_pixels)
			{
				if (TopMostIndex == -1 || pix.y < (StatsInt)TopMostIndex)
					TopMostIndex = pix.y;
				if (LowestIndex == -1 || pix.y > (StatsInt)LowestIndex)
					LowestIndex = pix.y;

				if (LeftMostIndex == -1 || pix.x < (StatsInt)LeftMostIndex)
					LeftMostIndex = pix.x;
				if (RightMostIndex == -1 || pix.x > (StatsInt)RightMostIndex)
					RightMostIndex = pix.x;
			}

			int TopMost_MostLeftIndex = -1;
			int TopMost_MostRightIndex = -1;
			int Lowest_MostLeftIndex = -1;
			int Lowest_MostRightIndex = -1;
			int LeftMost_Top = -1;
			int LeftMost_Bottom = -1;
			int RightMost_Top = -1;
			int RightMost_Bottom = -1;

			for (auto& pix : r.raw_pixels)
			{
				// Find leftmost and rightmost x-pixels of the top 
				if (pix.y == TopMostIndex && (TopMost_MostLeftIndex == -1 || pix.x < (StatsInt)TopMost_MostLeftIndex))
					TopMost_MostLeftIndex = pix.x;
				if (pix.y == TopMostIndex && (TopMost_MostRightIndex == -1 || pix.x > (StatsInt)TopMost_MostRightIndex))
					TopMost_MostRightIndex = pix.x;

				// Find leftmost and rightmost x-pixels of the bottom
				if (pix.y == LowestIndex && (Lowest_MostLeftIndex == -1 || pix.x < (StatsInt)Lowest_MostLeftIndex))
					Lowest_MostLeftIndex = pix.x;
				if (pix.y == LowestIndex && (Lowest_MostRightIndex == -1 || pix.x > (StatsInt)Lowest_MostRightIndex))
					Lowest_MostRightIndex = pix.x;

				// Find top and bottom y-pixels of the leftmost
				if (pix.x == LeftMostIndex && (LeftMost_Top == -1 || pix.y < (StatsInt)LeftMost_Top))
					LeftMost_Top = pix.y;
				if (pix.x == LeftMostIndex && (LeftMost_Bottom == -1 || pix.y > (StatsInt)LeftMost_Bottom))
					LeftMost_Bottom = pix.y;

				// Find top and bottom y-pixels of the rightmost
				if (pix.x == RightMostIndex && (RightMost_Top == -1 || pix.y < (StatsInt)RightMost_Top))
					RightMost_Top = pix.y;
				if (pix.x == RightMostIndex && (RightMost_Bottom == -1 || pix.y > (StatsInt)RightMost_Bottom))
					RightMost_Bottom = pix.y;
			}

			r.fvals[EXTREMA_P1_Y][0] = TopMostIndex; // -0.5 + Im.ROIHeightBeg;	// .extremaP1y
			r.fvals[EXTREMA_P1_X][0] = TopMost_MostLeftIndex; // -0.5 + Im.ROIWidthBeg;	// .extremaP1x

			r.fvals[EXTREMA_P2_Y][0] = TopMostIndex; // -0.5 + Im.ROIHeightBeg;	// .extremaP2y
			r.fvals[EXTREMA_P2_X][0] = TopMost_MostRightIndex; // +0.5 + Im.ROIWidthBeg;	// .extremaP2x

			r.fvals[EXTREMA_P3_Y][0] = RightMost_Top; // -0.5 + Im.ROIHeightBeg;	// .extremaP3y
			r.fvals[EXTREMA_P3_X][0] = RightMostIndex; // +0.5 + Im.ROIWidthBeg;	// .extremaP3x

			r.fvals[EXTREMA_P4_Y][0] = RightMost_Bottom; // +0.5 + Im.ROIHeightBeg;	// .extremaP4y
			r.fvals[EXTREMA_P4_X][0] = RightMostIndex; // +0.5 + Im.ROIWidthBeg;	// .extremaP4x

			r.fvals[EXTREMA_P5_Y][0] = LowestIndex; // +0.5 + Im.ROIHeightBeg;	//.extremaP5y
			r.fvals[EXTREMA_P5_X][0] = Lowest_MostRightIndex; // +0.5 + Im.ROIWidthBeg;	//.extremaP5x

			r.fvals[EXTREMA_P6_Y][0] = LowestIndex; // +0.5 + Im.ROIHeightBeg;	//.extremaP6y
			r.fvals[EXTREMA_P6_X][0] = Lowest_MostLeftIndex; // -0.5 + Im.ROIWidthBeg;	//.extremaP6x

			r.fvals[EXTREMA_P7_Y][0] = LeftMost_Bottom; // +0.5 + Im.ROIHeightBeg;	//.extremaP7y
			r.fvals[EXTREMA_P7_X][0] = LeftMostIndex; // -0.5 + Im.ROIWidthBeg;	//.extremaP7x

			r.fvals[EXTREMA_P8_Y][0] = LeftMost_Top; // -0.5 + Im.ROIHeightBeg;	//.extremaP8y
			r.fvals[EXTREMA_P8_X][0] = LeftMostIndex; // -0.5 + Im.ROIWidthBeg;	//.extremaP8x

			//==== Euler number
			EulerNumber eu(r.raw_pixels, r.aabb.get_xmin(), r.aabb.get_ymin(), r.aabb.get_xmax(), r.aabb.get_ymax(), 8);	// Using mode=8 following to WNDCHRM example
			r.fvals[EULER_NUMBER][0] = eu.euler_number;	
		}
	}
	//==== Feret diameters and angles
	if (theFeatureSet.anyEnabled ({MIN_FERET_DIAMETER, MAX_FERET_DIAMETER, MIN_FERET_ANGLE, MAX_FERET_ANGLE}) ||
		theFeatureSet.anyEnabled({ 
			STAT_FERET_DIAM_MIN,
			STAT_FERET_DIAM_MAX,
			STAT_FERET_DIAM_MEAN,
			STAT_FERET_DIAM_MEDIAN,
			STAT_FERET_DIAM_STDDEV,
			STAT_FERET_DIAM_MODE }))
	{
		STOPWATCH("Feret ...", "\tReduced Feret");
		runParallel(parallelReduceFeret, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Martin diameters
	if (theFeatureSet.anyEnabled({ STAT_MARTIN_DIAM_MIN,
		STAT_MARTIN_DIAM_MAX,
		STAT_MARTIN_DIAM_MEAN,
		STAT_MARTIN_DIAM_MEDIAN,
		STAT_MARTIN_DIAM_STDDEV,
		STAT_MARTIN_DIAM_MODE }))
	{
		STOPWATCH("Matrin ...", "\tReduced Martin");
		runParallel(parallelReduceMartin, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Nassenstein diameters
	if (theFeatureSet.anyEnabled({ STAT_NASSENSTEIN_DIAM_MIN,
		STAT_NASSENSTEIN_DIAM_MAX,
		STAT_NASSENSTEIN_DIAM_MEAN,
		STAT_NASSENSTEIN_DIAM_MEDIAN,
		STAT_NASSENSTEIN_DIAM_STDDEV,
		STAT_NASSENSTEIN_DIAM_MODE }))
	{
		STOPWATCH("Nassenstein ...", "\tReduced Nassenstein");
		runParallel(parallelReduceNassenstein, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	{
		STOPWATCH("Hexagonality, polygonality, enclosing circle, geodetic length & thickness ...", "\tReduced hexagonality, polygonality, enclosing circle, geodetic length & thickness");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;

			// Skip if the contour, convex hull, and neighbors are unavailable, otherwise the related features will be == NAN. Those feature will be equal to the default unassigned value.
			if (r.contour.contour_pixels.size() == 0 || r.convHull.CH.size() == 0 || r.fvals[NUM_NEIGHBORS][0] == 0)
				continue;

			//==== Hexagonality and polygonality
			Hexagonality_and_Polygonality hp;
			auto [polyAve, hexAve, hexSd] = hp.calculate(r.fvals[NUM_NEIGHBORS][0], r.raw_pixels.size(), r.fvals[PERIMETER][0], r.fvals[CONVEX_HULL_AREA][0], r.fvals[MIN_FERET_DIAMETER][0], r.fvals[MAX_FERET_DIAMETER][0]);
			r.fvals[POLYGONALITY_AVE][0] = polyAve;
			r.fvals[HEXAGONALITY_AVE][0] = hexAve;
			r.fvals[HEXAGONALITY_STDDEV][0] = hexSd;

			//==== Enclosing circle
			MinEnclosingCircle cir1;
			r.fvals[DIAMETER_MIN_ENCLOSING_CIRCLE][0] = cir1.calculate_diam(r.contour.contour_pixels);
			InscribingCircumscribingCircle cir2;
			auto [diamIns, diamCir] = cir2.calculateInsCir(r.contour.contour_pixels, r.fvals[CENTROID_X][0], r.fvals[CENTROID_Y][0]);
			r.fvals[DIAMETER_INSCRIBING_CIRCLE][0] = diamIns;
			r.fvals[DIAMETER_CIRCUMSCRIBING_CIRCLE][0] = diamCir;

			//==== Geodetic length thickness
			GeodeticLength_and_Thickness glt;
			auto [geoLen, thick] = glt.calculate(r.raw_pixels.size(), (StatsInt)r.fvals[PERIMETER][0]);
			r.fvals[GEODETIC_LENGTH][0] = geoLen;
			r.fvals[THICKNESS][0] = thick;
		}
	}

	//==== ROI radius
	if (theFeatureSet.anyEnabled({
		ROI_RADIUS_MEAN,
		ROI_RADIUS_MAX,
		ROI_RADIUS_MEDIAN
		}))
	{
		STOPWATCH("ROI radius min/max/median ...", "\tROI radius");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;
			// Prepare the contour if necessary
			if (r.contour.contour_pixels.size() == 0)
			{
				ImageMatrix im(r.raw_pixels, r.aabb);
				r.contour.calculate(im);
			}

			RoiRadius roir;
			roir.initialize (r.raw_pixels, r.contour.contour_pixels);
			auto [mean_r, max_r, median_r] = roir.get_min_max_median_radius();

			r.fvals[ROI_RADIUS_MEAN][0] = mean_r;
			r.fvals[ROI_RADIUS_MAX][0] = max_r;
			r.fvals[ROI_RADIUS_MEDIAN][0] = median_r;
		}
	}

	//==== Erosion pixels
	if (theFeatureSet.isEnabled(EROSION_PIXELS))
	{
		STOPWATCH("EROSION_PIXELS ...", "\tReduced EROSION_PIXELS");
		runParallel (parallelReduceErosionPixels, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}	

	//==== GLCM aka Haralick 2D 
	if (theFeatureSet.anyEnabled({ 
		TEXTURE_ANGULAR2NDMOMENT,
		TEXTURE_CONTRAST,
		TEXTURE_CORRELATION,
		TEXTURE_VARIANCE,
		TEXTURE_INVERSEDIFFERENCEMOMENT,
		TEXTURE_SUMAVERAGE,
		TEXTURE_SUMVARIANCE,
		TEXTURE_SUMENTROPY,
		TEXTURE_ENTROPY,
		TEXTURE_DIFFERENCEVARIANCE,
		TEXTURE_DIFFERENCEENTROPY,
		TEXTURE_INFOMEAS1,
		TEXTURE_INFOMEAS2 }))
	{
		STOPWATCH("Haralick2D ...", "\tReduced Haralick2D");
		runParallel (parallelReduceHaralick2D, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== GLRLM
	if (theFeatureSet.anyEnabled({
		GLRLM_SRE,
		GLRLM_LRE,
		GLRLM_GLN,
		GLRLM_GLNN,
		GLRLM_RLN,
		GLRLM_RLNN,
		GLRLM_RP,
		GLRLM_GLV,
		GLRLM_RV,
		GLRLM_RE,
		GLRLM_LGLRE,
		GLRLM_HGLRE,
		GLRLM_SRLGLE,
		GLRLM_SRHGLE,
		GLRLM_LRLGLE,
		GLRLM_LRHGLE
		}))
	{
		STOPWATCH("GLRLM ...", "\tReduced GLRLM");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;
			ImageMatrix im (r.raw_pixels, r.aabb);
			GLRLM_features glrlm;
			glrlm.initialize ((int) r.fvals[MIN][0], (int) r.fvals[MAX][0], im);
			glrlm.calc_SRE (r.fvals [GLRLM_SRE]);
			glrlm.calc_LRE (r.fvals [GLRLM_LRE]);
			glrlm.calc_GLN (r.fvals [GLRLM_GLN]);
			glrlm.calc_GLNN (r.fvals [GLRLM_GLNN]);
			glrlm.calc_RLN (r.fvals [GLRLM_RLN]);
			glrlm.calc_RLNN (r.fvals [GLRLM_RLNN]);
			glrlm.calc_RP (r.fvals [GLRLM_RP]);
			glrlm.calc_GLV (r.fvals [GLRLM_GLV]);
			glrlm.calc_RV (r.fvals [GLRLM_RV]);
			glrlm.calc_RE (r.fvals [GLRLM_RE]);
			glrlm.calc_LGLRE (r.fvals [GLRLM_LGLRE]);
			glrlm.calc_HGLRE (r.fvals [GLRLM_HGLRE]);
			glrlm.calc_SRLGLE (r.fvals [GLRLM_SRLGLE]);
			glrlm.calc_SRHGLE (r.fvals [GLRLM_SRHGLE]);
			glrlm.calc_LRLGLE (r.fvals [GLRLM_LRLGLE]);
			glrlm.calc_LRHGLE (r.fvals [GLRLM_LRHGLE]);
		}
	}	

	//==== GLSZM
	if (theFeatureSet.anyEnabled({
		GLSZM_SAE,
		GLSZM_LAE,
		GLSZM_GLN,
		GLSZM_GLNN,
		GLSZM_SZN,
		GLSZM_SZNN,
		GLSZM_ZP,
		GLSZM_GLV,
		GLSZM_ZV,
		GLSZM_ZE,
		GLSZM_LGLZE,
		GLSZM_HGLZE,
		GLSZM_SALGLE,
		GLSZM_SAHGLE,
		GLSZM_LALGLE,
		GLSZM_LAHGLE
		}))
	{
		STOPWATCH("GLSZM ...", "\tReduced GLSZM");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;
			ImageMatrix im(r.raw_pixels, r.aabb);
			GLSZM_features glszm;
			glszm.initialize ((int) r.fvals[MIN][0], (int) r.fvals[MAX][0], im);
			r.fvals[GLSZM_SAE][0] = glszm.calc_SAE();
			r.fvals[GLSZM_LAE][0] = glszm.calc_LAE();
			r.fvals[GLSZM_GLN][0] = glszm.calc_GLN();
			r.fvals[GLSZM_GLNN][0] = glszm.calc_GLNN();
			r.fvals[GLSZM_SZN][0] = glszm.calc_SZN();
			r.fvals[GLSZM_SZNN][0] = glszm.calc_SZNN();
			r.fvals[GLSZM_ZP][0] = glszm.calc_ZP();
			r.fvals[GLSZM_GLV][0] = glszm.calc_GLV();
			r.fvals[GLSZM_ZV][0] = glszm.calc_ZV();
			r.fvals[GLSZM_ZE][0] = glszm.calc_ZE();
			r.fvals[GLSZM_LGLZE][0] = glszm.calc_LGLZE();
			r.fvals[GLSZM_HGLZE][0] = glszm.calc_HGLZE();
			r.fvals[GLSZM_SALGLE][0] = glszm.calc_SALGLE();
			r.fvals[GLSZM_SAHGLE][0] = glszm.calc_SAHGLE();
			r.fvals[GLSZM_LALGLE][0] = glszm.calc_LALGLE();
			r.fvals[GLSZM_LAHGLE][0] = glszm.calc_LAHGLE();
		}
	}

	//==== GLDM
	if (theFeatureSet.anyEnabled({ 
		GLDM_SDE,
		GLDM_LDE,
		GLDM_GLN,
		GLDM_DN,
		GLDM_DNN,
		GLDM_GLV,
		GLDM_DV,
		GLDM_DE,
		GLDM_LGLE,
		GLDM_HGLE,
		GLDM_SDLGLE,
		GLDM_SDHGLE,
		GLDM_LDLGLE,
		GLDM_LDHGLE 
		}))
	{
		STOPWATCH("GLDM ...", "\tReduced GLDM");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;
			ImageMatrix im(r.raw_pixels, r.aabb);
			GLDM_features gldm;
			gldm.initialize ((int) r.fvals[MIN][0], (int) r.fvals[MAX][0], im);
			r.fvals[GLDM_SDE][0] = gldm.calc_SDE();
			r.fvals[GLDM_LDE][0] = gldm.calc_LDE();
			r.fvals[GLDM_GLN][0] = gldm.calc_GLN();
			r.fvals[GLDM_DN][0] = gldm.calc_DN();
			r.fvals[GLDM_DNN][0] = gldm.calc_DNN();
			r.fvals[GLDM_GLV][0] = gldm.calc_GLV();
			r.fvals[GLDM_DV][0] = gldm.calc_DV();
			r.fvals[GLDM_DE][0] = gldm.calc_DE();
			r.fvals[GLDM_LGLE][0] = gldm.calc_LGLE();
			r.fvals[GLDM_HGLE][0] = gldm.calc_HGLE();
			r.fvals[GLDM_SDLGLE][0] = gldm.calc_SDLGLE();
			r.fvals[GLDM_SDHGLE][0] = gldm.calc_SDHGLE();
			r.fvals[GLDM_LDLGLE][0] = gldm.calc_LDLGLE();
			r.fvals[GLDM_LDHGLE][0] = gldm.calc_LDHGLE();
		}
	}

	//==== NGTDM
	if (theFeatureSet.anyEnabled({ 
	NGTDM_COARSENESS,
	NGTDM_CONTRAST,
	NGTDM_BUSYNESS,
	NGTDM_COMPLEXITY,
	NGTDM_STRENGTH
		}))
	{
		STOPWATCH("NGTDM ...", "\tReduced NGTDM");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;
			ImageMatrix im(r.raw_pixels, r.aabb);
			NGTDM_features ngtdm;
			ngtdm.initialize ((int) r.fvals[MIN][0], (int) r.fvals[MAX][0], im);

			r.fvals[NGTDM_COARSENESS][0] = ngtdm.calc_Coarseness();
			r.fvals[NGTDM_CONTRAST][0] = ngtdm.calc_Contrast();
			r.fvals[NGTDM_BUSYNESS][0] = ngtdm.calc_Busyness();
			r.fvals[NGTDM_COMPLEXITY][0] = ngtdm.calc_Complexity();
			r.fvals[NGTDM_STRENGTH][0] = ngtdm.calc_Strength();
 		}
	}

	//==== Moments
	if (theFeatureSet.anyEnabled({ 
		SPAT_MOMENT_00,
		SPAT_MOMENT_01,
		SPAT_MOMENT_02,
		SPAT_MOMENT_03,
		SPAT_MOMENT_10,
		SPAT_MOMENT_11,
		SPAT_MOMENT_12,
		SPAT_MOMENT_20,
		SPAT_MOMENT_21,
		SPAT_MOMENT_30,
	
		CENTRAL_MOMENT_02,
		CENTRAL_MOMENT_03,
		CENTRAL_MOMENT_11,
		CENTRAL_MOMENT_12,
		CENTRAL_MOMENT_20,
		CENTRAL_MOMENT_21,
		CENTRAL_MOMENT_30,

		NORM_CENTRAL_MOMENT_02,
		NORM_CENTRAL_MOMENT_03,
		NORM_CENTRAL_MOMENT_11,
		NORM_CENTRAL_MOMENT_12,
		NORM_CENTRAL_MOMENT_20,
		NORM_CENTRAL_MOMENT_21,
		NORM_CENTRAL_MOMENT_30,

		HU_M1,
		HU_M2,
		HU_M3,
		HU_M4,
		HU_M5,
		HU_M6,
		HU_M7, 
		
		WEIGHTED_HU_M1,
		WEIGHTED_HU_M2,
		WEIGHTED_HU_M3,
		WEIGHTED_HU_M4,
		WEIGHTED_HU_M5,
		WEIGHTED_HU_M6,
		WEIGHTED_HU_M7 }))
	{
		STOPWATCH("Moments ...", "\tReduced moments");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;
			ImageMatrix im (r.raw_pixels, r.aabb);

			// Prepare the contour if necessary
			if (r.contour.contour_pixels.size() == 0)
				r.contour.calculate(im);

			ImageMatrix weighted_im(r.raw_pixels, r.aabb);
			weighted_im.apply_distance_to_contour_weights (r.raw_pixels, r.contour.contour_pixels);
			HuMoments hu;
			hu.initialize ((int) r.fvals[MIN][0], (int) r.fvals[MAX][0], im, weighted_im);

			double m1, m2, m3, m4, m5, m6, m7, m8, m9, m10;
			std::tie (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10) = hu.getSpatialMoments();
			r.fvals[SPAT_MOMENT_00][0] = m1;
			r.fvals[SPAT_MOMENT_01][0] = m2;
			r.fvals[SPAT_MOMENT_02][0] = m3;
			r.fvals[SPAT_MOMENT_03][0] = m4;
			r.fvals[SPAT_MOMENT_10][0] = m5; 
			r.fvals[SPAT_MOMENT_11][0] = m6; 
			r.fvals[SPAT_MOMENT_12][0] = m7;
			r.fvals[SPAT_MOMENT_20][0] = m8;
			r.fvals[SPAT_MOMENT_21][0] = m9;
			r.fvals[SPAT_MOMENT_30][0] = m10;

			std::tie (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10) = hu.getWeightedSpatialMoments();
			r.fvals[WEIGHTED_SPAT_MOMENT_00][0] = m1;
			r.fvals[WEIGHTED_SPAT_MOMENT_01][0] = m2;
			r.fvals[WEIGHTED_SPAT_MOMENT_02][0] = m3;
			r.fvals[WEIGHTED_SPAT_MOMENT_03][0] = m4;
			r.fvals[WEIGHTED_SPAT_MOMENT_10][0] = m5;
			r.fvals[WEIGHTED_SPAT_MOMENT_11][0] = m6;
			r.fvals[WEIGHTED_SPAT_MOMENT_12][0] = m7;
			r.fvals[WEIGHTED_SPAT_MOMENT_20][0] = m8;
			r.fvals[WEIGHTED_SPAT_MOMENT_21][0] = m9;
			r.fvals[WEIGHTED_SPAT_MOMENT_30][0] = m10;

			std::tie (m1, m2, m3, m4, m5, m6, m7) = hu.getCentralMoments();
			r.fvals[CENTRAL_MOMENT_02][0] = m1;
			r.fvals[CENTRAL_MOMENT_03][0] = m2;
			r.fvals[CENTRAL_MOMENT_11][0] = m3;
			r.fvals[CENTRAL_MOMENT_12][0] = m4;
			r.fvals[CENTRAL_MOMENT_20][0] = m5;
			r.fvals[CENTRAL_MOMENT_21][0] = m6;
			r.fvals[CENTRAL_MOMENT_30][0] = m7;
			
			std::tie(m1, m2, m3, m4, m5, m6, m7) = hu.getWeightedCentralMoments();
			r.fvals[WEIGHTED_CENTRAL_MOMENT_02][0] = m1;
			r.fvals[WEIGHTED_CENTRAL_MOMENT_03][0] = m2;
			r.fvals[WEIGHTED_CENTRAL_MOMENT_11][0] = m3;
			r.fvals[WEIGHTED_CENTRAL_MOMENT_12][0] = m4;
			r.fvals[WEIGHTED_CENTRAL_MOMENT_20][0] = m5;
			r.fvals[WEIGHTED_CENTRAL_MOMENT_21][0] = m6;
			r.fvals[WEIGHTED_CENTRAL_MOMENT_30][0] = m7;

			std::tie (m1, m2, m3, m4, m5, m6, m7) = hu.getNormCentralMoments();
			r.fvals[NORM_CENTRAL_MOMENT_02][0] = m1;
			r.fvals[NORM_CENTRAL_MOMENT_03][0] = m2;
			r.fvals[NORM_CENTRAL_MOMENT_11][0] = m3;
			r.fvals[NORM_CENTRAL_MOMENT_12][0] = m4;
			r.fvals[NORM_CENTRAL_MOMENT_20][0] = m5;
			r.fvals[NORM_CENTRAL_MOMENT_21][0] = m6;
			r.fvals[NORM_CENTRAL_MOMENT_30][0] = m7;

			std::tie(m1, m2, m3, m4, m5, m6, m7) = hu.getNormSpatialMoments();
			r.fvals[NORM_SPAT_MOMENT_00][0] = m1;
			r.fvals[NORM_SPAT_MOMENT_01][0] = m2;
			r.fvals[NORM_SPAT_MOMENT_02][0] = m3;
			r.fvals[NORM_SPAT_MOMENT_03][0] = m4;
			r.fvals[NORM_SPAT_MOMENT_10][0] = m5;
			r.fvals[NORM_SPAT_MOMENT_20][0] = m6;
			r.fvals[NORM_SPAT_MOMENT_30][0] = m7;

			std::tie (m1, m2, m3, m4, m5, m6, m7) = hu.getHuMoments();
			r.fvals[HU_M1][0] = m1;
			r.fvals[HU_M2][0] = m2;
			r.fvals[HU_M3][0] = m3;
			r.fvals[HU_M4][0] = m4;
			r.fvals[HU_M5][0] = m5;
			r.fvals[HU_M6][0] = m6;
			r.fvals[HU_M7][0] = m7;

			std::tie (m1, m2, m3, m4, m5, m6, m7) = hu.getWeightedHuMoments();
			r.fvals[WEIGHTED_HU_M1][0] = m1;
			r.fvals[WEIGHTED_HU_M2][0] = m2;
			r.fvals[WEIGHTED_HU_M3][0] = m3;
			r.fvals[WEIGHTED_HU_M4][0] = m4;
			r.fvals[WEIGHTED_HU_M5][0] = m5;
			r.fvals[WEIGHTED_HU_M6][0] = m6;
			r.fvals[WEIGHTED_HU_M7][0] = m7;
		}
	}

	//==== Gabor features
	if (theFeatureSet.isEnabled(GABOR))
	{
		STOPWATCH("Gabor features ...", "\tGabor features");
		runParallel (parallelGabor, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Radial distribution / Zernike 2D 
	if (theFeatureSet.isEnabled(ZERNIKE2D))
	{
		STOPWATCH("Zernike2D ...", "\tReduced Zernike2D");
		runParallel(parallelReduceZernike2D, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
	if (theFeatureSet.anyEnabled({FRAC_AT_D, MEAN_FRAC, RADIAL_CV}))
	{
		STOPWATCH("Radial distribution ...", "\tReduced radial distribution");
		for (auto& ld : labelData)
		{
			auto& r = ld.second;

			// Prepare the contour if necessary
			if (r.contour.contour_pixels.size() == 0)
			{
				ImageMatrix im(r.raw_pixels, r.aabb);
				r.contour.calculate(im);

				#if 0	// Debug
				int idxCtr = Pixel2::find_center(r.raw_pixels, r.contour.contour_pixels);

				int cx = r.raw_pixels[idxCtr].x, cy = r.raw_pixels[idxCtr].y;
				std::stringstream ss;
				ss << theIntFname << " Label " << ld.first;
				im.print(ss.str(), "", cx, cy, "(*)", { {cx, cy, "(*)"} });

				r.contour.calculate(im);
				ImageMatrix imContour(r.contour.contour_pixels, r.aabb);
				imContour.print("Contour", "", cx, cy, "(o)", { {cx, cy, "(o)"} });
				#endif		
			}

			// Calculate the radial distributions
			RadialDistribution rd;
			rd.initialize (r.raw_pixels, r.contour.contour_pixels);
			r.fvals[FRAC_AT_D] = rd.get_FracAtD();
			r.fvals[MEAN_FRAC] = rd.get_MeanFrac();
			r.fvals[RADIAL_CV] = rd.get_RadialCV();
		}
	}

}



