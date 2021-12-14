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
#include "globals.h"
#include "features/chords.h"
#include "features/ellipse_fitting.h"
#include "features/euler_number.h"
#include "features/circle.h"
#include "features/extrema.h"
#include "features/fractal_dim.h"
#include "features/f_erosion_pixels.h"
#include "features/f_radial_distribution.h"
#include "features/gabor.h"
#include "features/geodetic_len_thickness.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/glszm.h"
#include "features/gldm.h"
#include "features/hexagonality_and_polygonality.h"
#include "features/ngtdm.h"
#include "features/hu.h"
#include "features/moments.h"
#include "features/particle_metrics.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/timing.h"


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
	if (theEnvironment.xyRes == 0.0)
		r.fvals[AREA_UM2][0] = 0;
	else
		r.fvals[AREA_UM2][0] = std::pow (theEnvironment.pixelSizeUm, 2);

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
	r.fvals[EDGE_INTEGRATEDINTENSITY][0] =
	r.fvals[EDGE_MAXINTENSITY][0] =
	r.fvals[EDGE_MEANINTENSITY][0] =
	r.fvals[EDGE_MININTENSITY][0] =
	r.fvals[EDGE_STDDEVINTENSITY][0] = 0;
	
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

		//==== Reduce pixel intensity #1, including MIN and MAX
		//XXX--not using online approach any more--		lr.reduce_pixel_intensity_features();
		// --MIN, MAX
		lr.fvals[MIN][0] = lr.aux_min;
		lr.fvals[MAX][0] = lr.aux_max;

		double n = lr.raw_pixels.size();

		// --AREA
		lr.fvals[AREA_PIXELS_COUNT][0] = n;
		if (theEnvironment.xyRes > 0.0)
			lr.fvals[AREA_UM2][0] = n * std::pow (theEnvironment.pixelSizeUm, 2);

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

		//==== Calculate ROI's image matrix
		r.aux_image_matrix.use_roi (r.raw_pixels, r.aabb);

		//==== Contour, ROI perimeter, equivalent circle diameter
		//---	ImageMatrix im (r.raw_pixels, r.aabb);
		r.contour.calculate (r.aux_image_matrix);
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

void parallelReduceChords (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR & r = (*ptrLabelData)[lab];

		Chords chords;
		double cenx = r.fvals[CENTROID_X][0],
			ceny = r.fvals[CENTROID_Y][0];
		chords.initialize(r.raw_pixels, r.aabb, cenx, ceny);

		double
			_max = 0,
			_min = 0,
			_median = 0,
			_mean = 0,
			_mode = 0,
			_stddev = 0,
			_min_angle = 0,
			_max_angle = 0;

		std::tie (_max, _min, _median, _mean, _mode, _stddev, _min_angle, _max_angle) = chords.get_maxchords_stats();
		r.fvals[MAXCHORDS_MAX][0] = _max;
		r.fvals[MAXCHORDS_MAX_ANG][0] = _max_angle;
		r.fvals[MAXCHORDS_MIN][0] = _min;
		r.fvals[MAXCHORDS_MIN_ANG][0] = _min_angle;
		r.fvals[MAXCHORDS_MEDIAN][0] = _median;
		r.fvals[MAXCHORDS_MEAN][0] = _mean;
		r.fvals[MAXCHORDS_MODE][0] = _mode;
		r.fvals[MAXCHORDS_STDDEV][0] = _stddev;

		std::tie(_max, _min, _median, _mean, _mode, _stddev, _min_angle, _max_angle) = chords.get_allchords_stats();
		r.fvals[ALLCHORDS_MAX][0] = _max;
		r.fvals[ALLCHORDS_MAX_ANG][0] = _max_angle;
		r.fvals[ALLCHORDS_MIN][0] = _min;
		r.fvals[ALLCHORDS_MIN_ANG][0] = _min_angle;
		r.fvals[ALLCHORDS_MEDIAN][0] = _median;
		r.fvals[ALLCHORDS_MEAN][0] = _mean;
		r.fvals[ALLCHORDS_MODE][0] = _mode;
		r.fvals[ALLCHORDS_STDDEV][0] = _stddev;
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

	//=== Make ROI labels indexable 
	sortedUniqueLabels.clear();
	for (auto l : uniqueLabels)
		sortedUniqueLabels.push_back(l);
	
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
		STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
		runParallel (parallelReduceIntensityStats, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Neighbors
	if (theFeatureSet.anyEnabled({ NUM_NEIGHBORS, CLOSEST_NEIGHBOR1_DIST, CLOSEST_NEIGHBOR2_DIST }))
	{
		STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
		reduce_neighbors (5 /* collision radius [pixels] */);
	}

	//==== Fitting an ellipse
	if (theFeatureSet.anyEnabled({ MAJOR_AXIS_LENGTH, MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION, ROUNDNESS } ))
	{
		STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
		runParallel (EllipseFittingFeatures::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Contour-related ROI perimeter, equivalent circle diameter
	if (theFeatureSet.anyEnabled({ EQUIVALENT_DIAMETER, PERIMETER,
			CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY, // depend on PERIMETER
			EDGE_INTEGRATEDINTENSITY,
			EDGE_MAXINTENSITY,
			EDGE_MININTENSITY,
			EDGE_MEANINTENSITY,
			EDGE_STDDEVINTENSITY // depend on contour and PERIMETER
		}))
	{
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			runParallel (parallelReduceContour, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
		}
		//==== Convex hull related Solidity, Circularity, IntegratedIntensityEdge, MaxIntensityEdge, MinIntensityEdge, etc
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
			STOPWATCH("Morphology/Hull/H/#4aaaea", "\t=");
			runParallel (parallelReduceConvHull, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
		}	
	}

	//==== Extrema 
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
		EXTREMA_P8_X }))
	{
		STOPWATCH("Morphology/Extrema/Ex/#4aaaea", "\t=");
		runParallel (ExtremaFeatures::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Euler 
	if (theFeatureSet.isEnabled (EULER_NUMBER))
	{
		STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
		runParallel (EulerNumber::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
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
		STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
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
		STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
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
		STOPWATCH("Morphology/Nassenstein/N/#4aaaea", "\t=");
		runParallel(parallelReduceNassenstein, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Chords
	if (theFeatureSet.anyEnabled({ 
		MAXCHORDS_MAX,
		MAXCHORDS_MAX_ANG,
		MAXCHORDS_MIN,
		MAXCHORDS_MIN_ANG,
		MAXCHORDS_MEDIAN,
		MAXCHORDS_MEAN,
		MAXCHORDS_MODE,
		MAXCHORDS_STDDEV,
		ALLCHORDS_MAX,
		ALLCHORDS_MAX_ANG,
		ALLCHORDS_MIN,
		ALLCHORDS_MIN_ANG,
		ALLCHORDS_MEDIAN,
		ALLCHORDS_MEAN,
		ALLCHORDS_MODE,
		ALLCHORDS_STDDEV, }))
	{
		STOPWATCH("Morphology/Chords/Ch/#4aaaea", "\t=");
		runParallel(parallelReduceChords, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Hexagonality and polygonality
	if (theFeatureSet.anyEnabled ({POLYGONALITY_AVE, HEXAGONALITY_AVE, HEXAGONALITY_STDDEV}))
	{
		STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
		runParallel (Hexagonality_and_Polygonality::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Enclosing, inscribing, and circumscribing circle
	if (theFeatureSet.anyEnabled ({ DIAMETER_MIN_ENCLOSING_CIRCLE, DIAMETER_INSCRIBING_CIRCLE, DIAMETER_CIRCUMSCRIBING_CIRCLE }))
	{
		STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
		runParallel (EnclosingInscribingCircumscribingCircle::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Geodetic length and thickness
	if (theFeatureSet.anyEnabled({GEODETIC_LENGTH, THICKNESS}))
	{
		STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
		runParallel (GeodeticLength_and_Thickness::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== ROI radius
	if (theFeatureSet.anyEnabled({
		ROI_RADIUS_MEAN,
		ROI_RADIUS_MAX,
		ROI_RADIUS_MEDIAN
		}))
	{
		STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
		runParallel (RoiRadius::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Erosion pixels
	if (theFeatureSet.anyEnabled({ EROSIONS_2_VANISH, EROSIONS_2_VANISH_COMPLEMENT }))
	{
		STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
		runParallel (ErosionPixels::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}	

	//==== Fractal dimension
	if (theFeatureSet.anyEnabled({ FRACT_DIM_BOXCOUNT, FRACT_DIM_PERIMETER }))
	{
		STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
		runParallel (FractalDimension::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== GLCM aka Haralick 2D 
	if (theFeatureSet.anyEnabled({ 
		GLCM_ANGULAR2NDMOMENT,
		GLCM_CONTRAST,
		GLCM_CORRELATION,
		GLCM_VARIANCE,
		GLCM_INVERSEDIFFERENCEMOMENT,
		GLCM_SUMAVERAGE,
		GLCM_SUMVARIANCE,
		GLCM_SUMENTROPY,
		GLCM_ENTROPY,
		GLCM_DIFFERENCEVARIANCE,
		GLCM_DIFFERENCEENTROPY,
		GLCM_INFOMEAS1,
		GLCM_INFOMEAS2 }))
	{
		STOPWATCH("Texture/GLCM texture/GLCM/#bbbbbb", "\t=");
		runParallel (GLCM_features::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
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
		STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
		runParallel (GLRLM_features::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
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
		STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
		runParallel (GLSZM_features::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
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
		STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
		runParallel (GLDM_features::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
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
		STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
		runParallel (NGTDM_features::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
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
		STOPWATCH("Moments/Moments/2D moms/#FFFACD", "\t=");
		runParallel (HuMoments::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Gabor features
	if (theFeatureSet.isEnabled(GABOR))
	{
		STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
		runParallel (GaborFeatures::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Radial distribution / Zernike 2D 
	if (theFeatureSet.isEnabled(ZERNIKE2D))
	{
		STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
		runParallel (parallelReduceZernike2D, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

	//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
	if (theFeatureSet.anyEnabled({FRAC_AT_D, MEAN_FRAC, RADIAL_CV}))
	{
		STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
		runParallel (RadialDistribution::reduce, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
	}

}



