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
#include "environment.h"
#include "sensemaker.h"
#include "glrlm.h"
#include "glszm.h"
#include "timing.h"


constexpr int N2R = 100 * 1000;
constexpr int N2R_2 = 100 * 1000;
constexpr int smallestROI = 10;

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

	r.fvals[AREA_PIXELS_COUNT][0] = r.pixelCountRoiArea = 1;
	r.aux_PrevCount = 0;
	// Min
	r.fvals[MIN][0] = intensity; // r.min = intensity;
	// Max
	r.fvals[MAX][0] = intensity; // r.max = intensity;
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
	r.fvals[MEDIAN][0] = 0; // r.median = 0;
	r.fvals[STANDARD_DEVIATION][0] = 0; // r.stddev = 0;
	r.fvals[SKEWNESS][0] = 0; // r.skewness = 0;
	r.fvals[KURTOSIS][0] = 0; // r.kurtosis = 0;
	r.fvals[ROOT_MEAN_SQUARED][0] = 0; // r.RMS = 0;
	r.fvals[P10][0] = r.fvals[P25][0] = r.fvals[P75][0] = r.fvals[90][0] = 0; // r.p10 = r.p25 = r.p75 = r.p90 = 0;
	r.fvals[INTERQUARTILE_RANGE][0] = 0; // r.IQR = 0;
	r.fvals[ENTROPY][0] = 0; // r.entropy = 0;
	r.fvals[MODE][0] = 0; // r.mode = 0;
	r.fvals[UNIFORMITY][0] = 0; // r.uniformity = 0;
	r.fvals[ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = 0; // r.RMAD = 0;
	// CellProfiler	
	r.fvals[CELLPROFILER_INTENSITY_INTEGRATEDINTENSITYEDGE][0] = // r.CellProfiler_Intensity_IntegratedIntensityEdge = 
	r.fvals[CELLPROFILER_INTENSITY_MAXINTENSITYEDGE][0] = // r.CellProfiler_Intensity_MaxIntensityEdge = 
	r.fvals[CELLPROFILER_INTENSITY_MEANINTENSITYEDGE][0] = // r.CellProfiler_Intensity_MeanIntensityEdge = 
	r.fvals[CELLPROFILER_INTENSITY_MININTENSITYEDGE][0] = // r.CellProfiler_Intensity_MinIntensityEdge = 
	r.fvals[CELLPROFILER_INTENSITY_STDDEVINTENSITYEDGE][0] = 0; // r.CellProfiler_Intensity_StddevIntensityEdge = 0;
	
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

		if (lr.raw_pixels.size() < smallestROI)
		{
			lr.roi_disabled = true;
			continue;
		}

		//==== Reduce pixel intensity #1, including MIN and MAX
		lr.reduce_pixel_intensity_features();

		//==== Do not calculate features of all-blank intensities (to avoid NANs)
		if (lr.intensitiesAllZero())
			continue;

		auto n = lr.pixelCountRoiArea;	// Cardinality of the label value set

		//==== Pixel intensity stats

		// Mean absolute deviation
		lr.fvals[MEAN_ABSOLUTE_DEVIATION][0] /= n; // lr.MAD = lr.MAD / n;

		// Standard deviations
		lr.fvals[STANDARD_DEVIATION][0] = sqrt(lr.aux_variance); // lr.stddev = sqrt(lr.aux_variance);

		// Skewness
		lr.fvals[SKEWNESS][0] // lr.skewness 
			= std::sqrt(double(lr.pixelCountRoiArea)) * lr.aux_M3 / std::pow(lr.aux_M2, 1.5);

		// Kurtosis
		lr.fvals[KURTOSIS][0] // lr.kurtosis 
			= double(lr.pixelCountRoiArea) * lr.aux_M4 / (lr.aux_M2 * lr.aux_M2) - 3.0;

		// Root of mean squared
		lr.fvals[ROOT_MEAN_SQUARED][0] = sqrt(lr.fvals[ENERGY][0] / n); // lr.RMS = sqrt(lr.massEnergy / n);

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
		auto [mean_, mode_, p10_, p25_, p75_, p90_, iqr_, rmad_, entropy_, uniformity_] = H.get_stats();

		lr.fvals[MEDIAN][0] = mean_; // lr.median = mean_;
		lr.fvals[P10][0] = p10_; // lr.p10 = p10_;
		lr.fvals[P25][0] = p25_; // lr.p25 = p25_;
		lr.fvals[P75][0] = p75_; // lr.p75 = p75_;
		lr.fvals[P90][0] = p90_; // lr.p90 = p90_;
		lr.fvals[INTERQUARTILE_RANGE][0] = iqr_; // lr.IQR = iqr_;
		lr.fvals[ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = rmad_; // lr.RMAD = rmad_;
		lr.fvals[ENTROPY][0] = entropy_; // lr.entropy = entropy_;
		lr.fvals[MODE][0] = mode_; // lr.mode = mode_;
		lr.fvals[UNIFORMITY][0] = uniformity_; // lr.uniformity = uniformity_;

		// Weighted centroids
		lr.fvals[CENTROID_X][0] /= lr.pixelCountRoiArea; // lr.centroid_x = lr.centroid_x / lr.pixelCountRoiArea;
		lr.fvals[CENTROID_Y][0] /= lr.pixelCountRoiArea; // lr.centroid_y = lr.centroid_y / lr.pixelCountRoiArea;

		//==== Calculate pixel intensity stats directly rather than online
		int min_online_roi_size = 100;
		if (min_online_roi_size > 0)
		{
			// -- Standard deviation
			StatsReal sumM2 = 0.0,
				sumM3 = 0.0,
				sumM4 = 0.0,
				absSum = 0.0,
				sumEnergy = 0.0;
			for (auto& pix : lr.raw_pixels)
			{
				auto diff = pix.inten - lr.fvals[MEAN][0]; // lr.mean
				sumM2 += diff * diff;
				sumM3 += diff * diff * diff;
				sumM4 += diff * diff * diff * diff;
				absSum += abs(diff);
				sumEnergy += pix.inten * pix.inten;
			}
			double m2 = sumM2 / (n - 1),
				m3 = sumM3 / n,
				m4 = sumM4 / n;
			lr.aux_variance = sumM2 / (n - 1);
			lr.fvals[STANDARD_DEVIATION][0] = sqrt(lr.aux_variance);	// .stddev
			lr.fvals[MEAN_ABSOLUTE_DEVIATION][0] = absSum / n;	// .MAD
			lr.fvals[ROOT_MEAN_SQUARED][0] = sqrt(sumEnergy / n);	// .RMS
			lr.fvals[SKEWNESS][0] = m3 / pow(m2, 1.5);
			lr.fvals[KURTOSIS][0] = m4 / (m2 * m2) - 3;
			//
		}

		//==== Extent
		double bbArea = lr.aabb.get_area();
		lr.fvals[EXTENT][0] = (double)lr.pixelCountRoiArea / (double)bbArea;

		//==== Aspect ratio
		lr.fvals[ASPECT_RATIO][0] = lr.aabb.get_width() / lr.aabb.get_height();	// .aspectRatio

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

		//if (theEnvironment.verbosity_level & VERBOSITY_DETAILED)
		//	std::cout << "Contour for ROI " << lab << " of " << r.raw_pixels.size() << " pixels, equivalent matrix " << im.height << " x " << im.width << " - calculating\n";

		r.contour.calculate (im);
		r.fvals[PERIMETER][0] = r.contour.get_roi_perimeter();	
		r.fvals[EQUIVALENT_DIAMETER][0] = r.contour.get_diameter_equal_perimeter();	

		//	if (theEnvironment.verbosity_level & VERBOSITY_DETAILED)
		//		std::cout << "Contour for ROI " << lab << " length=" << r.contour.contour_pixels.size() << " ROI perimeter=" << r.fvals[PERIMETER][0] << " diameter_equal_perimeter=" << r.fvals[EQUIVALENT_DIAMETER][0]  << "\n";
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
		r.fvals[SOLIDITY][0] = r.pixelCountRoiArea / r.fvals[CONVEX_HULL_AREA][0];	// .solidity

		//==== Circularity
		r.fvals[CIRCULARITY][0] = 4.0 * M_PI * r.pixelCountRoiArea / (r.fvals[PERIMETER][0] * r.fvals[PERIMETER][0]); // r.circularity = 4.0 * M_PI * r.pixelCountRoiArea / (r.roiPerimeter * r.roiPerimeter);

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
			r.fvals[TEXTURE_ZERNIKE2D]);	// .Zernike2D

		// Fix calculated feature values due to all-0 intensity labels to avoid NANs in the output
		if (r.intensitiesAllZero())
		{
			for (int i = 0; i < r.fvals[TEXTURE_ZERNIKE2D].size(); i++)
				r.fvals[TEXTURE_ZERNIKE2D][i] = 0.0;
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
	if (theFeatureSet.isEnabled(NUM_NEIGHBORS))
	{
		STOPWATCH("Neighbors ...", "\tReduced neighbors");
		reduce_neighbors (5 /* collision radius [pixels] */);
	}

	//==== Fitting an ellipse
	if (theFeatureSet.anyEnabled({ MAJOR_AXIS_LENGTH, MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION } ))
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

			double uxx = XSquaredTmp / r.pixelCountRoiArea + 1.0 / 12.0;
			double uyy = YSquaredTmp / r.pixelCountRoiArea + 1.0 / 12.0;
			double uxy = XYSquaredTmp / r.pixelCountRoiArea;

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
			auto [polyAve, hexAve, hexSd] = hp.calculate(r.fvals[NUM_NEIGHBORS][0], r.pixelCountRoiArea, r.fvals[PERIMETER][0], r.fvals[CONVEX_HULL_AREA][0], r.fvals[MIN_FERET_DIAMETER][0], r.fvals[MAX_FERET_DIAMETER][0]);
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
			auto [geoLen, thick] = glt.calculate(r.pixelCountRoiArea, r.fvals[PERIMETER][0]);
			r.fvals[GEODETIC_LENGTH][0] = geoLen;
			r.fvals[THICKNESS][0] = thick;
		}
	}

	//==== Haralick 2D 
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

	//==== Zernike 2D 
	if (theFeatureSet.isEnabled(TEXTURE_ZERNIKE2D))
	{
		STOPWATCH("Zernike2D ...", "\tReduced Zernike2D");
		runParallel (parallelReduceZernike2D, nThr, workPerThread, tileSize, &sortedUniqueLabels, &labelData);
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
}



