#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include "sensemaker.h"

constexpr int N2R = 50 * 1000;
constexpr int N2R_2 = 50 * 1000;

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
void init_label_record (LR& lr, int x, int y, int label, PixIntens intensity)
{
	// Save the pixel
	lr.raw_pixels.push_back(Pixel2(x, y, intensity));

	lr.pixelCount = 1;
	lr.labelPrevCount = 0;
	// Min
	lr.labelMins = intensity;
	// Max
	lr.labelMaxs = intensity;
	// Moments
	lr.mean = intensity;
	lr.labelM2 = 0;
	lr.labelM3 = 0;
	lr.labelM4 = 0;
	// Energy
	lr.labelMassEnergy = intensity * intensity;
	// Variance and standard deviation
	lr.labelVariance = 0.0;
	// Mean absolute deviation
	lr.labelMAD = 0;
	// Previous intensity
	lr.labelPrevIntens = intensity;
	// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
	lr.centroid_x = StatsReal(x) + 1;
	lr.centroid_y = StatsReal(y) + 1;
	// Histogram
	std::shared_ptr<Histo> ptrH = std::make_shared <Histo>();
	ptrH->add_observation(intensity);
	lr.labelHistogram = ptrH;
	// Other fields
	lr.labelMedians = 0;
	lr.labelStddev = 0;
	lr.labelSkewness = 0;
	lr.labelKurtosis = 0;
	lr.labelRMS = 0;
	lr.labelP10 = lr.labelP25 = lr.labelP75 = lr.labelP90 = 0;
	lr.labelIQR = 0;
	lr.labelEntropy = 0;
	lr.labelMode = 0;
	lr.labelUniformity = 0;
	lr.labelRMAD = 0;

	//==== Morphology
	lr.init_aabb (x, y);

	#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
	// Dump intensities for testing
	if (label == SANITY_CHECK_INTENSITIES_FOR_LABEL)	// Put your label code of interest
		lr.raw_intensities.push_back(intensity);
	#endif

}

// This function 'digests' the 2nd and the following pixel of a label and updates the label's feature calculation state - the instance of structure 'LR'
void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity)
{
	// Save the pixel
	lr.raw_pixels.push_back(Pixel2(x, y, intensity));
	
	// Count of pixels belonging to the label
	auto prev_n = lr.pixelCount;	// Previous count
	lr.labelPrevCount = prev_n;
	auto n = prev_n + 1;	// New count
	lr.pixelCount = n;

	// Cumulants for moments calculation
	auto prev_mean = lr.mean;
	auto delta = intensity - prev_mean;
	auto delta_n = delta / n;
	auto delta_n2 = delta_n * delta_n;
	auto term1 = delta * delta_n * prev_n;

	// Mean
	auto mean = prev_mean + delta_n;
	lr.mean = mean;

	// Moments
	lr.labelM4 = lr.labelM4 + term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * lr.labelM2 - 4 * delta_n * lr.labelM3;
	lr.labelM3 = lr.labelM3 + term1 * delta_n * (n - 2) - 3 * delta_n * lr.labelM2;
	lr.labelM2 = lr.labelM2 + term1;

	// Min 
	lr.labelMins = std::min(lr.labelMins, (StatsInt)intensity);

	// Max
	lr.labelMaxs = std::max(lr.labelMaxs, (StatsInt)intensity);

	// Energy
	lr.labelMassEnergy = lr.labelMassEnergy + intensity * intensity;

	// Variance and standard deviation
	if (n >= 2)
	{
		double s_prev = lr.labelVariance,
			diff = double(intensity) - prev_mean,
			diff2 = diff * diff;
		lr.labelVariance = (n - 2) * s_prev / (n - 1) + diff2 / n;
	}
	else
		lr.labelVariance = 0;

	// Mean absolute deviation
	lr.labelMAD = lr.labelMAD + std::abs(intensity - mean);

	// Weighted centroids. Needs reduction. Do we need to make them 1-based for compatibility with Matlab and WNDCHRM?
	lr.centroid_x = lr.centroid_x + StatsReal(x);
	lr.centroid_y = lr.centroid_y + StatsReal(y);

	// Histogram
	auto ptrH = lr.labelHistogram;
	ptrH->add_observation(intensity);

	// Previous intensity for succeeding iterations
	lr.labelPrevIntens = intensity;

	//==== Morphology
	lr.update_aabb (x, y);

	#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
	// Dump intensities for testing
	if (label == SANITY_CHECK_INTENSITIES_FOR_LABEL)	// Put the label code you're tracking
		lr.raw_intensities.push_back(intensity);
	#endif
}

// The root function of handling a pixel being scanned
void update_label_stats(int x, int y, int label, PixIntens intensity)
{
	auto it = uniqueLabels.find(label);
	if (it == uniqueLabels.end())
	{
		// Remember this label
		uniqueLabels.insert(label);

		// Initialize the label record
		LR lr;
		init_label_record (lr, x, y, label, intensity);
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

// This function should be called once after a file pair processing is finished.
void reduce_all_labels (int min_online_roi_size)
{

	std::cout << "\treducing intensity stats" << std::endl;

	//==== Scan pixels. This will be followed by the Reduce step
	for (auto& ld : labelData) // for (auto& lv : labelUniqueIntensityValues)
	{
		auto l = ld.first;		// Label code
		auto& lr = ld.second;	// Label record

		auto n = lr.pixelCount;	// Cardinality of the label value set

		//==== Pixel intensity stats

		// Mean absolute deviation
		lr.labelMAD = lr.labelMAD / n;

		// Standard deviations
		lr.labelStddev = sqrt(lr.labelVariance);

		// Skewness
		lr.labelSkewness = std::sqrt(double(lr.pixelCount)) * lr.labelM3 / std::pow(lr.labelM2, 1.5);	

		// Kurtosis
		lr.labelKurtosis = double(lr.pixelCount) * lr.labelM4 / (lr.labelM2 * lr.labelM2) - 3.0;	

		// Root of mean squared
		lr.labelRMS = sqrt(lr.labelMassEnergy / n);

		// P10, 25, 75, 90
		auto ptrH = lr.labelHistogram;
		ptrH->build_histogram();
		auto [mean_, mode_, p10_, p25_, p75_, p90_, iqr_, rmad_, entropy_, uniformity_] = ptrH->get_stats();

		lr.labelMedians = mean_;
		lr.labelP10 = p10_; 
		lr.labelP25 = p25_; 
		lr.labelP75 = p75_; 
		lr.labelP90 = p90_; 
		lr.labelIQR = iqr_; 
		lr.labelRMAD = rmad_;
		lr.labelEntropy = entropy_;
		lr.labelMode = mode_; 
		lr.labelUniformity = uniformity_;

		// Weighted centroids
		lr.centroid_x = lr.centroid_x / lr.pixelCount;
		lr.centroid_y = lr.centroid_y / lr.pixelCount;

		//==== Calculate pixel intensity stats directly rather than online
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
				auto diff = pix.inten - lr.mean;
				sumM2 += diff * diff;
				sumM3 += diff * diff * diff;
				sumM4 += diff * diff * diff * diff;
				absSum += abs(diff);
				sumEnergy += pix.inten * pix.inten;
			}
			double m2 = sumM2 / (n - 1), 
				m3 = sumM3 / n,
				m4 = sumM4 / n;
			lr.labelVariance = sumM2 / (n - 1);
			lr.labelStddev = sqrt (lr.labelVariance);
			lr.labelMAD = absSum / n;	// Biased
			lr.labelRMS = sqrt (sumEnergy / n);
			lr.labelSkewness = m3 / pow(m2, 1.5);
			lr.labelKurtosis = m4 / (m2 * m2) - 3;
			//
		}

		//==== Morphology
		double bbArea = lr.aabb.get_area();
		lr.extent = (double)lr.pixelCount / (double)bbArea;

	}

	//==== Neighbors
	if (featureSet.isEnabled(NUM_NEIGHBORS))
	{
		std::cout << "\treducing neighbors" << std::endl;
		reduce_neighbors (5 /*collision radius*/);
	}

	//==== Fitting an ellipse
	if (featureSet.anyEnabled({MAJOR_AXIS_LENGTH, MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION}))
	{
		// 
		//Reference: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/19028/versions/1/previews/regiondata.m/index.html
		//		
		std::cout << "\treducing ellipticity (major, minor axes, eccentricity, orientation)" << std::endl;

		// Calculate normalized second central moments for the region.
		// 1/12 is the normalized second central moment of a pixel with unit length.
		for (auto& ld : labelData) // for (auto& lv : labelUniqueIntensityValues)
		{
			auto& r = ld.second;	// Label record

			double XSquaredTmp = 0, YSquaredTmp = 0, XYSquaredTmp = 0;
			for (auto& pix : r.raw_pixels)
			{
				auto diffX = r.centroid_x - pix.x, diffY = r.centroid_y - pix.y;
				XSquaredTmp += diffX * diffX; //(double(x) - (xCentroid - Im.ROIWidthBeg)) * (double(x) - (xCentroid - Im.ROIWidthBeg));
				YSquaredTmp += diffY * diffY; //(-double(y) + (yCentroid - Im.ROIHeightBeg)) * (-double(y) + (yCentroid - Im.ROIHeightBeg));
				XYSquaredTmp += diffX * diffY; //(double(x) - (xCentroid - Im.ROIWidthBeg)) * (-double(y) + (yCentroid - Im.ROIHeightBeg));
			}

			double uxx = XSquaredTmp / r.pixelCount + 1.0 / 12.0;
			double uyy = YSquaredTmp / r.pixelCount + 1.0 / 12.0;
			double uxy = XYSquaredTmp / r.pixelCount;

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

			r.major_axis_length = MajorAxisLength;
			r.minor_axis_length = MinorAxisLength;
			r.eccentricity = Eccentricity;
			r.orientation = Orientation;

			//==== Aspect ratio
			r.aspectRatio = r.aabb.get_width() / r.aabb.get_height();
		}
	}

	std::cout << "\treducing contours, hulls, and related (circularity, solidity, ...)" << std::endl;
	for (auto& ld : labelData) // for (auto& lv : labelUniqueIntensityValues)
	{
		auto& r = ld.second;	// Label record

		//==== Contour, ROI perimeter, equivalent circle diameter

		r.cntr.calculate(r.raw_pixels);
		r.roiPerimeter = (StatsInt)r.cntr.get_roi_perimeter();
		r.equivDiam = r.cntr.get_diameter_equal_perimeter();

		//==== Convex hull and solidity
		r.convHull.calculate(r.raw_pixels);
		r.convHullArea = r.convHull.getArea();
		r.solidity = r.pixelCount / r.convHullArea;

		//==== Circularity
		r.circularity = 4.0 * M_PI * r.pixelCount / (r.roiPerimeter * r.roiPerimeter);
	}

	//==== Extrema and Euler number
	std::cout << "\treducing extrema and Euler\n";
	for (auto& ld : labelData)
	{
		auto& r = ld.second;

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

		r.extremaP1y = TopMostIndex; // -0.5 + Im.ROIHeightBeg;
		r.extremaP1x = TopMost_MostLeftIndex; // -0.5 + Im.ROIWidthBeg;

		r.extremaP2y = TopMostIndex; // -0.5 + Im.ROIHeightBeg;
		r.extremaP2x = TopMost_MostRightIndex; // +0.5 + Im.ROIWidthBeg;

		r.extremaP3y = RightMost_Top; // -0.5 + Im.ROIHeightBeg;//
		r.extremaP3x = RightMostIndex; // +0.5 + Im.ROIWidthBeg;

		r.extremaP4y = RightMost_Bottom; // +0.5 + Im.ROIHeightBeg;
		r.extremaP4x = RightMostIndex; // +0.5 + Im.ROIWidthBeg;

		r.extremaP5y = LowestIndex; // +0.5 + Im.ROIHeightBeg;
		r.extremaP5x = Lowest_MostRightIndex; // +0.5 + Im.ROIWidthBeg;

		r.extremaP6y = LowestIndex; // +0.5 + Im.ROIHeightBeg;
		r.extremaP6x = Lowest_MostLeftIndex; // -0.5 + Im.ROIWidthBeg;

		r.extremaP7y = LeftMost_Bottom; // +0.5 + Im.ROIHeightBeg;
		r.extremaP7x = LeftMostIndex; // -0.5 + Im.ROIWidthBeg;

		r.extremaP8y = LeftMost_Top; // -0.5 + Im.ROIHeightBeg;
		r.extremaP8x = LeftMostIndex; // -0.5 + Im.ROIWidthBeg;
	
		//==== Euler number
		EulerNumber eu(r.raw_pixels, r.aabb.get_xmin(), r.aabb.get_ymin(), r.aabb.get_xmax(), r.aabb.get_ymax(), 8);	// Using mode=8 following to WNDCHRM example
		r.euler_number = eu.euler_number;	
	}

	//==== Feret diameters and angles
	if (featureSet.anyEnabled ({MIN_FERET_DIAMETER, MAX_FERET_DIAMETER, MIN_FERET_ANGLE, MAX_FERET_ANGLE}) ||
		featureSet.anyEnabled({ 
			STAT_FERET_DIAM_MIN,
			STAT_FERET_DIAM_MAX,
			STAT_FERET_DIAM_MEAN,
			STAT_FERET_DIAM_MEDIAN,
			STAT_FERET_DIAM_STDDEV,
			STAT_FERET_DIAM_MODE }))
	{
		std::cout << "\treducing Feret diameter, angle, and stats\n";
		for (auto& ld : labelData)
		{
			auto& r = ld.second;

			ParticleMetrics pm(r.convHull.CH);
			std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
			pm.calc_ferret(
				r.maxFeretDiameter,
				r.maxFeretAngle,
				r.minFeretDiameter,
				r.minFeretAngle,
				allD
			);

			auto structStat = ComputeCommonStatistics2(allD);
			r.feretStats_minDiameter = (double)structStat.min;	// ratios[59]
			r.feretStats_maxDiameter = (double)structStat.max;	// ratios[60]
			r.feretStats_meanDiameter = structStat.mean;	// ratios[61]
			r.feretStats_medianDiameter = structStat.median;	// ratios[62]
			r.feretStats_stddevDiameter = structStat.stdev;	// ratios[63]
			r.feretStats_modeDiameter = (double)structStat.mode;	// ratios[64]		
		}
	}

	//==== Martin diameters
	if (featureSet.anyEnabled({ STAT_MARTIN_DIAM_MIN,
		STAT_MARTIN_DIAM_MAX,
		STAT_MARTIN_DIAM_MEAN,
		STAT_MARTIN_DIAM_MEDIAN,
		STAT_MARTIN_DIAM_STDDEV,
		STAT_MARTIN_DIAM_MODE }))
	{
		std::cout << "\treducing Martin stats\n";
		for (auto& ld : labelData)
		{
			auto& r = ld.second;


			ParticleMetrics pm(r.convHull.CH);
			std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
			pm.calc_martin(allD);
			auto structStat = ComputeCommonStatistics2(allD);
			r.martinStats_minDiameter = (double)structStat.min;
			r.martinStats_maxDiameter = (double)structStat.max;
			r.martinStats_meanDiameter = structStat.mean;
			r.martinStats_medianDiameter = structStat.median;
			r.martinStats_stddevDiameter = structStat.stdev;
			r.martinStats_modeDiameter = (double)structStat.mode;
		}
	}

	//==== Nassenstein diameters
	if (featureSet.anyEnabled({ STAT_NASSENSTEIN_DIAM_MIN,
		STAT_NASSENSTEIN_DIAM_MAX,
		STAT_NASSENSTEIN_DIAM_MEAN,
		STAT_NASSENSTEIN_DIAM_MEDIAN,
		STAT_NASSENSTEIN_DIAM_STDDEV,
		STAT_NASSENSTEIN_DIAM_MODE }))
	{
		std::cout << "\treducing Nassenstein stats\n";
		for (auto& ld : labelData)
		{
			auto& r = ld.second;


			ParticleMetrics pm(r.convHull.CH);
			std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
			pm.calc_nassenstein(allD);
			auto structStat = ComputeCommonStatistics2(allD);
			r.nassStats_minDiameter = (double)structStat.min;
			r.nassStats_maxDiameter = (double)structStat.max;
			r.nassStats_meanDiameter = structStat.mean;
			r.nassStats_medianDiameter = structStat.median;
			r.nassStats_stddevDiameter = structStat.stdev;
			r.nassStats_modeDiameter = (double)structStat.mode;
		}
	}

	std::cout << "\treducing hexagonality, polygonality, enclosing circle, geodetic length & thickness\n";
	for (auto& ld : labelData)
	{
		auto& r = ld.second;

		//==== Hexagonality and polygonality
		Hexagonality_and_Polygonality hp;
		auto [polyAve, hexAve, hexSd] = hp.calculate(r.num_neighbors, r.pixelCount, r.roiPerimeter, r.convHullArea, r.minFeretDiameter, r.maxFeretDiameter);
		r.polygonality_ave = polyAve;
		r.hexagonality_ave = hexAve;
		r.hexagonality_stddev = hexSd;

		//==== Enclosing circle
		MinEnclosingCircle cir1;
		r.diameter_min_enclosing_circle = cir1.calculate_diam (r.cntr.theContour);
		InscribingCircumscribingCircle cir2;
		auto [diamIns, diamCir] = cir2.calculateInsCir (r.cntr.theContour, r.centroid_x, r.centroid_y);
		r.diameter_inscribing_circle = diamIns;
		r.diameter_circumscribing_circle = diamCir;

		//==== Geodetic length thickness
		GeodeticLength_and_Thickness glt;
		auto [geoLen, thick] = glt.calculate(r.pixelCount, r.roiPerimeter);
	}

	//==== Haralick 2D 
	if (featureSet.isEnabled(TEXTURE_HARALICK2D))
	{
		std::cout << "\treducing Haralick 2D\n";
		for (auto& ld : labelData)
		{
			// Get ahold of the label's data:
			auto& r = ld.second;

			haralick2D(
				// in
				r.raw_pixels,	// nonzero_intensity_pixels,
				r.aabb,			// AABB info not to calculate it again from 'raw_pixels' in the function
				0.0,			// distance,
				// out
				r.Texture_Feature_Angles,
				r.Texture_AngularSecondMoments,
				r.Texture_Contrast,
				r.Texture_Correlation,
				r.Texture_Variance,
				r.Texture_InverseDifferenceMoment,
				r.Texture_SumAverage,
				r.Texture_SumVariance,
				r.Texture_SumEntropy,
				r.Texture_Entropy,
				r.Texture_DifferenceVariance,
				r.Texture_DifferenceEntropy,
				r.Texture_InfoMeas1,
				r.Texture_InfoMeas2);
		}
	}

	//==== Zernike 2D 
	std::cout << "\treducing Zernike 2D\n";
	if (featureSet.isEnabled(TEXTURE_ZERNIKE2D))
	{
		for (auto& ld : labelData) 
		{
			// Get ahold of the label's data:
			auto& r = ld.second;	

			zernike2D(
				// in
				r.raw_pixels,	// nonzero_intensity_pixels,
				r.aabb,			// AABB info not to calculate it again from 'raw_pixels' in the function
				// out
				r.Zernike2D);
		}	
	}
	
}

void reduce_neighbors (int radius)
{
	#if 0	// Leaving the greedy implementation just for the record and the time when we want to run it on a GPU
	//==== Collision detection, method 1 (best with GPGPU)
	//  Calculate collisions into a triangular matrix
	int nul = uniqueLabels.size();
	std::vector <char> CM (nul*nul, false);	// collision matrix
	// --this loop can be parallel
	for (auto l1 : uniqueLabels)
	{
		LR & r1 = labelData[l1];
		for (auto l2 : uniqueLabels)
		{
			if (l1 == l2)
				continue;	// Trivial - diagonal element
			if (l1 > l2)
				continue;	// No need to check the upper triangle

			LR& r2 = labelData[l2];
			bool noOverlap = r2.aabb.get_xmin() > r1.aabb.get_xmax()+radius || r2.aabb.get_xmax() < r1.aabb.get_xmin()-radius 
				|| r2.aabb.get_ymin() > r1.aabb.get_ymax()+radius || r2.aabb.get_ymax() < r1.aabb.get_ymin()-radius;
			if (!noOverlap)
			{
				unsigned int idx = l1 * nul + l2;	
				CM[idx] = true;
			}
		}
	}

	// Harvest collision pairs
	for (auto l1 : uniqueLabels)
	{
		LR& r1 = labelData[l1];
		for (auto l2 : uniqueLabels)
		{
			if (l1 == l2)
				continue;	// Trivial - diagonal element
			if (l1 > l2)
				continue;	// No need to check the upper triangle

			unsigned int idx = l1 * nul + l2;
			if (CM[idx] == true)
			{
				LR& r2 = labelData[l2];
				r1.num_neighbors++;
				r2.num_neighbors++;
			}
		}
	}
	#endif

	//==== Collision detection, method 2
	int m = 10000;
	std::vector <std::vector<int>> HT (m);	// hash table
	for (auto l : uniqueLabels)
	{
		LR& r = labelData [l];
		auto h1 = spat_hash_2d (r.aabb.get_xmin(), r.aabb.get_ymin(), m),
			h2 = spat_hash_2d (r.aabb.get_xmin(), r.aabb.get_ymax(), m),
			h3 = spat_hash_2d (r.aabb.get_xmax(), r.aabb.get_ymin(), m),
			h4 = spat_hash_2d (r.aabb.get_xmax(), r.aabb.get_ymax(), m);
		HT[h1].push_back(l);
		HT[h2].push_back(l);
		HT[h3].push_back(l);
		HT[h4].push_back(l);
	}

	// Broad phase of collision detection
	for (auto& bin : HT)
	{
		// No need to bother about not colliding labels
		if (bin.size() <= 1)
			continue;

		// Perform the N^2 check
		for (auto& l1 : bin)
		{
			LR& r1 = labelData[l1];

			for (auto& l2 : bin)
			{
				if (l1 < l2)	// Lower triangle 
				{
					LR & r2 = labelData[l2];
					bool overlap = ! aabbNoOverlap (r1, r2, radius);
					if (overlap)
					{
						r1.num_neighbors++;
						r2.num_neighbors++;
					}
				}
			}
		}
	}
}

void LR::init_aabb (StatsInt x, StatsInt y)
{
	aabb.init_x(x); 
	aabb.init_y(y);
	num_neighbors = 0;
}

void LR::update_aabb (StatsInt x, StatsInt y)
{
	aabb.update_x(x);
	aabb.update_y(y);
}


