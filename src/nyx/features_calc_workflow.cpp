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
#include "features/erosion.h"
#include "features/radial_distribution.h"
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
#include "features/neighbors.h"
#include "features/particle_metrics.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/timing.h"
#include "parallel.h"

namespace Nyxus
{
	// Label buffer size related constants
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
	void init_label_record(LR& r, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity)
	{
		r.segFname = segFile;
		r.intFname = intFile;

		// Allocate the value matrix
		for (int i = 0; i < AvailableFeatures::_COUNT_; i++)
		{
			std::vector<StatsReal> row{ 0.0 };	// One value initialy. More values can be added for Haralick and Zernike type methods
			r.fvals.push_back(row);
		}

		r.label = label;

		// Save the pixel
		r.raw_pixels.push_back(Pixel2(x, y, intensity));

		r.fvals[AREA_PIXELS_COUNT][0] = 1;
		if (theEnvironment.xyRes == 0.0)
			r.fvals[AREA_UM2][0] = 0;
		else
			r.fvals[AREA_UM2][0] = std::pow(theEnvironment.pixelSizeUm, 2);

		r.aux_PrevCount = 0;

		// Min
		r.fvals[MIN][0] = r.aux_min = intensity;
		// Max
		r.fvals[MAX][0] = r.aux_max = intensity;
		// Moments
		r.fvals[MEAN][0] = intensity;
		r.aux_M2 = 0;
		r.aux_M3 = 0;
		r.aux_M4 = 0;
		r.fvals[ENERGY][0] = intensity * intensity;
		r.aux_variance = 0.0;
		r.fvals[MEAN_ABSOLUTE_DEVIATION][0] = 0.0;
		r.aux_PrevIntens = intensity; // Previous intensity
		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		r.fvals[CENTROID_X][0] = StatsReal(x) + 1;
		r.fvals[CENTROID_Y][0] = StatsReal(y) + 1;

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

		r.init_aabb(x, y);
	}

	// This function 'digests' the 2nd and the following pixel of a label and updates the label's feature calculation state - the instance of structure 'LR'
	void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity)
	{
		// Save the pixel
		lr.raw_pixels.push_back(Pixel2(x, y, intensity));

		// Update ROI intensity range
		lr.aux_min = std::min(lr.aux_min, intensity);
		lr.aux_max = std::max(lr.aux_max, intensity);

		// Update ROI bounds
		lr.update_aabb(x, y);
	}

	// The root function of handling a pixel being scanned
	void update_label(int x, int y, int label, PixIntens intensity)
	{
		auto it = uniqueLabels.find(label);
		if (it == uniqueLabels.end())
		{
			// Remember this label
			uniqueLabels.insert(label);

			// Initialize the label record
			LR lr;
			init_label_record(lr, theSegFname, theIntFname, x, y, label, intensity);
			labelData[label] = lr;
		}
		else
		{
			// Update label's stats
			LR& lr = labelData[label];
			update_label_record(lr, x, y, label, intensity);
		}
	}

	// This function should be called once after a file pair processing is finished.
	void reduce(int nThr, int min_online_roi_size)
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

		//=== Copy ROI labels to a vector to make them indexable 
		std::vector<int> roiLabelsVector;
		for (auto l : uniqueLabels)
			roiLabelsVector.push_back(l);

		//==== 	Parallel execution parameters 
		size_t tileSize = roiLabelsVector.size(),
			workPerThread = tileSize / nThr;

		//==== Pixel intensity stats. Calculate these basic features unconditionally
		{
			STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
			runParallel(parallelReduceIntensityStats, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Neighbors
		if (theFeatureSet.anyEnabled({ NUM_NEIGHBORS, CLOSEST_NEIGHBOR1_DIST, CLOSEST_NEIGHBOR2_DIST }))
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			NeighborFeatures::reduce(theEnvironment.pixelDistance);
		}

		//==== Fitting an ellipse
		if (theFeatureSet.anyEnabled({ MAJOR_AXIS_LENGTH, MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION, ROUNDNESS }))
		{
			STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
			runParallel(EllipseFittingFeatures::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Contour-related ROI perimeter, equivalent circle diameter
		if (theFeatureSet.anyEnabled({
				PERIMETER,
				EQUIVALENT_DIAMETER,
				EDGE_INTEGRATEDINTENSITY,
				EDGE_MAXINTENSITY,
				EDGE_MININTENSITY,
				EDGE_MEANINTENSITY,
				EDGE_STDDEVINTENSITY,
				// dependencies:
				CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY
			}))
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			runParallel(parallelReduceContour, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Convex hull related solidity, circularity
		if (theFeatureSet.anyEnabled({ CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY }))
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
			STOPWATCH("Morphology/Hull/H/#4aaaea", "\t=");
			runParallel(parallelReduceConvHull, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(ExtremaFeatures::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Euler 
		if (theFeatureSet.isEnabled(EULER_NUMBER))
		{
			STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
			runParallel(EulerNumber::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Feret diameters and angles
		if (theFeatureSet.anyEnabled({ MIN_FERET_DIAMETER, MAX_FERET_DIAMETER, MIN_FERET_ANGLE, MAX_FERET_ANGLE }) ||
			theFeatureSet.anyEnabled({
				STAT_FERET_DIAM_MIN,
				STAT_FERET_DIAM_MAX,
				STAT_FERET_DIAM_MEAN,
				STAT_FERET_DIAM_MEDIAN,
				STAT_FERET_DIAM_STDDEV,
				STAT_FERET_DIAM_MODE }))
		{
			STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
			runParallel(ParticleMetrics::reduce_feret, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(ParticleMetrics::reduce_martin, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(ParticleMetrics::reduce_nassenstein, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(Chords::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Hexagonality and polygonality
		if (theFeatureSet.anyEnabled({ POLYGONALITY_AVE, HEXAGONALITY_AVE, HEXAGONALITY_STDDEV }))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(Hexagonality_and_Polygonality::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Enclosing, inscribing, and circumscribing circle
		if (theFeatureSet.anyEnabled({ DIAMETER_MIN_ENCLOSING_CIRCLE, DIAMETER_INSCRIBING_CIRCLE, DIAMETER_CIRCUMSCRIBING_CIRCLE }))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(EnclosingInscribingCircumscribingCircle::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Geodetic length and thickness
		if (theFeatureSet.anyEnabled({ GEODETIC_LENGTH, THICKNESS }))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(GeodeticLength_and_Thickness::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== ROI radius
		if (theFeatureSet.anyEnabled({
			ROI_RADIUS_MEAN,
			ROI_RADIUS_MAX,
			ROI_RADIUS_MEDIAN
			}))
		{
			STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
			runParallel(RoiRadius::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Erosion pixels
		if (theFeatureSet.anyEnabled({ EROSIONS_2_VANISH, EROSIONS_2_VANISH_COMPLEMENT }))
		{
			STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
			runParallel(ErosionPixels::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Fractal dimension
		if (theFeatureSet.anyEnabled({ FRACT_DIM_BOXCOUNT, FRACT_DIM_PERIMETER }))
		{
			STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
			runParallel(FractalDimension::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(GLCM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(GLRLM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(GLSZM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(GLDM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(NGTDM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
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
			runParallel(HuMoments::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Gabor features
		if (theFeatureSet.isEnabled(GABOR))
		{
			STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
			runParallel(GaborFeatures::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Radial distribution / Zernike 2D 
		if (theFeatureSet.isEnabled(ZERNIKE2D))
		{
			STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
			runParallel(parallelReduceZernike2D, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (theFeatureSet.anyEnabled({ FRAC_AT_D, MEAN_FRAC, RADIAL_CV }))
		{
			STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
			runParallel(RadialDistribution::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

	}
}