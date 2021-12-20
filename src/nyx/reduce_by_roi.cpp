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
#include "features/image_moments.h"
#include "features/moments.h"
#include "features/neighbors.h"
#include "features/particle_metrics.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/timing.h"
#include "parallel.h"

namespace Nyxus
{
	// This function should be called once after scanning ROI pixels is finished
	void reduce_by_roi (int nThr, int min_online_roi_size)
	{
		//=== Copy ROI labels to a vector to make them indexable 
		std::vector<int> roiLabelsVector;
		for (auto l : uniqueLabels)
			roiLabelsVector.push_back(l);

		//==== 	Parallel execution parameters 
		size_t jobSize = roiLabelsVector.size(),
			workPerThread = jobSize / nThr;

		for (auto& ld : labelData) 
		{
			auto l = ld.first;		// ROI label code
			auto& r = ld.second;	// ROI info cache structure


			//==== Pixel intensity stats. These basic features are calculated unconditionally
			{
				STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
				calcRoiIntensityFeatures (r);  // runParallel(parallelReduceIntensityStats, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
			}

			//==== Fitting an ellipse
			if (EllipseFittingFeatures::required(theFeatureSet))
			{
				STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
				EllipseFittingFeatures f (r.raw_pixels, r.fvals[CENTROID_X][0], r.fvals[CENTROID_Y][0], r.fvals[AREA_PIXELS_COUNT][0]);
				r.fvals[MAJOR_AXIS_LENGTH][0] = f.get_major_axis_length();
				r.fvals[MINOR_AXIS_LENGTH][0] = f.get_minor_axis_length();
				r.fvals[ECCENTRICITY][0] = f.get_eccentricity();
				r.fvals[ORIENTATION][0] = f.get_orientation();
				r.fvals[ROUNDNESS][0] = f.get_roundness();
			}

			//==== Contour-related ROI perimeter, equivalent circle diameter
			if (Contour::required(theFeatureSet))
			{
				STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
				calcRoiContour(r); // runParallel(parallelReduceContour, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
			}

			//==== Convex hull related solidity, circularity
			if (ConvexHull::required(theFeatureSet))
			{
				// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
				STOPWATCH("Morphology/Hull/H/#4aaaea", "\t=");
				
				//==== Convex hull and solidity
				r.convHull.calculate(r.raw_pixels);
				r.fvals[CONVEX_HULL_AREA][0] = r.convHull.getArea();
				r.fvals[SOLIDITY][0] = r.raw_pixels.size() / r.fvals[CONVEX_HULL_AREA][0];

				//==== Circularity
				r.fvals[CIRCULARITY][0] = 4.0 * M_PI * r.raw_pixels.size() / (r.fvals[PERIMETER][0] * r.fvals[PERIMETER][0]);
			}

			//==== Extrema 
			if (ExtremaFeatures::required(theFeatureSet))
			{
				STOPWATCH("Morphology/Extrema/Ex/#4aaaea", "\t=");
				ExtremaFeatures ef(r.raw_pixels);
				auto [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8] = ef.get_values();
				r.fvals[EXTREMA_P1_Y][0] = y1;
				r.fvals[EXTREMA_P1_X][0] = x1;
				r.fvals[EXTREMA_P2_Y][0] = y2;
				r.fvals[EXTREMA_P2_X][0] = x2;
				r.fvals[EXTREMA_P3_Y][0] = y3;
				r.fvals[EXTREMA_P3_X][0] = x3;
				r.fvals[EXTREMA_P4_Y][0] = y4;
				r.fvals[EXTREMA_P4_X][0] = x4;
				r.fvals[EXTREMA_P5_Y][0] = y5;
				r.fvals[EXTREMA_P5_X][0] = x5;
				r.fvals[EXTREMA_P6_Y][0] = y6;
				r.fvals[EXTREMA_P6_X][0] = x6;
				r.fvals[EXTREMA_P7_Y][0] = y7;
				r.fvals[EXTREMA_P7_X][0] = x7;
				r.fvals[EXTREMA_P8_Y][0] = y8;
				r.fvals[EXTREMA_P8_X][0] = x8;
			}

			//==== Euler 
			if (EulerNumber::required(theFeatureSet))
			{
				STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
				EulerNumber eu(r.raw_pixels, r.aabb);
				r.fvals[EULER_NUMBER][0] = eu.get_feature_value();
			}

			//==== Feret diameters and angles
			if (ParticleMetrics::feret_required(theFeatureSet))
			{
				STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
				ParticleMetrics pm(r.convHull.CH);
				std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
				pm.calc_ferret(
					r.fvals[MAX_FERET_DIAMETER][0],
					r.fvals[MAX_FERET_ANGLE][0],
					r.fvals[MIN_FERET_DIAMETER][0],
					r.fvals[MIN_FERET_ANGLE][0],
					allD
				);
				auto structStat = ComputeCommonStatistics2(allD);
				r.fvals[STAT_FERET_DIAM_MIN][0] = (double)structStat.min;
				r.fvals[STAT_FERET_DIAM_MAX][0] = (double)structStat.max;
				r.fvals[STAT_FERET_DIAM_MEAN][0] = structStat.mean;
				r.fvals[STAT_FERET_DIAM_MEDIAN][0] = structStat.median;
				r.fvals[STAT_FERET_DIAM_STDDEV][0] = structStat.stdev;
				r.fvals[STAT_FERET_DIAM_MODE][0] = (double)structStat.mode;
			}

			//==== Martin diameters
			if (ParticleMetrics::martin_required(theFeatureSet))
			{
				STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
				//runParallel(ParticleMetrics::reduce_martin, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
				ParticleMetrics pm(r.convHull.CH);
				std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
				pm.calc_martin(allD);
				auto structStat = ComputeCommonStatistics2(allD);
				r.fvals[STAT_MARTIN_DIAM_MIN][0] = (double)structStat.min;
				r.fvals[STAT_MARTIN_DIAM_MAX][0] = (double)structStat.max;
				r.fvals[STAT_MARTIN_DIAM_MEAN][0] = structStat.mean;
				r.fvals[STAT_MARTIN_DIAM_MEDIAN][0] = structStat.median;
				r.fvals[STAT_MARTIN_DIAM_STDDEV][0] = structStat.stdev;
				r.fvals[STAT_MARTIN_DIAM_MODE][0] = (double)structStat.mode;
			}

			//==== Nassenstein diameters
			if (ParticleMetrics::nassenstein_required(theFeatureSet))
			{
				STOPWATCH("Morphology/Nassenstein/N/#4aaaea", "\t=");
				//runParallel(ParticleMetrics::reduce_nassenstein, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
				ParticleMetrics pm(r.convHull.CH);
				std::vector<double> allD;	// all the diameters at 0-180 degrees rotation
				pm.calc_nassenstein(allD);
				auto s = ComputeCommonStatistics2(allD);
				r.fvals[STAT_NASSENSTEIN_DIAM_MIN][0] = (double)s.min;
				r.fvals[STAT_NASSENSTEIN_DIAM_MAX][0] = (double)s.max;
				r.fvals[STAT_NASSENSTEIN_DIAM_MEAN][0] = s.mean;
				r.fvals[STAT_NASSENSTEIN_DIAM_MEDIAN][0] = s.median;
				r.fvals[STAT_NASSENSTEIN_DIAM_STDDEV][0] = s.stdev;
				r.fvals[STAT_NASSENSTEIN_DIAM_MODE][0] = (double)s.mode;
			}

			//==== Chords
			if (Chords::required(theFeatureSet))
			{
				STOPWATCH("Morphology/Chords/Ch/#4aaaea", "\t=");
				//runParallel(Chords::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
				double cenx = r.fvals[CENTROID_X][0],
					ceny = r.fvals[CENTROID_Y][0];
				Chords cho(r.raw_pixels, r.aabb, cenx, ceny);

				double
					_max = 0,
					_min = 0,
					_median = 0,
					_mean = 0,
					_mode = 0,
					_stddev = 0,
					_min_angle = 0,
					_max_angle = 0;

				std::tie(_max, _min, _median, _mean, _mode, _stddev, _min_angle, _max_angle) = cho.get_maxchords_stats();
				r.fvals[MAXCHORDS_MAX][0] = _max;
				r.fvals[MAXCHORDS_MAX_ANG][0] = _max_angle;
				r.fvals[MAXCHORDS_MIN][0] = _min;
				r.fvals[MAXCHORDS_MIN_ANG][0] = _min_angle;
				r.fvals[MAXCHORDS_MEDIAN][0] = _median;
				r.fvals[MAXCHORDS_MEAN][0] = _mean;
				r.fvals[MAXCHORDS_MODE][0] = _mode;
				r.fvals[MAXCHORDS_STDDEV][0] = _stddev;

				std::tie(_max, _min, _median, _mean, _mode, _stddev, _min_angle, _max_angle) = cho.get_allchords_stats();
				r.fvals[ALLCHORDS_MAX][0] = _max;
				r.fvals[ALLCHORDS_MAX_ANG][0] = _max_angle;
				r.fvals[ALLCHORDS_MIN][0] = _min;
				r.fvals[ALLCHORDS_MIN_ANG][0] = _min_angle;
				r.fvals[ALLCHORDS_MEDIAN][0] = _median;
				r.fvals[ALLCHORDS_MEAN][0] = _mean;
				r.fvals[ALLCHORDS_MODE][0] = _mode;
				r.fvals[ALLCHORDS_STDDEV][0] = _stddev;
			}

			//==== Hexagonality and polygonality
			if (Hexagonality_and_Polygonality::required(theFeatureSet)) 
			{
				STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
				
				// Skip if the contour, convex hull, and neighbors are unavailable, otherwise the related features will be == NAN. Those feature will be equal to the default unassigned value.
				if (r.contour.contour_pixels.size() == 0 || r.convHull.CH.size() == 0 || r.fvals[NUM_NEIGHBORS][0] == 0)
					continue;

				Hexagonality_and_Polygonality hp;
				auto [polyAve, hexAve, hexSd] = hp.calculate(r.fvals[NUM_NEIGHBORS][0], r.raw_pixels.size(), r.fvals[PERIMETER][0], r.fvals[CONVEX_HULL_AREA][0], r.fvals[MIN_FERET_DIAMETER][0], r.fvals[MAX_FERET_DIAMETER][0]);
				r.fvals[POLYGONALITY_AVE][0] = polyAve;
				r.fvals[HEXAGONALITY_AVE][0] = hexAve;
				r.fvals[HEXAGONALITY_STDDEV][0] = hexSd;
			}

			//==== Enclosing, inscribing, and circumscribing circle
			if (EnclosingInscribingCircumscribingCircle::required(theFeatureSet))
			{
				STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
						
				// Skip if the contour, convex hull, and neighbors are unavailable, otherwise the related features will be == NAN. Those feature will be equal to the default unassigned value.
				if (r.contour.contour_pixels.size() == 0 || r.convHull.CH.size() == 0 || r.fvals[NUM_NEIGHBORS][0] == 0)
					continue;

				EnclosingInscribingCircumscribingCircle cir;
				r.fvals[DIAMETER_MIN_ENCLOSING_CIRCLE][0] = cir.calculate_min_enclosing_circle_diam(r.contour.contour_pixels);

				auto [diamIns, diamCir] = cir.calculate_inscribing_circumscribing_circle(r.contour.contour_pixels, r.fvals[CENTROID_X][0], r.fvals[CENTROID_Y][0]);
				r.fvals[DIAMETER_INSCRIBING_CIRCLE][0] = diamIns;
				r.fvals[DIAMETER_CIRCUMSCRIBING_CIRCLE][0] = diamCir;
			}

			//==== Geodetic length and thickness
			if (GeodeticLength_and_Thickness::required(theFeatureSet)) 
			{
				STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
				//runParallel(GeodeticLength_and_Thickness::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);

				// Skip if the contour, convex hull, and neighbors are unavailable, otherwise the related features will be == NAN. Those feature will be equal to the default unassigned value.
				if (r.contour.contour_pixels.size() == 0 || r.convHull.CH.size() == 0 || r.fvals[NUM_NEIGHBORS][0] == 0)
					continue;

				GeodeticLength_and_Thickness glt(r.raw_pixels.size(), (StatsInt)r.fvals[PERIMETER][0]);
				r.fvals[GEODETIC_LENGTH][0] = glt.get_geodetic_length();
				r.fvals[THICKNESS][0] = glt.get_thickness();
			}

			//==== ROI radius
			if (RoiRadius::required (theFeatureSet))
			{
				STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");

				// Prepare the contour if necessary
				if (r.contour.contour_pixels.size() == 0)
					r.contour.calculate(r.aux_image_matrix);

				RoiRadius roir(r.raw_pixels, r.contour.contour_pixels);
				r.fvals[ROI_RADIUS_MEAN][0] = roir.get_mean_radius();
				r.fvals[ROI_RADIUS_MAX][0] = roir.get_max_radius();
				r.fvals[ROI_RADIUS_MEDIAN][0] = roir.get_median_radius();
			}

			//==== Erosion pixels
			if (ErosionPixels::required(theFeatureSet))
			{
				STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
				
				// Check if data is good
				if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
					continue;

				// Calculate feature
				ErosionPixels epix(r.aux_image_matrix);
				r.fvals[EROSIONS_2_VANISH][0] = epix.get_feature_value();
			}

			//==== Fractal dimension
			if (FractalDimension::required (theFeatureSet))
			{
				STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");

				// Skip calculation in case of bad data
				if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
					continue;

				// Calculate feature
				FractalDimension fd(r.raw_pixels, r.aabb);
				r.fvals[FRACT_DIM_BOXCOUNT][0] = fd.get_box_count_fd();
				r.fvals[FRACT_DIM_PERIMETER][0] = fd.get_perimeter_fd();
			}

			//==== GLCM aka Haralick 2D 
			if (GLCM_features::required(theFeatureSet))
			{
				STOPWATCH("Texture/GLCM texture/GLCM/#bbbbbb", "\t=");
				//runParallel(GLCM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
						
				//=== GLCM version 2
				// Skip calculation in case of bad data
				int minI = (int)r.fvals[MIN][0],
					maxI = (int)r.fvals[MAX][0];
				if (minI == maxI)
				{
					// Dfault values for all 4 standard angles
					r.fvals[GLCM_ANGULAR2NDMOMENT].resize(4, 0);
					r.fvals[GLCM_CONTRAST].resize(4, 0);
					r.fvals[GLCM_CORRELATION].resize(4, 0);
					r.fvals[GLCM_VARIANCE].resize(4, 0);
					r.fvals[GLCM_INVERSEDIFFERENCEMOMENT].resize(4, 0);
					r.fvals[GLCM_SUMAVERAGE].resize(4, 0);
					r.fvals[GLCM_SUMVARIANCE].resize(4, 0);
					r.fvals[GLCM_SUMENTROPY].resize(4, 0);
					r.fvals[GLCM_ENTROPY].resize(4, 0);
					r.fvals[GLCM_DIFFERENCEVARIANCE].resize(4, 0);
					r.fvals[GLCM_DIFFERENCEENTROPY].resize(4, 0);
					r.fvals[GLCM_INFOMEAS1].resize(4, 0);
					r.fvals[GLCM_INFOMEAS2].resize(4, 0);
					continue;
				}

				//---	ImageMatrix im(r.raw_pixels, r.aabb);
				GLCM_features f(minI, maxI, r.aux_image_matrix, 5);
				f.get_AngularSecondMoments(r.fvals[GLCM_ANGULAR2NDMOMENT]);
				f.get_Contrast(r.fvals[GLCM_CONTRAST]);
				f.get_Correlation(r.fvals[GLCM_CORRELATION]);
				f.get_Variance(r.fvals[GLCM_VARIANCE]);
				f.get_InverseDifferenceMoment(r.fvals[GLCM_INVERSEDIFFERENCEMOMENT]);
				f.get_SumAverage(r.fvals[GLCM_SUMAVERAGE]);
				f.get_SumVariance(r.fvals[GLCM_SUMVARIANCE]);
				f.get_SumEntropy(r.fvals[GLCM_SUMENTROPY]);
				f.get_Entropy(r.fvals[GLCM_ENTROPY]);
				f.get_DifferenceVariance(r.fvals[GLCM_DIFFERENCEVARIANCE]);
				f.get_DifferenceEntropy(r.fvals[GLCM_DIFFERENCEENTROPY]);
				f.get_InfoMeas1(r.fvals[GLCM_INFOMEAS1]);
				f.get_InfoMeas2(r.fvals[GLCM_INFOMEAS2]);
			}		

			//==== GLRLM
			if (GLRLM_features::required (theFeatureSet))
			{
				STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
				//runParallel(GLRLM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
				GLRLM_features glrlm((int)r.fvals[MIN][0], (int)r.fvals[MAX][0], r.aux_image_matrix);
				glrlm.calc_SRE(r.fvals[GLRLM_SRE]);
				glrlm.calc_LRE(r.fvals[GLRLM_LRE]);
				glrlm.calc_GLN(r.fvals[GLRLM_GLN]);
				glrlm.calc_GLNN(r.fvals[GLRLM_GLNN]);
				glrlm.calc_RLN(r.fvals[GLRLM_RLN]);
				glrlm.calc_RLNN(r.fvals[GLRLM_RLNN]);
				glrlm.calc_RP(r.fvals[GLRLM_RP]);
				glrlm.calc_GLV(r.fvals[GLRLM_GLV]);
				glrlm.calc_RV(r.fvals[GLRLM_RV]);
				glrlm.calc_RE(r.fvals[GLRLM_RE]);
				glrlm.calc_LGLRE(r.fvals[GLRLM_LGLRE]);
				glrlm.calc_HGLRE(r.fvals[GLRLM_HGLRE]);
				glrlm.calc_SRLGLE(r.fvals[GLRLM_SRLGLE]);
				glrlm.calc_SRHGLE(r.fvals[GLRLM_SRHGLE]);
				glrlm.calc_LRLGLE(r.fvals[GLRLM_LRLGLE]);
				glrlm.calc_LRHGLE(r.fvals[GLRLM_LRHGLE]);
			}		
			
			//==== GLSZM
			if (GLSZM_features::required (theFeatureSet))
			{
				STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
				GLSZM_features glszm((int)r.fvals[MIN][0], (int)r.fvals[MAX][0], r.aux_image_matrix);
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

			//==== GLDM
			if (GLDM_features::required (theFeatureSet))
			{
				STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
				GLDM_features gldm((int)r.fvals[MIN][0], (int)r.fvals[MAX][0], r.aux_image_matrix);
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

			//==== NGTDM
			if (NGTDM_features::required(theFeatureSet))
			{
				STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
				//runParallel(NGTDM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
				NGTDM_features ngtdm((int)r.fvals[MIN][0], (int)r.fvals[MAX][0], r.aux_image_matrix);
				r.fvals[NGTDM_COARSENESS][0] = ngtdm.calc_Coarseness();
				r.fvals[NGTDM_CONTRAST][0] = ngtdm.calc_Contrast();
				r.fvals[NGTDM_BUSYNESS][0] = ngtdm.calc_Busyness();
				r.fvals[NGTDM_COMPLEXITY][0] = ngtdm.calc_Complexity();
				r.fvals[NGTDM_STRENGTH][0] = ngtdm.calc_Strength();
			}

			//==== Moments
			if (ImageMoments::required(theFeatureSet))
			{
				STOPWATCH("Moments/Moments/2D moms/#FFFACD", "\t=");

						// Prepare the contour if necessary
				if (r.contour.contour_pixels.size() == 0)
					r.contour.calculate(r.aux_image_matrix);

				ImageMatrix weighted_im(r.raw_pixels, r.aabb);
				weighted_im.apply_distance_to_contour_weights(r.raw_pixels, r.contour.contour_pixels);
				ImageMoments immo((int)r.fvals[MIN][0], (int)r.fvals[MAX][0], r.aux_image_matrix, weighted_im);

				double m1, m2, m3, m4, m5, m6, m7, m8, m9, m10;
				std::tie(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10) = immo.getSpatialMoments();
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

				std::tie(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10) = immo.getWeightedSpatialMoments();
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

				std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getCentralMoments();
				r.fvals[CENTRAL_MOMENT_02][0] = m1;
				r.fvals[CENTRAL_MOMENT_03][0] = m2;
				r.fvals[CENTRAL_MOMENT_11][0] = m3;
				r.fvals[CENTRAL_MOMENT_12][0] = m4;
				r.fvals[CENTRAL_MOMENT_20][0] = m5;
				r.fvals[CENTRAL_MOMENT_21][0] = m6;
				r.fvals[CENTRAL_MOMENT_30][0] = m7;

				std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getWeightedCentralMoments();
				r.fvals[WEIGHTED_CENTRAL_MOMENT_02][0] = m1;
				r.fvals[WEIGHTED_CENTRAL_MOMENT_03][0] = m2;
				r.fvals[WEIGHTED_CENTRAL_MOMENT_11][0] = m3;
				r.fvals[WEIGHTED_CENTRAL_MOMENT_12][0] = m4;
				r.fvals[WEIGHTED_CENTRAL_MOMENT_20][0] = m5;
				r.fvals[WEIGHTED_CENTRAL_MOMENT_21][0] = m6;
				r.fvals[WEIGHTED_CENTRAL_MOMENT_30][0] = m7;

				std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getNormCentralMoments();
				r.fvals[NORM_CENTRAL_MOMENT_02][0] = m1;
				r.fvals[NORM_CENTRAL_MOMENT_03][0] = m2;
				r.fvals[NORM_CENTRAL_MOMENT_11][0] = m3;
				r.fvals[NORM_CENTRAL_MOMENT_12][0] = m4;
				r.fvals[NORM_CENTRAL_MOMENT_20][0] = m5;
				r.fvals[NORM_CENTRAL_MOMENT_21][0] = m6;
				r.fvals[NORM_CENTRAL_MOMENT_30][0] = m7;

				std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getNormSpatialMoments();
				r.fvals[NORM_SPAT_MOMENT_00][0] = m1;
				r.fvals[NORM_SPAT_MOMENT_01][0] = m2;
				r.fvals[NORM_SPAT_MOMENT_02][0] = m3;
				r.fvals[NORM_SPAT_MOMENT_03][0] = m4;
				r.fvals[NORM_SPAT_MOMENT_10][0] = m5;
				r.fvals[NORM_SPAT_MOMENT_20][0] = m6;
				r.fvals[NORM_SPAT_MOMENT_30][0] = m7;

				std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getHuMoments();
				r.fvals[HU_M1][0] = m1;
				r.fvals[HU_M2][0] = m2;
				r.fvals[HU_M3][0] = m3;
				r.fvals[HU_M4][0] = m4;
				r.fvals[HU_M5][0] = m5;
				r.fvals[HU_M6][0] = m6;
				r.fvals[HU_M7][0] = m7;

				std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getWeightedHuMoments();
				r.fvals[WEIGHTED_HU_M1][0] = m1;
				r.fvals[WEIGHTED_HU_M2][0] = m2;
				r.fvals[WEIGHTED_HU_M3][0] = m3;
				r.fvals[WEIGHTED_HU_M4][0] = m4;
				r.fvals[WEIGHTED_HU_M5][0] = m5;
				r.fvals[WEIGHTED_HU_M6][0] = m6;
				r.fvals[WEIGHTED_HU_M7][0] = m7;
			}

			//==== Gabor features
			if (GaborFeatures::required(theFeatureSet))
			{
				STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
				//runParallel(GaborFeatures::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		
				// Skip calculation in case of bad data
				if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
				{
					r.fvals[GABOR].resize(GaborFeatures::num_features, 0.0);
					continue;
				}

				GaborFeatures gf(r.aux_image_matrix);
				gf.get_feature_values(r.fvals[GABOR]);
			}

			//==== Radial distribution / Zernike 2D 
			if (ZernikeFeatures::required(theFeatureSet))
			{
				STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
				calcRoiZernike(r);
			}

			//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
			if (RadialDistribution::required(theFeatureSet))
			{
				STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");

				// Prepare the contour if necessary
				if (r.contour.contour_pixels.size() == 0)
					r.contour.calculate (r.aux_image_matrix);

				// Calculate the radial distribution
				RadialDistribution rd(r.raw_pixels, r.contour.contour_pixels);
				r.fvals[FRAC_AT_D] = rd.get_FracAtD();
				r.fvals[MEAN_FRAC] = rd.get_MeanFrac();
				r.fvals[RADIAL_CV] = rd.get_RadialCV();
			}
		}

		//==== Neighbors are reduced
		if (NeighborFeatures::required(theFeatureSet))
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			NeighborFeatures::reduce(theEnvironment.pixelDistance);
		}
	}
}

