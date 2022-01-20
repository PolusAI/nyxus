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
#include "features/convex_hull.h"
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
#include "features/caliper.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/timing.h"
#include "parallel.h"

namespace Nyxus
{
	// This function should be called once after a file pair processing is finished.
	void reduce_by_feature (int nThr, int min_online_roi_size)
	{
		//=== Copy ROI labels to a vector to make them indexable 
		std::vector<int> roiLabelsVector;
		for (auto l : uniqueLabels)
			roiLabelsVector.push_back(l);

		//==== 	Parallel execution parameters 
		size_t jobSize = roiLabelsVector.size(),
			workPerThread = jobSize / nThr;

		//==== Pixel intensity stats. Calculate these basic features unconditionally
		{
			STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
			runParallel(parallelReduceIntensityStats, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Neighbors
		if (Neighbor_features::required(theFeatureSet)) 
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			Neighbor_features::reduce(theEnvironment.get_pixel_distance());
		}

		//==== Fitting an ellipse
		if (EllipseFittingFeature::required(theFeatureSet)) 
		{
			STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
			runParallel(EllipseFittingFeature::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Contour-related ROI perimeter, equivalent circle diameter
		if (ContourFeature::required(theFeatureSet)) 
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			runParallel(
				ContourFeature::ContourFeature::parallel_process_1_batch, // parallelReduceContour,
				nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Convex hull related solidity, circularity
		if (ConvexHullFeature::required(theFeatureSet))
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
			STOPWATCH("Morphology/Hull/H/#4aaaea", "\t=");
			runParallel(parallelReduceConvHull, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Extrema 
		if (ExtremaFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Extrema/Ex/#4aaaea", "\t=");
			runParallel(ExtremaFeature::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Euler 
		if (EulerNumberFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
			runParallel(EulerNumberFeature::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Feret diameters and angles
		if (ParticleMetrics_features::feret_required(theFeatureSet)) 
		{
			STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
			runParallel(ParticleMetrics_features::reduce_feret, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Martin diameters
		if (ParticleMetrics_features::martin_required(theFeatureSet))
		{
			STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
			runParallel(ParticleMetrics_features::reduce_martin, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Nassenstein diameters
		if (ParticleMetrics_features::nassenstein_required(theFeatureSet))
		{
			STOPWATCH("Morphology/Nassenstein/N/#4aaaea", "\t=");
			runParallel(ParticleMetrics_features::reduce_nassenstein, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Chords
		if (ChordsFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Chords/Ch/#4aaaea", "\t=");
			runParallel(ChordsFeature::process_1_batch, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Hexagonality and polygonality
		if (Hexagonality_and_Polygonality_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(Hexagonality_and_Polygonality_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Enclosing, inscribing, and circumscribing circle
		if (EnclosingInscribingCircumscribingCircle_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(EnclosingInscribingCircumscribingCircle_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Geodetic length and thickness
		if (GeodeticLength_and_Thickness_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(GeodeticLength_and_Thickness_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== ROI radius
		if (RoiRadius_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
			runParallel(RoiRadius_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Erosion pixels
		if (ErosionPixels_feature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
			runParallel(ErosionPixels_feature::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Fractal dimension
		if (FractalDimension_feature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
			runParallel(FractalDimension_feature::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== GLCM aka Haralick 2D 
		if (GLCM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLCM texture/GLCM/#bbbbbb", "\t=");
			runParallel(GLCM_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== GLRLM
		if (GLRLM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
			runParallel(GLRLM_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== GLSZM
		if (GLSZM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
			runParallel(GLSZM_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== GLDM
		if (GLDM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
			runParallel(GLDM_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== NGTDM
		if (NGTDM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
			runParallel(NGTDM_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Moments
		if (ImageMoments_features::required(theFeatureSet))
		{
			STOPWATCH("Moments/Moments/2D moms/#FFFACD", "\t=");
			runParallel(ImageMoments_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Gabor features
		if (GaborFeature::required(theFeatureSet))
		{
			STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
			runParallel(GaborFeature::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Radial distribution / Zernike 2D 
		if (Zernike_features::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
			runParallel(parallelReduceZernike2D, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (RadialDistribution_features::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
			runParallel(RadialDistribution_features::reduce, nThr, workPerThread, jobSize, &roiLabelsVector, &roiData);
		}

	}
}

