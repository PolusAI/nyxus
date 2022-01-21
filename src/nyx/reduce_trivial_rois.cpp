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
#include "features/basic_morphology.h"
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
	// Calculating features in parallel with automatic feature order
	void reduce_trivial_rois (std::vector<int>& PendingRoisLabels)
	{
		int nrf = theFeatureMgr.get_num_requested_features();
		for (int i = 0; i < nrf; i++)
		{
			auto feature = theFeatureMgr.get_feature_method(i);
			feature->parallel_process (PendingRoisLabels, roiData, theEnvironment.n_reduce_threads);
		}
	}

	// Calculating features in parallel with hard-coded feature order. This function should be called once after a file pair processing is finished.
	void reduce_trivial_rois_manual (std::vector<int> & PendingRoisLabels)
	{
		//==== 	Parallel execution parameters 
		int n_reduce_threads = theEnvironment.n_reduce_threads;		
		size_t jobSize = PendingRoisLabels.size(),
			workPerThread = jobSize / n_reduce_threads;

		//==== Pixel intensity stats. Calculate these basic features unconditionally
		{
			STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
			runParallel (parallelReduceIntensityStats, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Fitting an ellipse
		if (BasicMorphologyFeatures::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Basic/E/#4aaaea", "\t=");
			runParallel(BasicMorphologyFeatures::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Fitting an ellipse
		if (EllipseFittingFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
			runParallel(EllipseFittingFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Contour-related ROI perimeter, equivalent circle diameter
		if (ContourFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			runParallel(parallelReduceContour, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Convex hull related solidity, circularity
		if (ConvexHullFeature::required(theFeatureSet))
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
			STOPWATCH("Morphology/Hull/H/#4aaaea", "\t=");
			runParallel(parallelReduceConvHull, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Extrema 
		if (ExtremaFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Extrema/Ex/#4aaaea", "\t=");
			runParallel(ExtremaFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Euler 
		if (EulerNumberFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
			runParallel(EulerNumberFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

#if 0 // Temporarily disabled 
		//==== Feret diameters and angles
		if (CaliperFeretFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
			runParallel(CaliperFeretFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Martin diameters
		if (CaliperMartinFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
			runParallel(CaliperMartinFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}
#endif

		//==== Nassenstein diameters
		if (CaliperNassensteinFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Nassenstein/N/#4aaaea", "\t=");
			runParallel(CaliperNassensteinFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Chords
		if (ChordsFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Chords/Ch/#4aaaea", "\t=");
			runParallel(ChordsFeature::process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Hexagonality and polygonality
		if (Hexagonality_and_Polygonality_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(Hexagonality_and_Polygonality_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Enclosing, inscribing, and circumscribing circle
		if (EnclosingInscribingCircumscribingCircle_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(EnclosingInscribingCircumscribingCircle_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Geodetic length and thickness
		if (GeodeticLength_and_Thickness_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(GeodeticLength_and_Thickness_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== ROI radius
		if (RoiRadius_features::required(theFeatureSet))
		{
			STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
			runParallel(RoiRadius_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Erosion pixels
		if (ErosionPixels_feature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
			runParallel(ErosionPixels_feature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Fractal dimension
		if (FractalDimension_feature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
			runParallel(FractalDimension_feature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLCM aka Haralick 2D 
		if (GLCM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLCM texture/GLCM/#bbbbbb", "\t=");
			runParallel(GLCM_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLRLM
		if (GLRLM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
			runParallel(GLRLM_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLSZM
		if (GLSZM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
			runParallel(GLSZM_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLDM
		if (GLDM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
			runParallel(GLDM_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== NGTDM
		if (NGTDM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
			runParallel(NGTDM_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Moments
		if (ImageMoments_features::required(theFeatureSet))
		{
			STOPWATCH("Moments/Moments/2D moms/#FFFACD", "\t=");
			runParallel(ImageMoments_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Gabor features
		if (GaborFeature::required(theFeatureSet))
		{
			STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
			runParallel(GaborFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Radial distribution / Zernike 2D 
		if (Zernike_features::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
			runParallel(parallelReduceZernike2D, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (RadialDistribution_features::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
			runParallel(RadialDistribution_features::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}
	}

	void reduce_neighbors()
	{
		if (Neighbor_features::required(theFeatureSet))
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			Neighbor_features::reduce(theEnvironment.get_pixel_distance());
		}
	}


}

