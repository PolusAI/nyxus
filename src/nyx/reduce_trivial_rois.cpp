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
#include "features/hexagonality_polygonality.h"
#include "features/ngtdm.h"
#include "features/image_moments.h"
#include "features/intensity.h"
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
		//==== Parallel execution parameters 
		int n_reduce_threads = theEnvironment.n_reduce_threads;		
		size_t jobSize = PendingRoisLabels.size(),
			workPerThread = jobSize / n_reduce_threads;

		//==== Pixel intensity stats
		if (PixelIntensityFeatures::required(theFeatureSet))
		{
			STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
			runParallel(PixelIntensityFeatures::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Basic morphology
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
		if (ContourFeature::required(theFeatureSet) 
			|| ConvexHullFeature::required(theFeatureSet)
			|| FractalDimensionFeature::required(theFeatureSet) 
			|| GeodeticLengthThicknessFeature::required(theFeatureSet)
			|| NeighborsFeature::required(theFeatureSet)
			|| RoiRadiusFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			runParallel(ContourFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Convex hull related solidity, circularity
		if (ConvexHullFeature::required(theFeatureSet) 
			|| CaliperFeretFeature::required(theFeatureSet) 
			|| CaliperMartinFeature::required(theFeatureSet) 
			|| CaliperNassensteinFeature::required(theFeatureSet) 
			|| HexagonalityPolygonalityFeature::required(theFeatureSet))
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

		//==== Feret diameters and angles
		if (CaliperFeretFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
			runParallel(CaliperFeretFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Martin diameters
		if (CaliperMartinFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
			runParallel(CaliperMartinFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

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

		//==== Geodetic length and thickness
		if (GeodeticLengthThicknessFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(GeodeticLengthThicknessFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== ROI radius
		if (RoiRadiusFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
			runParallel(RoiRadiusFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Erosion pixels
		if (ErosionPixelsFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
			runParallel(ErosionPixelsFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Fractal dimension
		if (FractalDimensionFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
			runParallel(FractalDimensionFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLCM aka Haralick 2D 
		if (GLCMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLCM/GLCM/#bbbbbb", "\t=");
			runParallel(GLCMFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLRLM
		if (GLRLMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
			runParallel(GLRLMFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLSZM
		if (GLSZMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
			runParallel(GLSZMFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== GLDM
		if (GLDMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
			runParallel(GLDMFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== NGTDM
		if (NGTDMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
			runParallel(NGTDMFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Moments
		if (ImageMomentsFeature::required(theFeatureSet))
		{
			#ifndef USE_GPU
				STOPWATCH("Moments/Moments/2D moms/#FFFACD", "\t=");
				runParallel(ImageMomentsFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
			#else
				// Did the user opted out from using GPU?
				if (theEnvironment.using_gpu() == false)
				{				
					// Route calculation via the regular CPU-multithreaded way
					STOPWATCH("Moments/Moments/2D moms/#FFFACD", "\t=");
					runParallel(ImageMomentsFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
				}
				else
				{
					// Calculate the feature via GPU
					STOPWATCH("GPU-Moments/GPU-Moments/2D moms/#FFFACD", "\t=");
					ImageMomentsFeature::gpu_process_all_rois(PendingRoisLabels, roiData);
				}
			#endif
		}

		//==== Gabor features
		if (GaborFeature::required(theFeatureSet))
		{
			#ifndef USE_GPU

				STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
				runParallel(GaborFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
				
			#else 
				
				if (theEnvironment.using_gpu() == false) {

					STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
					runParallel(GaborFeature::reduce, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);

				} else {

					STOPWATCH("GPU-Gabor/GPU-Gabor/Gabor/#f58231", "\t=");
					GaborFeature::gpu_process_all_rois(PendingRoisLabels, roiData);

				}
				
			#endif
		}

		//==== Radial distribution / Zernike 2D 
		if (ZernikeFeature::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
			runParallel(ZernikeFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (RadialDistributionFeature::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
			runParallel(RadialDistributionFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &PendingRoisLabels, &roiData);
		}
	}

	void reduce_neighbors_and_dependencies_manual ()
	{
		// A (ll) L (abels)
		std::vector <int> AL;
		AL.reserve (uniqueLabels.size());
		AL.insert (AL.end(), uniqueLabels.begin(), uniqueLabels.end());

		//==== Parallel execution parameters 
		int n_reduce_threads = theEnvironment.n_reduce_threads;		
		size_t jobSize = AL.size(),
			workPerThread = jobSize / n_reduce_threads;

		//==== Neighbors
		if (NeighborsFeature::required(theFeatureSet) || HexagonalityPolygonalityFeature::required(theFeatureSet) || EnclosingInscribingCircumscribingCircleFeature::required(theFeatureSet))
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			NeighborsFeature::manual_reduce();
		}

		//==== Hexagonality and polygonality
		if (HexagonalityPolygonalityFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(HexagonalityPolygonalityFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &AL, &roiData);
		}

		//==== Enclosing, inscribing, and circumscribing circle
		if (EnclosingInscribingCircumscribingCircleFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(EnclosingInscribingCircumscribingCircleFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &AL, &roiData);
		}
	}

}

