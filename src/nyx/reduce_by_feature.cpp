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
	// This function should be called once after a file pair processing is finished.
	void reduce_by_feature (int nThr, int min_online_roi_size)
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
		if (NeighborFeatures::required(theFeatureSet)) 
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			NeighborFeatures::reduce(theEnvironment.pixelDistance);
		}

		//==== Fitting an ellipse
		if (EllipseFittingFeatures::required(theFeatureSet)) 
		{
			STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
			runParallel(EllipseFittingFeatures::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Contour-related ROI perimeter, equivalent circle diameter
		if (Contour::required(theFeatureSet)) 
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			runParallel(parallelReduceContour, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Convex hull related solidity, circularity
		if (ConvexHull::required(theFeatureSet))
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
			STOPWATCH("Morphology/Hull/H/#4aaaea", "\t=");
			runParallel(parallelReduceConvHull, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Extrema 
		if (ExtremaFeatures::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Extrema/Ex/#4aaaea", "\t=");
			runParallel(ExtremaFeatures::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Euler 
		if (EulerNumber::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
			runParallel(EulerNumber::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Feret diameters and angles
		if (ParticleMetrics::feret_required(theFeatureSet)) 
		{
			STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
			runParallel(ParticleMetrics::reduce_feret, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Martin diameters
		if (ParticleMetrics::martin_required(theFeatureSet))
		{
			STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
			runParallel(ParticleMetrics::reduce_martin, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Nassenstein diameters
		if (ParticleMetrics::nassenstein_required(theFeatureSet))
		{
			STOPWATCH("Morphology/Nassenstein/N/#4aaaea", "\t=");
			runParallel(ParticleMetrics::reduce_nassenstein, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Chords
		if (Chords::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Chords/Ch/#4aaaea", "\t=");
			runParallel(Chords::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Hexagonality and polygonality
		if (Hexagonality_and_Polygonality::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(Hexagonality_and_Polygonality::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Enclosing, inscribing, and circumscribing circle
		if (EnclosingInscribingCircumscribingCircle::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(EnclosingInscribingCircumscribingCircle::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Geodetic length and thickness
		if (GeodeticLength_and_Thickness::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(GeodeticLength_and_Thickness::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== ROI radius
		if (RoiRadius::required(theFeatureSet))
		{
			STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
			runParallel(RoiRadius::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Erosion pixels
		if (ErosionPixels::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
			runParallel(ErosionPixels::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Fractal dimension
		if (FractalDimension::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
			runParallel(FractalDimension::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== GLCM aka Haralick 2D 
		if (GLCM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLCM texture/GLCM/#bbbbbb", "\t=");
			runParallel(GLCM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== GLRLM
		if (GLRLM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
			runParallel(GLRLM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== GLSZM
		if (GLSZM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
			runParallel(GLSZM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== GLDM
		if (GLDM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
			runParallel(GLDM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== NGTDM
		if (NGTDM_features::required(theFeatureSet))
		{
			STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
			runParallel(NGTDM_features::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Moments
		if (ImageMoments::required(theFeatureSet))
		{
			STOPWATCH("Moments/Moments/2D moms/#FFFACD", "\t=");
			runParallel(ImageMoments::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Gabor features
		if (GaborFeatures::required(theFeatureSet))
		{
			STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
			runParallel(GaborFeatures::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Radial distribution / Zernike 2D 
		if (ZernikeFeatures::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
			runParallel(parallelReduceZernike2D, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (RadialDistribution::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
			runParallel(RadialDistribution::reduce, nThr, workPerThread, tileSize, &roiLabelsVector, &labelData);
		}

	}
}

