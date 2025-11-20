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
#include "parallel.h"
#include "features/basic_morphology.h"
#include "features/chords.h"
#include "features/convex_hull.h"
#include "features/ellipse_fitting.h"
#include "features/euler_number.h"
#include "features/circle.h"
#include "features/erosion.h"
#include "features/extrema.h"
#include "features/fractal_dim.h"
#include "features/erosion.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/geodetic_len_thickness.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/gldzm.h"
#include "features/glszm.h"
#include "features/gldm.h"
#include "features/hexagonality_polygonality.h"
#include "features/ngldm.h"
#include "features/ngtdm.h"
#include "features/2d_geomoments.h"
#include "features/intensity.h"
#include "features/moments.h"
#include "features/neighbors.h"
#include "features/caliper.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "features/3d_intensity.h"
#include "features/3d_glcm.h"
#include "features/3d_gldm.h"
#include "features/3d_gldzm.h"
#include "features/3d_ngldm.h"
#include "features/3d_ngtdm.h"
#include "features/3d_glszm.h"
#include "features/3d_glrlm.h"
#include "features/3d_surface.h"
#include "features/focus_score.h"
#include "features/power_spectrum.h"
#include "features/saturation.h"
#include "features/sharpness.h"
#include "helpers/timing.h"

namespace Nyxus
{
	void reduce_trivial_2d (Environment & env, std::vector<int>& L, int n_threads, size_t work_per_thread, size_t job_size)
	{
		//==== Pixel intensity stats
		if (PixelIntensityFeatures::required(env.theFeatureSet))
		{
			STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_PixelIntensity;
			runParallel (PixelIntensityFeatures::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Basic morphology
		if (BasicMorphologyFeatures::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Basic/E/#4aaaea", "\t=");
			const Fsettings& fst = env.fsett_BasicMorphology;
			runParallel (BasicMorphologyFeatures::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Fitting an ellipse
		if (EllipseFittingFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_EllipseFitting;
			runParallel (EllipseFittingFeature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Contour-related ROI perimeter, equivalent circle diameter
		if (ContourFeature::required(env.theFeatureSet)
			|| ConvexHullFeature::required(env.theFeatureSet)
			|| FractalDimensionFeature::required(env.theFeatureSet)
			|| GeodeticLengthThicknessFeature::required(env.theFeatureSet)
			|| NeighborsFeature::required(env.theFeatureSet)
			|| RoiRadiusFeature::required(env.theFeatureSet)
			|| Imoms2D_feature::required(env.theFeatureSet)
			|| Smoms2D_feature::required(env.theFeatureSet)	)
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_Contour;
			runParallel (ContourFeature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Convex hull related solidity, circularity
		if (ConvexHullFeature::required(env.theFeatureSet)
			|| CaliperFeretFeature::required(env.theFeatureSet)
			|| CaliperMartinFeature::required(env.theFeatureSet)
			|| CaliperNassensteinFeature::required(env.theFeatureSet)
			|| HexagonalityPolygonalityFeature::required(env.theFeatureSet))
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on PERIMETER
			STOPWATCH("Morphology/Hull/H/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_ConvexHull;
			runParallel (parallelReduceConvHull, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Extrema 
		if (ExtremaFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Extrema/Ex/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_Extrema;
			runParallel (ExtremaFeature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Euler 
		if (EulerNumberFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_EulerNumber;
			runParallel (EulerNumberFeature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Feret diameters and angles
		if (CaliperFeretFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_CaliperFeret;
			runParallel (CaliperFeretFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Martin diameters
		if (CaliperMartinFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_CaliperMartin;
			runParallel (CaliperMartinFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Nassenstein diameters
		if (CaliperNassensteinFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Nassenstein/N/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_CaliperNassenstein;
			runParallel (CaliperNassensteinFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Chords
		if (ChordsFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Chords/Ch/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_Chords;
			runParallel (ChordsFeature::process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Geodetic length and thickness
		if (GeodeticLengthThicknessFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_GeodeticLengthThickness;
			runParallel (GeodeticLengthThicknessFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== ROI radius
		if (RoiRadiusFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_RoiRadius;
			runParallel (RoiRadiusFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Fractal dimension
		if (FractalDimensionFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_FractalDimension;
			runParallel (FractalDimensionFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== GLCM aka Haralick 2D 
		if (GLCMFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Texture/GLCM/GLCM/#bbbbbb", "\t=");
			Fsettings& fst = env.fsett_GLCM;
			runParallel (GLCMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== GLRLM
		if (GLRLMFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
			Fsettings& fst = env.fsett_GLRLM;
			runParallel (GLRLMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== GLDZM
		if (GLDZMFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Texture/GLDZM/DZ/#bbbbbb", "\t=");
			Fsettings& fst = env.fsett_GLDZM;
			runParallel (GLDZMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== GLSZM
		if (GLSZMFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
			Fsettings& fst = env.fsett_GLSZM;
			runParallel (GLSZMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== GLDM
		if (GLDMFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
			Fsettings& fst = env.fsett_GLDM;
			runParallel (GLDMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== NGLDM
		if (NGLDMfeature::required(env.theFeatureSet))
		{
			STOPWATCH("Texture/NGLDM/NG/#bbbbbb", "\t=");
			Fsettings& fst = env.fsett_NGLDM;
			runParallel (NGLDMfeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== NGTDM
		if (NGTDMFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
			Fsettings& fst = env.fsett_NGTDM;
			runParallel (NGTDMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Erosion, moments, and Gabor selectively run CPU-side or on GPU
#ifdef USE_GPU // Assuming GPU is available

		//==== Share ROI data (clouds, contours) with GPU-enabled features

		bool doImoms = Imoms2D_feature::required(env.theFeatureSet),
			doSmoms = Smoms2D_feature::required(env.theFeatureSet),
			doEros = ErosionPixelsFeature::required (env.theFeatureSet),
			doGabor = GaborFeature::required (env.theFeatureSet);
		if (doEros || doImoms || doSmoms || doGabor)
		{
			if (env.using_gpu())
			{
				size_t gbl = env.devCache.gpu_batch_len;		// former NyxusGpu::gpu_batch_len
				// Cache ROI clouds and contours
				size_t nrois = L.size();
				int nb = std::ceil ((float)nrois / (float)gbl);	
				for (int b = 0; b < nb; b++)
				{
					size_t off_this_batch = b * gbl;
					size_t actual_batch_len = off_this_batch + gbl > nrois ? nrois % gbl : gbl;

					{
						// upload a batch of ROIs
						STOPWATCH("Upload2Gpu/Upload/U/#ababab", "\t=");
						env.devCache.send_roi_data_gpuside (L, env.roiData, off_this_batch, actual_batch_len);
					}

					// Calculate GPU-enabled features via GPU

					if (doEros)
					{
						STOPWATCH("GPU-Morphology/GPU-Erosion/Er/#4aaaea", "\t=");
						ErosionPixelsFeature::gpu_process_all_rois (L, env.roiData, off_this_batch, actual_batch_len, env.devCache);
					}

					if (doImoms)
					{
						STOPWATCH("GPU-I-moments/GPU-I-moments/GM/#FFFACD", "\t=");
						Imoms2D_feature::gpu_process_all_rois (L, env.roiData, off_this_batch, actual_batch_len, env.singleROI, env.devCache);
					}
					
					if (doSmoms)
					{
						STOPWATCH("GPU-S-moments/GPU-S-moments/GM/#FFFACD", "\t=");
						Smoms2D_feature::gpu_process_all_rois (L, env.roiData, off_this_batch, actual_batch_len, env.singleROI, env.devCache);
					}

					if (doGabor)
					{
						STOPWATCH("GPU-Gabor/GPU-Gabor/Gabor/#f58231", "\t=");
						GaborFeature::gpu_process_all_rois (L, env.roiData, off_this_batch, actual_batch_len, env.devCache);
					}

				} // ROI batches
			}
			else
			{
				// user refused the GPU opportunity, so route calculation via the regular CPU-multithreaded way

				if (doEros)
				{
					STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
					Fsettings& fst = env.fsett_ErosionPixels;
					runParallel (ErosionPixelsFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
				}
				if (doImoms)
				{
					STOPWATCH("I-moments/I-moments/GM/#FFFACE", "\t=");
					Fsettings& fst = env.fsett_Imoms2D;
					runParallel (Imoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
				}
				if (doSmoms)
				{
					STOPWATCH("S-moments/S-moments/GM/#FFFACE", "\t=");
					Fsettings& fst = env.fsett_Smoms2D;
					runParallel (Smoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
				}
				if (doGabor)
				{
					STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
					Fsettings& fst = env.fsett_Gabor;
					runParallel (GaborFeature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
				}
			}
		}
#else	
		// GPU unavailable

		if (ErosionPixelsFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
			Fsettings& fst = env.fsett_ErosionPixels;
			runParallel (ErosionPixelsFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		if (Imoms2D_feature::required(env.theFeatureSet))
		{
			STOPWATCH("I-moments/I-moments/GM/#FFFACE", "\t=");
			Fsettings& fst = env.fsett_Imoms2D;
			runParallel (Imoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		if (Smoms2D_feature::required(env.theFeatureSet))
		{
			STOPWATCH("S-moments/S-moments/GM/#FFFACE", "\t=");
			Fsettings& fst = env.fsett_Smoms2D;
			runParallel (Smoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		if (GaborFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
			Fsettings& fst = env.fsett_Gabor;
			runParallel (GaborFeature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}
#endif

		//==== Radial distribution / Zernike 2D 
		if (ZernikeFeature::required(env.theFeatureSet))
		{
			STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
			Fsettings& fst = env.fsett_Zernike;
			runParallel (ZernikeFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (RadialDistributionFeature::required(env.theFeatureSet))
		{
			STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
			Fsettings& fst = env.fsett_RadialDistribution;
			runParallel (RadialDistributionFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		//==== Image quality features
		if (FocusScoreFeature::required(env.theFeatureSet))
		{
			STOPWATCH("ImageQuality/FocusScore/Rd/#00FFFF", "\t=");
			Fsettings& fst = env.fsett_FocusScore;
			runParallel (FocusScoreFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		if (PowerSpectrumFeature::required(env.theFeatureSet)) {
			STOPWATCH("ImageQuality/PowerSpectrum/Rd/#00FFFF", "\t=");
			Fsettings& fst = env.fsett_PowerSpectrum;
			runParallel (PowerSpectrumFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		if (SaturationFeature::required(env.theFeatureSet)) {
			STOPWATCH("ImageQuality/Saturation/Rd/#00FFFF", "\t=");
			Fsettings& fst = env.fsett_Saturation;
			runParallel (SaturationFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

		if (SharpnessFeature::required(env.theFeatureSet)) {
			STOPWATCH("ImageQuality/Sharpness/Rd/#00FFFF", "\t=");
			Fsettings& fst = env.fsett_Sharpness;
			runParallel (SharpnessFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}
	}

	void reduce_trivial_2d_wholeslide (Environment& env, LR & r)
	{
		//==== Pixel intensity stats
		if (PixelIntensityFeatures::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_PixelIntensity;
			PixelIntensityFeatures::extract (r, s, env.dataset);
		}

		//==== Basic morphology
		if (BasicMorphologyFeatures::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_BasicMorphology;
			BasicMorphologyFeatures::extract (r, s);
		}

		//==== Fitting an ellipse
		if (EllipseFittingFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_EllipseFitting;
			EllipseFittingFeature::extract (r, s);
		}

		//==== Contour-related ROI perimeter, equivalent circle diameter
		if (ContourFeature::required(env.theFeatureSet)
			|| ConvexHullFeature::required(env.theFeatureSet)
			|| FractalDimensionFeature::required(env.theFeatureSet)
			|| GeodeticLengthThicknessFeature::required(env.theFeatureSet)
			|| NeighborsFeature::required(env.theFeatureSet)
			|| RoiRadiusFeature::required(env.theFeatureSet)
			|| Imoms2D_feature::required(env.theFeatureSet)
			|| Smoms2D_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_Contour;
			ContourFeature::extract (r, s);
		}

		//==== Convex hull related solidity, circularity
		if (ConvexHullFeature::required(env.theFeatureSet)
			|| CaliperFeretFeature::required(env.theFeatureSet)
			|| CaliperMartinFeature::required(env.theFeatureSet)
			|| CaliperNassensteinFeature::required(env.theFeatureSet)
			|| HexagonalityPolygonalityFeature::required(env.theFeatureSet))
		{
			// CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY // depends on feature PERIMETER
			Fsettings& s = env.fsett_ConvexHull;
			ConvexHullFeature::extract (r, s);
		}

		//==== Extrema 
		if (ExtremaFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_Extrema;
			ExtremaFeature::extract (r, s);
		}

		//==== Euler 
		if (EulerNumberFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_EulerNumber;
			EulerNumberFeature::extract (r, s);
		}

		//==== Feret diameters and angles
		if (CaliperFeretFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_CaliperFeret;
			CaliperFeretFeature::extract (r, s);
		}

		//==== Martin diameters
		if (CaliperMartinFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_CaliperMartin;
			CaliperMartinFeature::extract (r, s);
		}

		//==== Nassenstein diameters
		if (CaliperNassensteinFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_CaliperNassenstein;
			CaliperNassensteinFeature::extract (r, s);
		}

		//==== Chords
		if (ChordsFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_Chords;
			ChordsFeature::extract (r, s);
		}

		//==== Geodetic length and thickness
		if (GeodeticLengthThicknessFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_GeodeticLengthThickness;
			GeodeticLengthThicknessFeature::extract (r, s);
		}

		//==== ROI radius
		if (RoiRadiusFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_RoiRadius;
			RoiRadiusFeature::extract (r, s);
		}

		//==== Fractal dimension
		if (FractalDimensionFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_FractalDimension;
			FractalDimensionFeature::extract (r, s);
		}

		//==== GLCM aka Haralick 2D 
		if (GLCMFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_GLCM;
			GLCMFeature::extract (r, s);
		}

		//==== GLRLM
		if (GLRLMFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_GLRLM;
			GLRLMFeature::extract (r, s);
		}

		//==== GLDZM
		if (GLDZMFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_GLDZM;
			GLDZMFeature::extract (r, s);
		}

		//==== GLSZM
		if (GLSZMFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_GLSZM;
			GLSZMFeature::extract (r, s);
		}

		//==== GLDM
		if (GLDMFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_GLDM;
			GLDMFeature::extract (r, s);
		}

		//==== NGLDM
		if (NGLDMfeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_NGLDM;
			NGLDMfeature::extract (r, s);
		}

		//==== NGTDM
		if (NGTDMFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_NGTDM;
			NGTDMFeature::extract (r, s);
		}

		//
		// future: run expensive features on available GPU devices
		//
		// future: otherwise, run expensive features CPU-side
		//

		if (ErosionPixelsFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_ErosionPixels;
			ErosionPixelsFeature::extract (r, s);
		}

		if (Imoms2D_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_Imoms2D;
			Imoms2D_feature::extract (r, s);
		}

		if (Smoms2D_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_Smoms2D;
			Smoms2D_feature::extract (r, s);
		}

		if (GaborFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_Gabor;
			GaborFeature::extract (r, s);
		}

		//
		// future: end run expensive features on available GPU devices
		//

		//==== Radial distribution / Zernike 2D 
		if (ZernikeFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_Zernike;
			ZernikeFeature::extract (r, s);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (RadialDistributionFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_RadialDistribution;
			RadialDistributionFeature::extract (r, s);
		}

		//==== Image quality features
		if (FocusScoreFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_FocusScore;
			FocusScoreFeature::extract (r, s);
		}

		if (PowerSpectrumFeature::required(env.theFeatureSet)) 
		{
			Fsettings& s = env.fsett_PowerSpectrum;
			PowerSpectrumFeature::extract (r, s);
		}

		if (SaturationFeature::required(env.theFeatureSet)) 
		{
			Fsettings& s = env.fsett_Saturation;
			SaturationFeature::extract (r, s);
		}

		if (SharpnessFeature::required(env.theFeatureSet)) 
		{
			Fsettings& s = env.fsett_Sharpness;
			SharpnessFeature::extract (r, s);
		}
	}

	void reduce_trivial_3d_wholevolume (Environment& env, LR& r)
	{
		//==== intensity
		if (D3_VoxelIntensityFeatures::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_VoxelIntensity;
			D3_VoxelIntensityFeatures::extract (r, s);
		}
		//==== shape
		if (D3_SurfaceFeature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_Surface;
			D3_SurfaceFeature::extract (r, s);
		}
		//==== texture		
		if (D3_GLCM_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_GLCM;
			D3_GLCM_feature::extract (r, s);
		}
		if (D3_GLDM_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_GLDM;
			D3_GLDM_feature::extract (r, s);
		}
		if (D3_GLDZM_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_GLDZM;
			D3_GLDZM_feature::extract (r, s);
		}
		if (D3_NGLDM_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_NGLDM;
			D3_NGLDM_feature::extract (r, s);
		}
		if (D3_NGTDM_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_NGTDM;
			D3_NGTDM_feature::extract (r, s);
		}
		if (D3_GLSZM_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_GLDZM;
			D3_GLSZM_feature::extract (r, s);
		}
		if (D3_GLRLM_feature::required(env.theFeatureSet))
		{
			Fsettings& s = env.fsett_D3_GLRLM;
			D3_GLRLM_feature::extract (r, s);
		}
	}

	void reduce_trivial_3d (Environment & env, std::vector<int>& L, int n_threads, size_t work_per_thread, size_t job_size)
	{
		//==== intensity
		if (D3_VoxelIntensityFeatures::required(env.theFeatureSet))
		{
			STOPWATCH("3D intensity/3Dintensity/3DI/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_VoxelIntensity;
			runParallel (D3_VoxelIntensityFeatures::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}
		//==== shape
		if (D3_SurfaceFeature::required(env.theFeatureSet))
		{
			STOPWATCH("3D shape/3Dshape/3Dsh/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_Surface;
			runParallel (D3_SurfaceFeature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}
		//==== texture		
		if (D3_GLCM_feature::required(env.theFeatureSet))
		{
			STOPWATCH("3D GLCM/3DGLCM/3DGLCM/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_GLCM;
			runParallel (D3_GLCM_feature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}		
		if (D3_GLDM_feature::required(env.theFeatureSet))
		{
			STOPWATCH("3D GLDM/3DGLDM/3DGLDM/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_GLDM;
			runParallel (D3_GLDM_feature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}	
		if (D3_GLDZM_feature::required(env.theFeatureSet))
		{
			STOPWATCH("3D GLDZM/3DGLDZM/3DGLDZM/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_GLDZM;
			runParallel (D3_GLDZM_feature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}		
		if (D3_NGLDM_feature::required(env.theFeatureSet))
		{
			STOPWATCH("3D NGLDM/3DNGLDM/3DNGLDM/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_NGLDM;
			runParallel (D3_NGLDM_feature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}
		if (D3_NGTDM_feature::required(env.theFeatureSet))
		{
			STOPWATCH("3D NGTDM/3DNGTDM/3DNGTDM/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_NGTDM;
			runParallel (D3_NGTDM_feature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}
		if (D3_GLSZM_feature::required(env.theFeatureSet))
		{
			STOPWATCH("3D GLSZM/3DGLSZM/3DGLSZM/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_GLSZM;
			runParallel (D3_GLSZM_feature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}		
		if (D3_GLRLM_feature::required(env.theFeatureSet))
		{
			STOPWATCH("3D GLRLM/3DGLRLM/3DGLRLM/#FFFF00", "\t=");
			const Fsettings & fst = env.fsett_D3_GLRLM;
			runParallel (D3_GLRLM_feature::reduce, n_threads, work_per_thread, job_size, &L, &env.roiData, fst, env.dataset);
		}

	}

	// Calculating features in parallel with hard-coded feature order. This function should be called once after a file pair processing is finished.
	void reduce_trivial_rois_manual (std::vector<int> & PendingRoisLabels, Environment & env)
	{
		//==== Parallel execution parameters 
		int n_reduce_threads = env.n_reduce_threads;
		size_t jobSize = PendingRoisLabels.size(),
			workPerThread = jobSize / n_reduce_threads;

		switch (env.dim())
		{
		case 2:
			reduce_trivial_2d (env, PendingRoisLabels, n_reduce_threads, workPerThread, jobSize);
			break;
		case 3:
			reduce_trivial_3d (env, PendingRoisLabels, n_reduce_threads, workPerThread, jobSize);
			break;
		default:
			std::string msg = "ERROR: unsupported dimension " + std::to_string(env.dim());
			#ifdef WITH_PYTHON_H
				throw std::out_of_range(msg.c_str());
			#endif			
			std::cerr << msg << "\n";
			break;
		}
	}

	// Calculates features sequentially
	void reduce_trivial_wholeslide (Environment & env, LR & vroi)
	{
		switch (env.dim())
		{
		case 2:
			reduce_trivial_2d_wholeslide (env, vroi); // former reduce_trivial_2d()
			break;
		case 3:
			{
				std::vector<int> p = { 1 };
				reduce_trivial_3d (env, p, 1, 1, 1);
			}
			break;
		default:
			std::string msg = "ERROR: unsupported dimension " + std::to_string(env.dim());
			#ifdef WITH_PYTHON_H
				throw std::out_of_range(msg.c_str());
			#endif			
			std::cerr << msg << "\n";
			break;
		}
	}

	void reduce_neighbors_and_dependencies_manual (Environment & env)
	{
		std::vector <int> L;
		L.reserve (env.uniqueLabels.size());
		L.insert (L.end(), env.uniqueLabels.begin(), env.uniqueLabels.end());

		//==== Parallel execution parameters 
		int n_reduce_threads = 1;		
		size_t jobSize = L.size(),
			workPerThread = jobSize / n_reduce_threads;

		//==== Neighbors
		bool needNeigs = NeighborsFeature::required(env.theFeatureSet),
			needHexpol = HexagonalityPolygonalityFeature::required(env.theFeatureSet),
			needEnclosing = EnclosingInscribingCircumscribingCircleFeature::required(env.theFeatureSet);
		if (needNeigs || needHexpol || needEnclosing)
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			const Fsettings& s = env.fsett_Neighbors;
			NeighborsFeature::manual_reduce (env.roiData, s, env.uniqueLabels);
		}

		//==== Hexagonality and polygonality
		if (HexagonalityPolygonalityFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			const Fsettings & fst = env.fsett_HexagonalityPolygonality;
			runParallel (HexagonalityPolygonalityFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &L, &env.roiData, fst, env.dataset);
		}

		//==== Enclosing, inscribing, and circumscribing circle
		if (EnclosingInscribingCircumscribingCircleFeature::required(env.theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			const Fsettings & fst = env.fsett_HexagonalityPolygonality;
			runParallel (EnclosingInscribingCircumscribingCircleFeature::parallel_process_1_batch, n_reduce_threads, workPerThread, jobSize, &L, &env.roiData, fst, env.dataset);
		}
	}

}

