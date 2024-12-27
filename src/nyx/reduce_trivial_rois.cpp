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
#include "features/3d_gldzm.h"
#include "features/3d_glszm.h"
//--future-- #include "features/3d_surface.h"

#include "features/focus_score.h"
#include "features/power_spectrum.h"
#include "features/saturation.h"
#include "features/sharpness.h"

#include "helpers/timing.h"
#include "parallel.h"
#ifdef USE_GPU
	#include "gpucache.h"
#endif

namespace Nyxus
{
	void reduce_trivial_2d(std::vector<int>& L, int n_threads, size_t work_per_thread, size_t job_size)
	{
		//==== Pixel intensity stats
		if (PixelIntensityFeatures::required(theFeatureSet))
		{
			STOPWATCH("Intensity/Intensity/Int/#FFFF00", "\t=");
			runParallel(PixelIntensityFeatures::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Basic morphology
		if (BasicMorphologyFeatures::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Basic/E/#4aaaea", "\t=");
			runParallel(BasicMorphologyFeatures::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Fitting an ellipse
		if (EllipseFittingFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Ellipticity/E/#4aaaea", "\t=");
			runParallel(EllipseFittingFeature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Contour-related ROI perimeter, equivalent circle diameter
		if (ContourFeature::required(theFeatureSet)
			|| ConvexHullFeature::required(theFeatureSet)
			|| FractalDimensionFeature::required(theFeatureSet)
			|| GeodeticLengthThicknessFeature::required(theFeatureSet)
			|| NeighborsFeature::required(theFeatureSet)
			|| RoiRadiusFeature::required(theFeatureSet)
			|| Imoms2D_feature::required(theFeatureSet)
			|| Smoms2D_feature::required(theFeatureSet)	)
		{
			STOPWATCH("Morphology/Contour/C/#4aaaea", "\t=");
			runParallel(ContourFeature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
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
			runParallel(parallelReduceConvHull, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Extrema 
		if (ExtremaFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Extrema/Ex/#4aaaea", "\t=");
			runParallel(ExtremaFeature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Euler 
		if (EulerNumberFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Euler/Eu/#4aaaea", "\t=");
			runParallel(EulerNumberFeature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Feret diameters and angles
		if (CaliperFeretFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Feret/F/#4aaaea", "\t=");
			runParallel(CaliperFeretFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Martin diameters
		if (CaliperMartinFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Martin/M/#4aaaea", "\t=");
			runParallel(CaliperMartinFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Nassenstein diameters
		if (CaliperNassensteinFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Nassenstein/N/#4aaaea", "\t=");
			runParallel(CaliperNassensteinFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Chords
		if (ChordsFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Chords/Ch/#4aaaea", "\t=");
			runParallel(ChordsFeature::process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Geodetic length and thickness
		if (GeodeticLengthThicknessFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/HexPolygEncloInsCircleGeodetLenThickness/HP/#4aaaea", "\t=");
			runParallel(GeodeticLengthThicknessFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== ROI radius
		if (RoiRadiusFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/RoiR/R/#4aaaea", "\t=");
			runParallel(RoiRadiusFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Fractal dimension
		if (FractalDimensionFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Fractal dimension/Fd/#4aaaea", "\t=");
			runParallel(FractalDimensionFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== GLCM aka Haralick 2D 
		if (GLCMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLCM/GLCM/#bbbbbb", "\t=");
			runParallel(GLCMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== GLRLM
		if (GLRLMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLRLM/RL/#bbbbbb", "\t=");
			runParallel(GLRLMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== GLDZM
		if (GLDZMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLDZM/DZ/#bbbbbb", "\t=");
			runParallel(GLDZMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== GLSZM
		if (GLSZMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLSZM/SZ/#bbbbbb", "\t=");
			runParallel(GLSZMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== GLDM
		if (GLDMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/GLDM/D/#bbbbbb", "\t=");
			runParallel(GLDMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== NGLDM
		if (NGLDMfeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/NGLDM/NG/#bbbbbb", "\t=");
			runParallel(NGLDMfeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== NGTDM
		if (NGTDMFeature::required(theFeatureSet))
		{
			STOPWATCH("Texture/NGTDM/NG/#bbbbbb", "\t=");
			runParallel(NGTDMFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

#ifdef USE_GPU // Assuming GPU is available

		//==== Share ROI data (clouds, contours) with GPU-enabled features

		bool doImoms = Imoms2D_feature::required(theFeatureSet),
			doSmoms = Smoms2D_feature::required(theFeatureSet),
			doEros = ErosionPixelsFeature::required (theFeatureSet),
			doGabor = GaborFeature::required (theFeatureSet);
		if (doEros || doImoms || doSmoms || doGabor)
		{
			if (theEnvironment.using_gpu())
			{
				// Cache ROI clouds and contours
				int nrois = L.size(),
					nb = std::ceil((float)nrois / (float)NyxusGpu::gpu_batch_len);
				for (int b = 0; b < nb; b++)
				{
					size_t off_this_batch = b * NyxusGpu::gpu_batch_len;
					size_t actual_batch_len = off_this_batch + NyxusGpu::gpu_batch_len > nrois ? nrois % NyxusGpu::gpu_batch_len : NyxusGpu::gpu_batch_len;

					{
						// upload a batch of ROIs
						STOPWATCH("Upload2Gpu/Upload/U/#ababab", "\t=");
						NyxusGpu::send_roi_data_gpuside (L, roiData, off_this_batch, actual_batch_len);
					}

					// Calculate GPU-enabled features via GPU

					if (doEros)
					{
						STOPWATCH("GPU-Morphology/GPU-Erosion/Er/#4aaaea", "\t=");
						ErosionPixelsFeature::gpu_process_all_rois(L, roiData, off_this_batch, actual_batch_len);
					}

					if (doImoms)
					{
						STOPWATCH("GPU-I-moments/GPU-I-moments/GM/#FFFACD", "\t=");
						Imoms2D_feature::gpu_process_all_rois (L, roiData, off_this_batch, actual_batch_len);
					}
					
					if (doSmoms)
					{
						STOPWATCH("GPU-S-moments/GPU-S-moments/GM/#FFFACD", "\t=");
						Smoms2D_feature::gpu_process_all_rois (L, roiData, off_this_batch, actual_batch_len);
					}

					if (doGabor)
					{
						STOPWATCH("GPU-Gabor/GPU-Gabor/Gabor/#f58231", "\t=");
						GaborFeature::gpu_process_all_rois (L, roiData, off_this_batch, actual_batch_len);
					}

				} // ROI batches
			}
			else
			{
				// user refused the GPU opportunity, so route calculation via the regular CPU-multithreaded way

				if (doEros)
				{
					STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
					runParallel(ErosionPixelsFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
				}
				if (doImoms)
				{
					STOPWATCH("I-moments/I-moments/GM/#FFFACE", "\t=");
					runParallel (Imoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
				}
				if (doSmoms)
				{
					STOPWATCH("S-moments/S-moments/GM/#FFFACE", "\t=");
					runParallel (Smoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
				}
				if (doGabor)
				{
					STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
					runParallel(GaborFeature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
				}
			}
		}
#else	
		// GPU unavailable

		if (ErosionPixelsFeature::required(theFeatureSet))
		{
			STOPWATCH("Morphology/Erosion/Er/#4aaaea", "\t=");
			runParallel(ErosionPixelsFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		if (Imoms2D_feature::required(theFeatureSet))
		{
			STOPWATCH("I-moments/I-moments/GM/#FFFACE", "\t=");
			runParallel(Imoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		if (Smoms2D_feature::required(theFeatureSet))
		{
			STOPWATCH("S-moments/S-moments/GM/#FFFACE", "\t=");
			runParallel(Smoms2D_feature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		if (GaborFeature::required(theFeatureSet))
		{
			STOPWATCH("Gabor/Gabor/Gabor/#f58231", "\t=");
			runParallel(GaborFeature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}
#endif

		//==== Radial distribution / Zernike 2D 
		if (ZernikeFeature::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Zernike/Rz/#00FFFF", "\t=");
			runParallel(ZernikeFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Radial distribution / FracAtD, MeanFraq, and RadialCV
		if (RadialDistributionFeature::required(theFeatureSet))
		{
			STOPWATCH("RDistribution/Rdist/Rd/#00FFFF", "\t=");
			runParallel(RadialDistributionFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== Image quality features
		if (FocusScoreFeature::required(theFeatureSet))
		{
			STOPWATCH("ImageQuality/FocusScore/Rd/#00FFFF", "\t=");
			runParallel(FocusScoreFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		if (PowerSpectrumFeature::required(theFeatureSet)) {
			STOPWATCH("ImageQuality/PowerSpectrum/Rd/#00FFFF", "\t=");
			runParallel(PowerSpectrumFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		if (SaturationFeature::required(theFeatureSet)) {
			STOPWATCH("ImageQuality/Saturation/Rd/#00FFFF", "\t=");
			runParallel(SaturationFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		if (SharpnessFeature::required(theFeatureSet)) {
			STOPWATCH("ImageQuality/Sharpness/Rd/#00FFFF", "\t=");
			runParallel(SharpnessFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size, &L, &roiData);
		}
	}

	void reduce_trivial_3d (std::vector<int>& L, int n_threads, size_t work_per_thread, size_t job_size)
	{
		//==== intensity
		if (D3_PixelIntensityFeatures::required(theFeatureSet))
		{
			STOPWATCH("3D intensity/3Dintensity/3DI/#FFFF00", "\t=");
			runParallel (D3_PixelIntensityFeatures::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}
		//==== texture		
		if (D3_GLCM_feature::required(theFeatureSet))
		{
			STOPWATCH("3D GLCM/3DGLCM/3DGLCM/#FFFF00", "\t=");
			runParallel (D3_GLCM_feature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}		
		if (D3_GLDZM_feature::required(theFeatureSet))
		{
			STOPWATCH("3D GLDZM/3DGLDZM/3DGLDZM/#FFFF00", "\t=");
			runParallel (D3_GLDZM_feature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}		
		if (D3_GLSZM_feature::required(theFeatureSet))
		{
			STOPWATCH("3D GLSZM/3DGLSZM/3DGLSZM/#FFFF00", "\t=");
			runParallel (D3_GLSZM_feature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		}

		//==== morphology/surface
		//--future--
		//		if (D3_SurfaceFeature::required(theFeatureSet))
		//		{
		//			STOPWATCH("3Dsurface/3Dsurface/3Dsurf/#FFFF00", "\t=");
		//			runParallel(D3_SurfaceFeature::reduce, n_threads, work_per_thread, job_size, &L, &roiData);
		//		}
	}

	// Calculating features in parallel with hard-coded feature order. This function should be called once after a file pair processing is finished.
	void reduce_trivial_rois_manual (std::vector<int> & PendingRoisLabels)
	{
		//==== Parallel execution parameters 
		int n_reduce_threads = theEnvironment.n_reduce_threads;
		size_t jobSize = PendingRoisLabels.size(),
			workPerThread = jobSize / n_reduce_threads;

		switch (theEnvironment.dim())
		{
		case 2:
			reduce_trivial_2d (PendingRoisLabels, n_reduce_threads, workPerThread, jobSize);
			break;
		case 3:
			reduce_trivial_3d (PendingRoisLabels, n_reduce_threads, workPerThread, jobSize);
			break;
		default:
			std::string msg = "ERROR: unsupported dimension " + std::to_string(theEnvironment.dim());
			#ifdef WITH_PYTHON_H
				throw std::out_of_range(msg.c_str());
			#endif			
			std::cerr << msg << "\n";
			break;
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
		bool needNeigs = NeighborsFeature::required(theFeatureSet),
			needHexpol = HexagonalityPolygonalityFeature::required(theFeatureSet),
			needEnclosing = EnclosingInscribingCircumscribingCircleFeature::required(theFeatureSet);
		if (needNeigs || needHexpol || needEnclosing)
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

