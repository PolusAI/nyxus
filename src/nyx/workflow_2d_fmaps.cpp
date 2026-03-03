#include "helpers/fsystem.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <map>
#include <array>
#include <regex>
#include <string>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
	#include <pybind11/stl.h>
	#include <pybind11/numpy.h>
	namespace py = pybind11;
#endif

#include "constants.h"
#include "dirs_and_files.h"
#include "environment.h"
#include "globals.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "raw_image_loader.h"

namespace Nyxus
{

	/// @brief Generates kernel-sized child ROIs by sliding a kernel across a parent ROI's image matrix.
	/// Only creates child ROIs where the kernel center pixel is within the mask (non-zero intensity in the image matrix).
	/// @param parent The parent ROI (must have aux_image_matrix populated)
	/// @param kernel_size The kernel size (odd integer >= 3)
	/// @param childLabels [out] Set of child ROI labels
	/// @param childRoiData [out] Map of child label -> child LR
	/// @param childToParentMap [out] Map of child label -> (parent_label, center_x, center_y)
	/// @param startLabel Starting label for child ROIs to avoid collisions across parents
	/// @return Number of child ROIs generated
	//
	// Disable optimizations for this function: at -O3 the loop vectorizer
	// miscompiles inlined code (LR ctor, push_back, init_aabb, etc.) within
	// the sliding-kernel loops, causing heap corruption.  The function is
	// dominated by memory allocation (LR construction, hash-map insertion)
	// so vectorization provides no benefit here.
#pragma clang optimize off
	int generateChildRois (
		const LR & parent,
		int kernel_size,
		// out:
		std::unordered_set<int> & childLabels,
		std::unordered_map<int, LR> & childRoiData,
		std::unordered_map<int, FmapChildInfo> & childToParentMap,
		int64_t startLabel)
	{
		int half = kernel_size / 2;
		int childCount = 0;

		int parentW = parent.aabb.get_width();
		int parentH = parent.aabb.get_height();
		int parentXmin = parent.aabb.get_xmin();
		int parentYmin = parent.aabb.get_ymin();

		const auto& pixelMatrix = parent.aux_image_matrix.ReadablePixels();
		const auto* pixelData = pixelMatrix.data();
		int matW = parentW;  // image matrix stride (width)

		// Slide kernel across the parent ROI's bounding box
		for (int cy = half; cy < parentH - half; cy++)
		{
			for (int cx = half; cx < parentW - half; cx++)
			{
				// Check if the kernel center pixel is within the mask
				auto centerVal = pixelData[matW * cy + cx];
				if (centerVal == 0)
					continue;

				int64_t childLabel64 = startLabel + childCount;
				if (childLabel64 > std::numeric_limits<int>::max())
					throw std::overflow_error("Child ROI label overflow — too many child ROIs generated across all parents");
				int childLabel = (int)childLabel64;

				// Create child LR
				LR child(childLabel);

				// Set up child's AABB in image coordinates
				int childXmin = parentXmin + cx - half;
				int childYmin = parentYmin + cy - half;
				int childXmax = parentXmin + cx + half;
				int childYmax = parentYmin + cy + half;

				child.init_aabb(childXmin, childYmin);
				child.update_aabb(childXmax, childYmax);
				child.make_nonanisotropic_aabb();

				// Extract pixels from the parent's image matrix
				double cmin = std::numeric_limits<double>::max();
				double cmax = std::numeric_limits<double>::lowest();
				for (int ky = 0; ky < kernel_size; ky++)
				{
					for (int kx = 0; kx < kernel_size; kx++)
					{
						int localY = cy - half + ky;
						int localX = cx - half + kx;
						auto val = pixelData[matW * localY + localX];
						if (val != 0)
						{
							int imgX = parentXmin + localX;
							int imgY = parentYmin + localY;
							child.raw_pixels.push_back(Pixel2(imgX, imgY, val));
							child.aux_area++;
							if (val < cmin) cmin = val;
							if (val > cmax) cmax = val;
						}
					}
				}
				if (child.aux_area > 0)
				{
					// Use parent ROI's min/max so all kernel patches share
					// the same binning range (consistent with pyradiomics
					// fixed-width global binning).  Local cmin/cmax would
					// cause each tiny patch to get its own adaptive range,
					// collapsing most pixels to the same bin.
					child.aux_min = parent.aux_min;
					child.aux_max = parent.aux_max;
				}

				// Allocate and populate image matrix
				child.aux_image_matrix.allocate(kernel_size, kernel_size);
				child.aux_image_matrix.calculate_from_pixelcloud(child.raw_pixels, child.aabb);

				// Initialize feature value buffers
				child.initialize_fvals();

				// Store
				childLabels.insert(childLabel);
				childRoiData[childLabel] = std::move(child);
				childToParentMap[childLabel] = { parent.label, parentXmin + cx, parentYmin + cy };

				childCount++;
			}
		}

		return childCount;
	}
#pragma clang optimize on

	/// @brief Processes a single intensity-segmentation image pair in feature maps mode.
	/// Loads parent ROIs, generates child ROIs per kernel, computes features, outputs results.
	/// @param globalChildLabel [in/out] Running child label counter, persists across file pairs to avoid label collisions.
	static bool processIntSegImagePair_fmaps (
		Environment & env,
		const std::string & intens_fpath,
		const std::string & label_fpath,
		int filepair_index,
		int tot_num_filepairs,
		int64_t & globalChildLabel)
	{
		// Display progress
		if (tot_num_filepairs > 0)
		{
			int digits = (tot_num_filepairs >= 100) ? (int)std::log10(tot_num_filepairs/100.) + 1 : 1,
				k = (int)std::pow(10.f, digits);
			float perCent = float(filepair_index + 1) * 100.f / float(tot_num_filepairs);
			perCent = std::round(perCent * k) / k;
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "[ " << filepair_index+1 << " = " << std::setw(digits + 2) << perCent << "% ]\t" << intens_fpath << "\n")
		}

		// Phase 1: gather ROI metrics (unchanged)
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Gathering ROI metrics\n");
		bool okGather = gatherRoisMetrics (filepair_index, intens_fpath, label_fpath, env, env.theImLoader);
		if (!okGather)
		{
			std::string msg = "Error gathering ROI metrics from " + intens_fpath + " / " + label_fpath + "\n";
			std::cerr << msg;
			throw (std::runtime_error(msg));
		}

		if (env.uniqueLabels.size() == 0)
			return true;

		// Collect parent ROI labels (all trivial for fmaps — kernel-sized children are always small)
		std::vector<int> parentLabels;
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			if (env.roi_is_blacklisted("", lab))
			{
				r.blacklisted = true;
				continue;
			}
			parentLabels.push_back(lab);
		}

		// Phase 2 (fmaps): Process each parent ROI
		for (auto parentLab : parentLabels)
		{
			LR& parentROI = env.roiData[parentLab];

			// Check that the parent ROI is large enough for the kernel
			int parentW = parentROI.aabb.get_width();
			int parentH = parentROI.aabb.get_height();
			if (parentW < env.fmaps_kernel_size() || parentH < env.fmaps_kernel_size())
			{
				VERBOSLVL2 (env.get_verbosity_level(),
					std::cout << "Skipping ROI " << parentLab
						<< " (too small for kernel: " << parentW << "x" << parentH
						<< " < " << env.fmaps_kernel_size() << ")\n");
				continue;
			}

			// Step a: Load parent ROI pixels
			std::vector<int> singleParent = { parentLab };

			if (env.anisoOptions.customized() == false)
				scanTrivialRois (singleParent, intens_fpath, label_fpath, env, env.theImLoader);
			else
			{
				double ax = env.anisoOptions.get_aniso_x(),
					ay = env.anisoOptions.get_aniso_y();
				scanTrivialRois_anisotropic (singleParent, intens_fpath, label_fpath, env, env.theImLoader, ax, ay);
			}

			// Step b: Allocate image matrix for the parent
			allocateTrivialRoisBuffers (singleParent, env.roiData, env.hostCache);

			// Step c: Generate child ROIs
			std::unordered_set<int> childLabels;
			std::unordered_map<int, LR> childRoiData;
			std::unordered_map<int, FmapChildInfo> childToParentMap;

			int nChildren = generateChildRois (
				parentROI,
				env.fmaps_kernel_size(),
				childLabels,
				childRoiData,
				childToParentMap,
				globalChildLabel);

			VERBOSLVL2 (env.get_verbosity_level(),
				std::cout << "ROI " << parentLab << ": generated " << nChildren << " child ROIs\n");

			globalChildLabel += nChildren;

			if (nChildren > 0)
			{
				// Capture parent geometry before the swap invalidates the reference
				int parentXmin = parentROI.aabb.get_xmin();
				int parentYmin = parentROI.aabb.get_ymin();

				// Step d: Compute features on child ROIs
				// RAII guard swaps env's ROI data with child data and restores on scope exit (even on exception)
				EnvRoiSwapGuard guard (env, std::move(childLabels), std::move(childRoiData));

				std::vector<int> childLabelVec(env.uniqueLabels.begin(), env.uniqueLabels.end());
				std::sort(childLabelVec.begin(), childLabelVec.end());

				reduce_trivial_rois_manual (childLabelVec, env);

				// Step e: Output child ROI features
				if (env.saveOption == SaveOption::saveBuffer)
				{
					save_features_2_fmap_arrays (
						env.theResultsCache,
						env,
						intens_fpath,
						label_fpath,
						parentLab,
						parentXmin, parentYmin,
						parentW, parentH,
						env.fmaps_kernel_size(),
						env.uniqueLabels,
						env.roiData,
						childToParentMap);
				}
				else
				{
					save_features_2_csv_fmaps (
						env,
						intens_fpath,
						label_fpath,
						env.output_dir,
						filepair_index,
						env.uniqueLabels,
						env.roiData,
						childToParentMap);
				}
			}

			// Free parent ROI buffers
			freeTrivialRoisBuffers (singleParent, env.roiData);

#ifdef WITH_PYTHON_H
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
#endif
		}

		return true;
	}

	int processDataset_2D_fmaps (
		Environment & env,
		const std::vector<std::string> & intensFiles,
		const std::vector<std::string> & labelFiles,
		int numReduceThreads,
		const SaveOption saveOption,
		const std::string & outputPath)
	{
		reset_csv_header_state();

#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
			Stopwatch::reset();
#endif

		//********************** prescan (unchanged from segmented workflow) ***********************

		size_t nf = intensFiles.size();

		{ STOPWATCH("prescan/p0/P/#ccbbaa", "\t=");
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "phase 0 (prescanning)\n");

		env.dataset.reset_dataset_props();

		for (size_t i = 0; i < nf; i++)
		{
			SlideProps& p = env.dataset.dataset_props.emplace_back (intensFiles[i], labelFiles[i]);

			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "prescanning " << p.fname_int);
			if (! scan_slide_props(p, 2, env.anisoOptions, env.resultOptions.need_annotation()))
			{
				VERBOSLVL1 (env.get_verbosity_level(), std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return 1;
			}
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t " << p.slide_w << " W x " << p.slide_h << " H\tmax ROI " << p.max_roi_w << " x " << p.max_roi_h << "\tmin-max I " << Nyxus::virguler_real(p.min_preroi_inten) << "-" << Nyxus::virguler_real(p.max_preroi_inten) << "\t" << p.lolvl_slide_descr << "\n");
		}

		env.dataset.update_dataset_props_extrema();
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t finished prescanning \n");
		} // prescan timing

		// One-time initialization
		init_slide_rois (env.uniqueLabels, env.roiData);

		bool ok = true;
		int64_t globalChildLabel = 1;	// Persists across file pairs to avoid label collisions

		// Iterate intensity-segmentation pairs
		for (int i = 0; i < nf; i++)
		{
#ifdef CHECKTIMING
			if (Stopwatch::exclusive())
				Stopwatch::reset();
#endif

			clear_slide_rois (env.uniqueLabels, env.roiData);

			auto& ifp = intensFiles[i],
				& lfp = labelFiles[i];

			SlideProps& p = env.dataset.dataset_props[i];
			ok = env.theImLoader.open (p, env.fpimageOptions);
			if (ok == false)
			{
				std::cerr << "Terminating\n";
				return 1;
			}

			// Feature maps processing
			if (! processIntSegImagePair_fmaps(env, ifp, lfp, i, nf, globalChildLabel))
			{
				std::cerr << "Error in feature maps processing for " << ifp << " @ " << __FILE__ << ":" << __LINE__ << "\n";
				return 1;
			}

			env.theImLoader.close();

#ifdef WITH_PYTHON_H
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
#endif
		}

		return 0; // success
	}

} // namespace Nyxus
