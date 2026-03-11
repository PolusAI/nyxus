#ifdef WITH_PYTHON_H

#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "constants.h"
#include "dirs_and_files.h"
#include "environment.h"
#include "globals.h"
#include "raw_image_loader.h"
#include "features/contour.h"
#include "features/erosion.h"
#include "features/gabor.h"
#include "features/2d_geomoments.h"
#include "helpers/fsystem.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"


namespace Nyxus
{
	bool scan_slide_props_montage(SlideProps& p, int dim, const AnisotropyOptions& aniso);

	bool processIntSegImagePairInMemory (Environment & env, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label, int pair_index, const std::string& intens_name, const std::string& seg_name, std::vector<int> unprocessed_rois)
	{
		std::vector<int> trivRoiLabels;

		// Phase 1: gather ROI metrics

		if (! gatherRoisMetricsInMemory (env, intens, label, pair_index))	// Output - set of ROI labels, label-ROI cache mappings
			return false;

		// ROI metrics are gathered, let's publish them non-anisotropically into ROI's AABB
		// (Montage does not support anisotropy by design leaving it to Python user's control.)
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			r.make_nonanisotropic_aabb();
		}

		// Allocate each ROI's feature value buffer
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			r.initialize_fvals();
		}

		// Distribute ROIs among phases
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			if (size_t roiFootprint = r.get_ram_footprint_estimate(env.uniqueLabels.size()), 
				ramLim = env.get_ram_limit(); 
				roiFootprint >= ramLim)
			{
				unprocessed_rois.push_back(lab);
			}
			else
				trivRoiLabels.push_back(lab);
		}

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			processTrivialRoisInMemory (env, trivRoiLabels, intens, label, pair_index, env.get_ram_limit());
		}

		// Phase 3: skip nontrivial ROIs

		return true;
	}
	
	/// @brief In-memory feature maps processing for a single image pair.
	/// Gathers parent ROIs, generates child ROIs per kernel, computes features, saves to buffer.
	/// @param globalChildLabel [in/out] Running child label counter, persists across file pairs to avoid label collisions.
	bool processIntSegImagePairInMemory_fmaps (
		Environment & env,
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens,
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label,
		int pair_index,
		const std::string& intens_name,
		const std::string& seg_name,
		int64_t & globalChildLabel)
	{
		// Phase 1: gather parent ROI metrics
		if (! gatherRoisMetricsInMemory (env, intens, label, pair_index))
			return false;

		if (env.uniqueLabels.size() == 0)
			return true;

		// Set up parent ROIs
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			r.make_nonanisotropic_aabb();
			r.initialize_fvals();
		}

		// Collect parent labels, skipping blacklisted and oversized ROIs
		std::vector<int> parentLabels;
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			if (env.roi_is_blacklisted("", lab))
			{
				r.blacklisted = true;
				continue;
			}

			// Skip oversized ROIs that exceed RAM limit (matching non-fmaps behavior)
			size_t roiFootprint = r.get_ram_footprint_estimate(env.uniqueLabels.size());
			size_t ramLim = env.get_ram_limit();
			if (roiFootprint >= ramLim)
			{
				VERBOSLVL1 (env.get_verbosity_level(),
					std::cout << "Skipping oversized ROI " << lab
						<< " (estimated " << roiFootprint << " bytes >= RAM limit " << ramLim << " bytes)\n");
				continue;
			}

			parentLabels.push_back(lab);
		}

		// Phase 2: process each parent ROI
		for (auto parentLab : parentLabels)
		{
			LR& parentROI = env.roiData[parentLab];

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

			// Scan parent ROI pixels from in-memory arrays
			std::vector<int> singleParent = { parentLab };
			scanTrivialRoisInMemory (singleParent, intens, label, pair_index, env);

			// Allocate image matrix for parent
			allocateTrivialRoisBuffers (singleParent, env.roiData, env.hostCache);

			// Generate child ROIs
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

				// RAII guard swaps env's ROI data with child data and restores on scope exit (even on exception)
				EnvRoiSwapGuard guard (env, std::move(childLabels), std::move(childRoiData));

				std::vector<int> childLabelVec(env.uniqueLabels.begin(), env.uniqueLabels.end());
				std::sort(childLabelVec.begin(), childLabelVec.end());

				// Compute features on child ROIs
				reduce_trivial_rois_manual (childLabelVec, env);

				// Save as spatial feature map arrays
				save_features_2_fmap_arrays (
					env.theResultsCache,
					env,
					intens_name,
					seg_name,
					parentLab,
					parentXmin, parentYmin, /*parent_zmin=*/ 0,
					parentW, parentH, /*parent_d=*/ 1,
					env.fmaps_kernel_size(),
					env.uniqueLabels,
					env.roiData,
					childToParentMap);
			}

			// Free parent ROI buffers
			freeTrivialRoisBuffers (singleParent, env.roiData);

			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
		}

		return true;
	}

	std::optional<std::string> processMontage(
		Environment & env,
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intensity_images,
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images,
		int numReduceThreads,
		const std::vector<std::string>& intensity_names,
		const std::vector<std::string>& seg_names,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		// prepare the output

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		if (env.fmaps_prevents_arrow())
			return { "Arrow/Parquet output is not supported in feature maps (fmaps) mode." };

		if (write_apache)
		{
			env.arrow_stream = ArrowOutputStream();
			auto [status, msg] = env.arrow_stream.create_arrow_file (saveOption, get_arrow_filename(outputPath, env.nyxus_result_fname, saveOption), Nyxus::get_header(env));
			if (!status)
				return { "error creating Arrow file: " + msg.value() };
		}

		auto rI = intensity_images.unchecked<3>();
		size_t n_pairs = rI.shape(0);

		//****** prescan

		env.dataset.reset_dataset_props();

		for (size_t i = 0; i < n_pairs; i++)
		{
			SlideProps& p = env.dataset.dataset_props.emplace_back (intensity_names[i], seg_names[i]);

			// slide metrics
			if (!scan_slide_props_montage(p, 2, env.anisoOptions))
			{
				VERBOSLVL1(env.get_verbosity_level(), std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return {"error prescanning montage slide "+ std::to_string(i)};
			}
			VERBOSLVL1(env.get_verbosity_level(), std::cout << "\t " << p.slide_w << " W x " << p.slide_h << " H\tmax ROI " << p.max_roi_w << " x " << p.max_roi_h << "\tmin-max I " << Nyxus::virguler_real(p.min_preroi_inten) << "-" << Nyxus::virguler_real(p.max_preroi_inten) << "\t" << p.lolvl_slide_descr << "\n");
		}

		// update dataset's summary
		env.dataset.update_dataset_props_extrema();
		VERBOSLVL1(env.get_verbosity_level(), std::cout << "\t finished prescanning \n");

		//****** extract features

		int64_t globalChildLabel = 1;	// Persists across file pairs to avoid label collisions in fmaps mode

		for (int i_pair = 0; i_pair < n_pairs; i_pair++)
		{
			VERBOSLVL4 (env.get_verbosity_level(), std::cout << "processMontage() pair " << i_pair << "/" << n_pairs << "\n");

			clear_slide_rois (env.uniqueLabels, env.roiData);	// Clear ROI label list, ROI data, etc.

			if (env.fmaps_mode)
			{
				// Feature maps mode: generate child ROIs and compute features
				if (! processIntSegImagePairInMemory_fmaps (env, intensity_images, label_images, i_pair, intensity_names[i_pair], seg_names[i_pair], globalChildLabel))
					return { "error processing fmaps for a slide pair" };
			}
			else
			{
				std::vector<int> unprocessed_rois;

				if (! processIntSegImagePairInMemory (env, intensity_images, label_images, i_pair, intensity_names[i_pair], seg_names[i_pair], unprocessed_rois))
					return { "error processing a slide pair" };

				if (write_apache)
				{
					auto [status, msg] = env.arrow_stream.write_arrow_file (Nyxus::get_feature_values(env.theFeatureSet, env.uniqueLabels, env.roiData, env.dataset));
					if (!status)
						return { "error writing Arrow file: " + msg.value() };
				}
				else
				{
					if (! save_features_2_buffer(env.theResultsCache, env, DEFAULT_T_INDEX))
						return { "error saving results to a buffer" };
				}

				if (unprocessed_rois.size() > 0)
				{
					std::string erm = "the following ROIS are oversized and cannot be processed: ";
					for (const auto& roi: unprocessed_rois)
					{
						erm += std::to_string(roi);
						erm += ", ";
					}

					// remove trailing space and comma
					erm.pop_back();
					erm.pop_back();

					return { erm };
				}
			}

			// allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
		}

		if (write_apache)
		{
			// close arrow file after use
			auto [status, msg] = env.arrow_stream.close_arrow_file();
			if (!status)
				return { "error closing Arrow file: " + msg.value() };
		}
		return std::nullopt; // success
	}

}

#endif
