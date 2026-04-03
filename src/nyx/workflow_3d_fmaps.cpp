/// @file workflow_3d_fmaps.cpp
/// @brief 3D feature maps (fmaps) workflow: slides a cubic kernel across each parent ROI
///        to produce volumetric feature maps.  Analogous to workflow_2d_fmaps.cpp but
///        operates on 3D image cubes and supports time-frame iteration.

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

	/// @brief Generates kernel-sized child ROIs by sliding a 3D kernel across a parent ROI's image cube.
	/// Only creates child ROIs where the kernel center voxel is non-zero.
	/// @param parent The parent ROI (must have aux_image_cube populated)
	/// @param kernel_size The kernel size (odd integer >= 3)
	/// @param childLabels [out] Set of child ROI labels
	/// @param childRoiData [out] Map of child label -> child LR
	/// @param childToParentMap [out] Map of child label -> (parent_label, center_x, center_y, center_z)
	/// @param startLabel Starting label for child ROIs to avoid collisions across parents
	/// @return Number of child ROIs generated
#pragma clang optimize off
	int generateChildRois_3D (
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
		int parentD = parent.aabb.get_z_depth();
		int parentXmin = parent.aabb.get_xmin();
		int parentYmin = parent.aabb.get_ymin();
		int parentZmin = parent.aabb.get_zmin();

		const auto& cube = parent.aux_image_cube;

		// Slide kernel across the parent ROI's bounding box in 3D
		for (int cz = half; cz < parentD - half; cz++)
		{
			for (int cy = half; cy < parentH - half; cy++)
			{
				for (int cx = half; cx < parentW - half; cx++)
				{
					// Check if the kernel center voxel is within the mask
					auto centerVal = cube.xyz(cx, cy, cz);
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
					int childZmin = parentZmin + cz - half;
					int childXmax = parentXmin + cx + half;
					int childYmax = parentYmin + cy + half;
					int childZmax = parentZmin + cz + half;

					child.init_aabb_3D(childXmin, childYmin, childZmin);
					child.update_aabb_3D(childXmax, childYmax, childZmax);
					child.make_nonanisotropic_aabb();

					// Extract voxels from the parent's image cube
					double cmin = std::numeric_limits<double>::max();
					double cmax = std::numeric_limits<double>::lowest();
					for (int kz = 0; kz < kernel_size; kz++)
					{
						for (int ky = 0; ky < kernel_size; ky++)
						{
							for (int kx = 0; kx < kernel_size; kx++)
							{
								int localX = cx - half + kx;
								int localY = cy - half + ky;
								int localZ = cz - half + kz;
								auto val = cube.xyz(localX, localY, localZ);
								if (val != 0)
								{
									int imgX = parentXmin + localX;
									int imgY = parentYmin + localY;
									int imgZ = parentZmin + localZ;
									child.raw_pixels_3D.push_back(Pixel3(imgX, imgY, imgZ, val));
									child.aux_area++;
									if (val < cmin) cmin = val;
									if (val > cmax) cmax = val;
								}
							}
						}
					}
					if (child.aux_area > 0)
					{
						// Use parent ROI's min/max so all kernel patches share
						// the same binning range (consistent with pyradiomics
						// fixed-width global binning).
						child.aux_min = parent.aux_min;
						child.aux_max = parent.aux_max;
					}

					// Allocate and populate image cube
					child.aux_image_cube.allocate(kernel_size, kernel_size, kernel_size);
					child.aux_image_cube.calculate_from_pixelcloud(child.raw_pixels_3D, child.aabb);

					// Initialize feature value buffers
					child.initialize_fvals();

					// Store
					childLabels.insert(childLabel);
					childRoiData[childLabel] = std::move(child);
					childToParentMap[childLabel] = { parent.label, parentXmin + cx, parentYmin + cy, parentZmin + cz };

					childCount++;
				}
			}
		}

		return childCount;
	}
#pragma clang optimize on

	/// @brief Processes a single intensity-segmentation image pair in 3D feature maps mode.
	/// Gathers parent ROI metrics, loads voxels, generates 3D child ROIs per kernel,
	/// and outputs features for each child.
	/// @param globalChildLabel [in/out] Running child label counter, persists across file pairs.
	static bool processIntSegImagePair_3D_fmaps (
		Environment & env,
		const std::string & intens_fpath,
		const std::string & label_fpath,
		size_t filepair_index,
		size_t t_index,
		const std::vector<std::string>& z_indices,
		int64_t & globalChildLabel)
	{
		// Phase 1: gather ROI metrics
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Gathering ROI metrics (3D fmaps)\n");
		bool okGather = false;
		if (z_indices.size())
			okGather = gatherRoisMetrics_25D (env, filepair_index, intens_fpath, label_fpath, z_indices);
		else
			okGather = gatherRoisMetrics_3D (env, filepair_index, intens_fpath, label_fpath, t_index);
		if (!okGather)
		{
			std::string msg = "Error gathering ROI metrics from " + intens_fpath + " / " + label_fpath + "\n";
			std::cerr << msg;
			throw (std::runtime_error(msg));
		}

		if (env.uniqueLabels.size() == 0)
			return true;

		// Collect parent ROI labels
		std::vector<int> parentLabels;
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			r.initialize_fvals();
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

			// Check that the parent ROI is large enough for the kernel in all dimensions
			int parentW = parentROI.aabb.get_width();
			int parentH = parentROI.aabb.get_height();
			int parentD = parentROI.aabb.get_z_depth();
			if (parentW < env.fmaps_kernel_size() || parentH < env.fmaps_kernel_size() || parentD < env.fmaps_kernel_size())
			{
				VERBOSLVL2 (env.get_verbosity_level(),
					std::cout << "Skipping ROI " << parentLab
						<< " (too small for kernel: " << parentW << "x" << parentH << "x" << parentD
						<< " < " << env.fmaps_kernel_size() << ")\n");
				continue;
			}

			// Step a: Load parent ROI voxels and allocate image cube
			std::vector<int> singleParent = { parentLab };

			if (env.anisoOptions.customized() == false)
				scanTrivialRois_3D (env, singleParent, intens_fpath, label_fpath, t_index);
			else
			{
				double ax = env.anisoOptions.get_aniso_x(),
					ay = env.anisoOptions.get_aniso_y(),
					az = env.anisoOptions.get_aniso_z();
				scanTrivialRois_3D_anisotropic (env, singleParent, intens_fpath, label_fpath, t_index, ax, ay, az);
			}

			allocateTrivialRoisBuffers_3D (singleParent, env.roiData, env.hostCache);

			// Step b: Generate child ROIs
			std::unordered_set<int> childLabels;
			std::unordered_map<int, LR> childRoiData;
			std::unordered_map<int, FmapChildInfo> childToParentMap;

			int nChildren = generateChildRois_3D (
				parentROI,
				env.fmaps_kernel_size(),
				childLabels,
				childRoiData,
				childToParentMap,
				globalChildLabel);

			VERBOSLVL2 (env.get_verbosity_level(),
				std::cout << "ROI " << parentLab << ": generated " << nChildren << " child ROIs (3D)\n");

			globalChildLabel += nChildren;

			if (nChildren > 0)
			{
				// Capture parent geometry before the swap invalidates the reference
				int parentXmin = parentROI.aabb.get_xmin();
				int parentYmin = parentROI.aabb.get_ymin();
				int parentZmin = parentROI.aabb.get_zmin();

				// Step c: Compute features on child ROIs and output results
				reduceAndOutputChildRois_fmaps (
					env,
					std::move(childLabels), std::move(childRoiData), childToParentMap,
					intens_fpath, label_fpath,
					parentLab,
					parentXmin, parentYmin, parentZmin,
					parentW, parentH, parentD);
			}

			// Free parent ROI buffers
			freeTrivialRoisBuffers_3D (singleParent, env.roiData);

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

	/// @brief Entry point for the 3D feature maps workflow.
	/// Prescans all image pairs, then iterates time frames and file pairs,
	/// generating 3D child ROIs and computing volumetric features at each kernel position.
	/// @return 0 on success, nonzero on error.
	int processDataset_3D_fmaps (
		Environment & env,
		const std::vector <Imgfile3D_layoutA>& intensFiles,
		const std::vector <Imgfile3D_layoutA>& labelFiles,
		int numReduceThreads,
		const SaveOption saveOption,
		const std::string & outputPath)
	{
		reset_csv_header_state();

		//********************** prescan ***********************

		size_t nf = intensFiles.size();

		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "phase 0 (3D fmaps prescanning)\n");
		env.dataset.reset_dataset_props();

		for (size_t i = 0; i < nf; i++)
		{
			SlideProps& p = env.dataset.dataset_props.emplace_back (intensFiles[i].fdir + intensFiles[i].fname, labelFiles[i].fdir + labelFiles[i].fname);

			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "prescanning " << p.fname_int);
			if (! scan_slide_props(p, 3, env.anisoOptions, env.resultOptions.need_annotation()))
			{
				VERBOSLVL1 (env.get_verbosity_level(), std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return 1;
			}
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t " << p.slide_w << " W x " << p.slide_h << " H x " << p.volume_d << " D\n");
		}

		env.dataset.update_dataset_props_extrema();
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t finished prescanning \n");

		// One-time initialization
		init_slide_rois (env.uniqueLabels, env.roiData);

		bool ok = true;
		int64_t globalChildLabel = 1;

		// Iterate intensity-mask pairs
		for (size_t i = 0; i < nf; i++)
		{
			// Iterate time frames
			for (size_t t = 0; t < env.dataset.dataset_props[i].inten_time; t++)
			{
				clear_slide_rois (env.uniqueLabels, env.roiData);

				auto& ifile = intensFiles[i],
					& mfile = labelFiles[i];

				int digits = 2, k = (int)std::pow(10.f, digits);
				float perCent = float(i * 100 * k / nf) / float(k);
				VERBOSLVL1 (env.get_verbosity_level(), std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << ifile.fname << " SEG: " << mfile.fname << " T:" << t << "\n")

				if (! processIntSegImagePair_3D_fmaps(env, ifile.fdir+ifile.fname, mfile.fdir+mfile.fname, i, t, intensFiles[i].z_indices, globalChildLabel))
				{
					std::cerr << "Error in 3D feature maps processing for " << ifile.fname << " @ " << __FILE__ << ":" << __LINE__ << "\n";
					return 1;
				}

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
				{
					sureprint("\nAborting per user input\n");
					throw pybind11::error_already_set();
				}
#endif
			} //- time frames
		} //- inten-mask pairs

		return 0; // success
	}

} // namespace Nyxus
