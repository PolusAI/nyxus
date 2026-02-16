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

#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
	#include <pybind11/stl.h>
	#include <pybind11/numpy.h>
	namespace py = pybind11;
#endif

#include "constants.h"
#include "dirs_and_files.h"
#include "environment.h"
#include "features/contour.h"
#include "features/erosion.h"
#include "features/gabor.h"
#include "features/2d_geomoments.h"
#include "globals.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "raw_image_loader.h"

namespace Nyxus
{

	bool processIntSegImagePair (Environment & env, const std::string& intens_fpath, const std::string& label_fpath, int filepair_index, int tot_num_filepairs)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

		// Timing block (image scanning)
		{
			{ STOPWATCH("Image scan1/scan1/s1/#aabbcc", "\t=");

			VERBOSLVL2 (env.get_verbosity_level(),
				// Report the amount of free RAM
				unsigned long long freeRamAmt = Nyxus::getAvailPhysMemory();
				static unsigned long long initial_freeRamAmt = 0;
				if (initial_freeRamAmt == 0)
					initial_freeRamAmt = freeRamAmt;
				unsigned long long memDiff = 0;
				char sgn;
				if (freeRamAmt > initial_freeRamAmt)
				{
					memDiff = freeRamAmt - initial_freeRamAmt;
					sgn = '+';
				}
				else // system memory can be freed by other processes
				{
					memDiff = initial_freeRamAmt - freeRamAmt;
					sgn = '-';
				}
				std::cout << std::setw(15) << freeRamAmt << " b free (" << sgn << memDiff << ") ";
			)
				// Display (1) dataset progress info and (2) file pair info
				int digits = std::log10(float(tot_num_filepairs)/100.) + 1,
					k = std::pow(10.f, std::abs(digits));
				float perCent = float(filepair_index + 1) * 100. / float(tot_num_filepairs);
				perCent = std::round(perCent * k) / k;
				VERBOSLVL1 (env.get_verbosity_level(), std::cout << "[ " << filepair_index+1 << " = " << std::setw(digits + 2) << perCent << "% ]\t" << intens_fpath << "\n")
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n")
			}

			{ STOPWATCH("Image scan2a/scan2a/s2a/#aabbcc", "\t=");
				// Phase 1: gather ROI metrics
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Gathering ROI metrics\n");
				bool okGather = gatherRoisMetrics (filepair_index, intens_fpath, label_fpath, env, env.theImLoader);	// Output - set of ROI labels, label-ROI cache mappings
				if (!okGather)
				{
					std::string msg = "Error gathering ROI metrics from " + intens_fpath + " / " + label_fpath + "\n";
					std::cerr << msg;
					throw (std::runtime_error(msg));
					return false;
				}

				// Any ROIs in this slide? (Such slides may exist, it's normal.)
				if (env.uniqueLabels.size() == 0)
				{
					return true;
				}
			}

			{ STOPWATCH("Image scan2b/scan2b/s2b/#aabbcc", "\t=");
				// Allocate each ROI's feature value buffer
				for (auto lab : env.uniqueLabels)
				{
					LR& r = env.roiData[lab];
					r.initialize_fvals();
				}

				#ifndef WITH_PYTHON_H
				// Dump ROI metrics to the output directory
				VERBOSLVL2 (env.get_verbosity_level(), dump_roi_metrics(env.dim(), env.output_dir, env.get_ram_limit(), label_fpath, env.uniqueLabels, env.roiData))
				#endif		
			}

			{ STOPWATCH("Image scan3/scan3/s3/#aabbcc", "\t=");
				// Support of ROI blacklist
				fs::path fp (label_fpath);
				std::string shortSegFname = fp.stem().string() + fp.extension().string();

				// Distribute ROIs among phases
				for (auto lab : env.uniqueLabels)
				{
					LR& r = env.roiData[lab];

					// Skip blacklisted ROI
					if (env.roi_is_blacklisted(shortSegFname, lab))
					{
						r.blacklisted = true;
						VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Skipping blacklisted ROI " << lab << " for mask " << shortSegFname << "\n");
						continue;
					}

					// Examine ROI's memory footprint
					if (size_t roiFootprint = r.get_ram_footprint_estimate (env.uniqueLabels.size()),
						ramLim = env.get_ram_limit();
						roiFootprint >= ramLim)
					{
						VERBOSLVL2 (env.get_verbosity_level(),
							std::cout << "oversized ROI " << lab
							<< " (S=" << r.aux_area
							<< " W=" << r.aabb.get_width()
							<< " H=" << r.aabb.get_height()
							<< " px footprint=" << roiFootprint << " b"
							<< ")\n"
						);
						nontrivRoiLabels.push_back(lab);
					}
					else
						trivRoiLabels.push_back(lab);
				}
			}
		}

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Processing trivial ROIs\n");
			processTrivialRois (env, trivRoiLabels, intens_fpath, label_fpath, env.get_ram_limit());
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRoiLabels.size())
		{
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Processing oversized ROIs\n");
			processNontrivialRois (env, nontrivRoiLabels, intens_fpath, label_fpath);
		}

		return true;
	}

	int processDataset_2D_segmented (
		Environment & env,
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		int numReduceThreads,
		const SaveOption saveOption,
		const std::string& outputPath)
	{

#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
			Stopwatch::reset();
#endif		

		//********************** prescan ***********************

		// slide properties
		size_t nf = intensFiles.size();

		{ STOPWATCH("prescan/p0/P/#ccbbaa", "\t=");
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "phase 0: prescanning " << nf << " slides \n");

		env.dataset.reset_dataset_props();

		for (size_t i = 0; i < nf; i++)
		{
			SlideProps& p = env.dataset.dataset_props.emplace_back (intensFiles[i], labelFiles[i]);

			// slide metrics
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "prescanning " << p.fname_int);
			if (! scan_slide_props(p, 2, env.anisoOptions, env.resultOptions.need_annotation()))
			{
				VERBOSLVL1 (env.get_verbosity_level(), std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return 1;
			}
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t " << p.slide_w << " W x " << p.slide_h << " H\tmax ROI " << p.max_roi_w << " x " << p.max_roi_h << "\tmin-max I " << Nyxus::virguler_real(p.min_preroi_inten) << "-" << Nyxus::virguler_real(p.max_preroi_inten) << "\t" << p.lolvl_slide_descr << "\n");
		}

		// update dataset's summary
		env.dataset.update_dataset_props_extrema();
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t finished prescanning \n");

		//********************** allocate the GPU cache ***********************

#ifdef USE_GPU
		// what parts of GPU cache we need to bother about ?
		bool needContour = ContourFeature::required(env.theFeatureSet),
			needErosion = ErosionPixelsFeature::required(env.theFeatureSet),
			needGabor = GaborFeature::required(env.theFeatureSet),
			needImoments = Imoms2D_feature::required(env.theFeatureSet),
			needSmoments = Smoms2D_feature::required(env.theFeatureSet),
			needMoments = needImoments || needSmoments;

		// whole slide's contour is just 4 vertices long
		size_t kontrLen = env.singleROI ? 4 : env.dataset.dataset_max_combined_roicloud_len;

		if (env.using_gpu())
		{
			// allocate
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "allocating GPU cache \n");
			auto allocErr = env.devCache.allocate_gpu_cache(
				// out
				env.devCache.gpu_roiclouds_2d,
				env.devCache.gpu_roicontours_2d,
				&env.devCache.dev_realintens,
				&env.devCache.dev_prereduce,
				env.devCache.gpu_featurestatebuf,
				env.devCache.devicereduce_temp_storage_szb,
				&env.devCache.dev_devicereduce_temp_storage,
				env.devCache.gpu_batch_len,
				&env.devCache.dev_imat1,
				&env.devCache.dev_imat2,
				env.devCache.gabor_linear_image,
				env.devCache.gabor_linear_kernel,
				env.devCache.gabor_result,
				env.devCache.gabor_energy_image,
				// in
				needContour,
				needErosion,
				needGabor,
				needMoments,
				env.dataset.dataset_max_combined_roicloud_len, // desired totCloLen,
				kontrLen, // desired totKontLen,
				env.dataset.dataset_max_n_rois,	// labels.size()
				env.dataset.dataset_max_roi_area,
				env.dataset.dataset_max_roi_w,
				env.dataset.dataset_max_roi_h,
				GaborFeature::f0_theta_pairs.size(),
				GaborFeature::n);
			if (allocErr.has_value())	// we need max ROI area inside the function to calculate the batch size if 'dataset_max_combined_roicloud_len' doesn't fit in RAM
			{
				std::cerr << "allocating GPU cache failed: " << allocErr.value() << "\n";
				return 1;
			}

			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t ---done allocating GPU cache \n");
		}
#endif
		} // prescan timing

		// One-time initialization
		init_slide_rois (env.uniqueLabels, env.roiData);

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize arrow writer if needed
		if (write_apache) {

			env.arrow_stream = ArrowOutputStream();
			auto [status, msg] = env.arrow_stream.create_arrow_file(
				saveOption,
				get_arrow_filename (outputPath, env.nyxus_result_fname, saveOption),
				Nyxus::get_header (env));
			if (!status) 
			{
				std::cerr << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		bool ok = true;

		// Iterate intensity-segmentation pairs and process ROIs
		for (int i = 0; i < nf; i++)
		{
#ifdef CHECKTIMING
			if (Stopwatch::exclusive())
				Stopwatch::reset();
#endif

			// Clear ROI data cached for the previous image
			clear_slide_rois (env.uniqueLabels, env.roiData);

			auto& ifp = intensFiles[i],
				& lfp = labelFiles[i];

			if (ifp != env.dataset.dataset_props[i].fname_int || lfp != env.dataset.dataset_props[i].fname_seg)
			{
				std::cout << "\nMISMATCH! " << ifp << ":" << env.dataset.dataset_props[i].fname_int << ":" << lfp << ":" << env.dataset.dataset_props[i].fname_seg << "\n";
			}

			// Scan one label-intensity pair 
			SlideProps& p = env.dataset.dataset_props[i];
			ok = env.theImLoader.open (p, env.fpimageOptions);
			if (ok == false)
			{
				std::cerr << "Terminating\n";
				return 1;
			}

			// Do phased processing: prescan, trivial ROI processing, oversized ROI processing
			if (! processIntSegImagePair(env, ifp, lfp, i, nf))
			{
				std::cerr << "Error featurizing segmented slide " << ifp << " @ " << __FILE__ << ":" << __LINE__ << "\n";
				return 1;
			}

			if (write_apache)
			{

				auto [status, msg] = env.arrow_stream.write_arrow_file (Nyxus::get_feature_values(env.theFeatureSet, env.uniqueLabels, env.roiData, env.dataset));
				if (!status) 
				{
					std::cerr << "Error writing Arrow file: " << msg.value() << std::endl;
					return 2;
				}
			}
			else 
				if (saveOption == SaveOption::saveCSV)
				{
					ok = save_features_2_csv (env, ifp, lfp, outputPath, 0/*pass 0 as t_index is not used in 2D scenario*/ , env.resultOptions.need_aggregation());

					if (ok == false)
					{
						std::cout << "error saving results to CSV file, details: " << __FILE__ << ":" << __LINE__ << std::endl;
						return 2;
					}
				}
				else
				{
					ok = save_features_2_buffer (env.theResultsCache, env, DEFAULT_T_INDEX);

					if (ok == false)
					{
						std::cout << "error saving results to a buffer, details: " << __FILE__ << ":" << __LINE__ << std::endl;
						return 2;
					}
				}

			env.theImLoader.close();

			// Save nested ROI related info of this image
			if (env.nestedOptions.defined())
				save_nested_roi_info (nestedRoiData, env.uniqueLabels, env.roiData, env.dataset);

#ifdef WITH_PYTHON_H
			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
#endif

#ifdef CHECKTIMING
			if (Stopwatch::exclusive())
			{
				// Detailed timing - on the screen
				VERBOSLVL1 (env.get_verbosity_level(), Stopwatch::print_stats());

				// Details - also to a file
				VERBOSLVL1 (env.get_verbosity_level(),
					Stopwatch::save_stats(env.output_dir + "/" + p.fname_seg + "_nyxustiming.csv");
				);
			}
#endif
		}

#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
		{
			// Detailed timing - on the screen
			VERBOSLVL1 (env.get_verbosity_level(), Stopwatch::print_stats());

			// Details - also to a file
			VERBOSLVL1 (env.get_verbosity_level(),
				Stopwatch::save_stats(env.output_dir + "/inclusive_nyxustiming.csv");
			);
		}
#endif

		if (write_apache) 
		{
			// close arrow file after use
			auto [status, msg] = env.arrow_stream.close_arrow_file();
			if (!status) 
			{
				std::cerr << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}

#ifdef USE_GPU
		if (env.using_gpu())
		{
			if (! env.devCache.free_gpu_cache(
				env.devCache.gpu_roiclouds_2d,
				env.devCache.gpu_roicontours_2d,
				env.devCache.dev_realintens,
				env.devCache.dev_prereduce,
				env.devCache.gpu_featurestatebuf,
				env.devCache.dev_devicereduce_temp_storage,
				env.devCache.dev_imat1,
				env.devCache.dev_imat2,
				env.devCache.gabor_linear_image,
				env.devCache.gabor_result,
				env.devCache.gabor_linear_kernel,
				env.devCache.gabor_energy_image))
			{
				std::cerr << "error in free_gpu_cache()\n";
				return 1;
			}
		}
#endif

		return 0; // success
	}

}