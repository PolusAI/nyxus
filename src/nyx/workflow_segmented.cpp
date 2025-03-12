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

#ifdef USE_GPU
	#include "gpucache.h"
#endif

namespace Nyxus
{

	bool processIntSegImagePair (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

		// Timing block (image scanning)
		{
			{ STOPWATCH("Image scan1/scan1/s1/#aabbcc", "\t=");

			VERBOSLVL2(
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
				int digits = 2, k = std::pow(10.f, digits);
				float perCent = float(filepair_index) * 100. * k / float(tot_num_filepairs) / float(k);
				VERBOSLVL1(std::cout << "[ " << filepair_index << " = " << std::setw(digits + 2) << perCent << "% ]\t" << intens_fpath << "\n")
				VERBOSLVL2(std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n")
			}

			{ STOPWATCH("Image scan2a/scan2a/s2a/#aabbcc", "\t=");
				// Phase 1: gather ROI metrics
				VERBOSLVL2(std::cout << "Gathering ROI metrics\n");
				bool okGather = gatherRoisMetrics(intens_fpath, label_fpath, theImLoader);	// Output - set of ROI labels, label-ROI cache mappings
				if (!okGather)
				{
					std::string msg = "Error gathering ROI metrics from " + intens_fpath + " / " + label_fpath + "\n";
					std::cerr << msg;
					throw (std::runtime_error(msg));
					return false;
				}

				// Any ROIs in this slide? (Such slides may exist, it's normal.)
				if (uniqueLabels.size() == 0)
				{
					return true;
				}
			}

			{ STOPWATCH("Image scan2b/scan2b/s2b/#aabbcc", "\t=");
				// Allocate each ROI's feature value buffer
				for (auto lab : uniqueLabels)
				{
					LR& r = roiData[lab];
					r.initialize_fvals();
				}

				#ifndef WITH_PYTHON_H
				// Dump ROI metrics to the output directory
				VERBOSLVL2(dump_roi_metrics(label_fpath))
				#endif		
			}

			{ STOPWATCH("Image scan3/scan3/s3/#aabbcc", "\t=");
				// Support of ROI blacklist
				fs::path fp(theSegFname);
				std::string shortSegFname = fp.stem().string() + fp.extension().string();

				// Distribute ROIs among phases
				for (auto lab : uniqueLabels)
				{
					LR& r = roiData[lab];

					// Skip blacklisted ROI
					if (theEnvironment.roi_is_blacklisted(shortSegFname, lab))
					{
						r.blacklisted = true;
						VERBOSLVL2(std::cout << "Skipping blacklisted ROI " << lab << " for mask " << shortSegFname << "\n");
						continue;
					}

					// Examine ROI's memory footprint
					if (size_t roiFootprint = r.get_ram_footprint_estimate(),
						ramLim = theEnvironment.get_ram_limit();
						roiFootprint >= ramLim)
					{
						VERBOSLVL2(
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
			VERBOSLVL2(std::cout << "Processing trivial ROIs\n";)
				processTrivialRois(trivRoiLabels, intens_fpath, label_fpath, num_FL_threads, theEnvironment.get_ram_limit());
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRoiLabels.size())
		{
			VERBOSLVL2(std::cout << "Processing oversized ROIs\n";)
				processNontrivialRois(nontrivRoiLabels, intens_fpath, label_fpath, num_FL_threads);
		}

		return true;
	}

	int processDataset_2D_segmented (
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		int numFastloaderThreads,
		int numSensemakerThreads,
		int numReduceThreads,
		int min_online_roi_size,
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

		VERBOSLVL1(std::cout << "phase 0 (prescanning)\n");

		LR::reset_dataset_props();
		LR::dataset_props.resize(nf);
		for (size_t i = 0; i < nf; i++)
		{
			// slide file names
			SlideProps& p = LR::dataset_props[i];
			p.fname_int = intensFiles[i];
			p.fname_seg = labelFiles[i];

			// slide metrics
			VERBOSLVL1(std::cout << "prescanning " << p.fname_int);
			if (! scan_slide_props(p))
			{
				VERBOSLVL1(std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return 1;
			}
			VERBOSLVL1(std::cout << "\tmax ROI " << p.max_roi_w << " x " << p.max_roi_h << "\tmin-max I " << p.min_preroi_inten << "-" << p.max_preroi_inten << "\n");
		}

		// global properties
		LR::dataset_max_combined_roicloud_len = 0;
		LR::dataset_max_n_rois = 0;
		LR::dataset_max_roi_area = 0;
		LR::dataset_max_roi_w = 0;
		LR::dataset_max_roi_h = 0;

		for (SlideProps& p : LR::dataset_props)
		{
			size_t sup_s_n = p.n_rois * p.max_roi_area;
			LR::dataset_max_combined_roicloud_len = (std::max)(LR::dataset_max_combined_roicloud_len, sup_s_n);

			LR::dataset_max_n_rois = (std::max)(LR::dataset_max_n_rois, p.n_rois);
			LR::dataset_max_roi_area = (std::max)(LR::dataset_max_roi_area, p.max_roi_area);

			LR::dataset_max_roi_w = (std::max)(LR::dataset_max_roi_w, p.max_roi_w);
			LR::dataset_max_roi_h = (std::max)(LR::dataset_max_roi_h, p.max_roi_h);
		}

		VERBOSLVL1(std::cout << "\t finished prescanning \n");

		//********************** allocate the GPU cache ***********************

#ifdef USE_GPU
		// what parts of GPU cache we need to bother about ?
		bool needContour = ContourFeature::required(theFeatureSet),
			needErosion = ErosionPixelsFeature::required(theFeatureSet),
			needGabor = GaborFeature::required(theFeatureSet),
			needImoments = Imoms2D_feature::required(theFeatureSet),
			needSmoments = Smoms2D_feature::required(theFeatureSet),
			needMoments = needImoments || needSmoments;

		// whole slide's contour is just 4 vertices long
		size_t kontrLen = Nyxus::theEnvironment.singleROI ? 4 : LR::dataset_max_combined_roicloud_len;

		if (theEnvironment.using_gpu())
		{
			// allocate
			VERBOSLVL1(std::cout << "allocating GPU cache \n");

			if (!NyxusGpu::allocate_gpu_cache(
				// out
				NyxusGpu::gpu_roiclouds_2d,
				NyxusGpu::gpu_roicontours_2d,
				&NyxusGpu::dev_realintens,
				&NyxusGpu::dev_prereduce,
				NyxusGpu::gpu_featurestatebuf,
				NyxusGpu::devicereduce_temp_storage_szb,
				&NyxusGpu::dev_devicereduce_temp_storage,
				NyxusGpu::gpu_batch_len,
				&NyxusGpu::dev_imat1,
				&NyxusGpu::dev_imat2,
				NyxusGpu::gabor_linear_image,
				NyxusGpu::gabor_linear_kernel,
				NyxusGpu::gabor_result,
				NyxusGpu::gabor_energy_image,
				// in
				needContour,
				needErosion,
				needGabor,
				needMoments,
				LR::dataset_max_combined_roicloud_len, // desired totCloLen,
				kontrLen, // desired totKontLen,
				LR::dataset_max_n_rois,	// labels.size()
				LR::dataset_max_roi_area,
				LR::dataset_max_roi_w,
				LR::dataset_max_roi_h,
				GaborFeature::f0_theta_pairs.size(),
				GaborFeature::n
			))	// we need max ROI area inside the function to calculate the batch size if 'dataset_max_combined_roicloud_len' doesn't fit in RAM
			{
				std::cerr << "error in " << __FILE__ << ":" << __LINE__ << "\n";
				return 1;
			}

			VERBOSLVL1(std::cout << "\t ---done allocating GPU cache \n");
		}
#endif
		} // prescan timing

		// One-time initialization
		init_slide_rois();

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize arrow writer if needed
		if (write_apache) {

			theEnvironment.arrow_stream = ArrowOutputStream();
			auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file(
				saveOption,
				get_arrow_filename(outputPath, theEnvironment.nyxus_result_fname, saveOption),
				Nyxus::get_header(theFeatureSet.getEnabledFeatures()));

			if (!status) {
				std::cout << "Error creating Arrow file: " << msg.value() << std::endl;
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
			clear_slide_rois();

			auto& ifp = intensFiles[i],
				& lfp = labelFiles[i];

			// Cache the file names to be picked up by labels to know their file origin
			fs::path p_int(ifp), p_seg(lfp);
			theSegFname = p_seg.string();
			theIntFname = p_int.string();

			// Scan one label-intensity pair 
			SlideProps& p = LR::dataset_props[i];
			ok = theImLoader.open(p);
			if (ok == false)
			{
				std::cerr << "Terminating\n";
				return 1;
			}

			// Do phased processing: prescan, trivial ROI processing, oversized ROI processing
			ok = processIntSegImagePair(ifp, lfp, numFastloaderThreads, i, nf);

			if (ok == false)
			{
				std::cout << "processIntSegImagePair() returned an error code while processing file pair " << ifp << " and " << lfp << std::endl;
				return 1;
			}

			if (write_apache) {

				auto [status, msg] = theEnvironment.arrow_stream.write_arrow_file(Nyxus::get_feature_values());

				if (!status) {
					std::cout << "Error writing Arrow file: " << msg.value() << std::endl;
					return 2;
				}
			}
			else if (saveOption == SaveOption::saveCSV)
			{
				ok = save_features_2_csv(ifp, lfp, outputPath);

				if (ok == false)
				{
					std::cout << "save_features_2_csv() returned an error code" << std::endl;
					return 2;
				}
			}
			else
			{
				ok = save_features_2_buffer(theResultsCache);

				if (ok == false)
				{
					std::cout << "save_features_2_buffer() returned an error code" << std::endl;
					return 2;
				}
			}

			theImLoader.close();

			// Save nested ROI related info of this image
			if (theEnvironment.nestedOptions.defined())
				save_nested_roi_info(nestedRoiData, uniqueLabels, roiData);

#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
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
				VERBOSLVL1(Stopwatch::print_stats());

				// Details - also to a file
				VERBOSLVL1(
					fs::path p(theSegFname);
				Stopwatch::save_stats(theEnvironment.output_dir + "/" + p.stem().string() + "_nyxustiming.csv");
				);
			}
#endif
		}

#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
		{
			// Detailed timing - on the screen
			VERBOSLVL1(Stopwatch::print_stats());

			// Details - also to a file
			VERBOSLVL1(
				fs::path p(theSegFname);
			Stopwatch::save_stats(theEnvironment.output_dir + "/inclusive_nyxustiming.csv");
			);
		}
#endif

		if (write_apache) {
			// close arrow file after use
			auto [status, msg] = theEnvironment.arrow_stream.close_arrow_file();
			if (!status) {
				std::cout << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}

#ifdef USE_GPU
		if (theEnvironment.using_gpu())
		{
			if (!NyxusGpu::free_gpu_cache(
				NyxusGpu::gpu_roiclouds_2d,
				NyxusGpu::gpu_roicontours_2d,
				NyxusGpu::dev_realintens,
				NyxusGpu::dev_prereduce,
				NyxusGpu::gpu_featurestatebuf,
				NyxusGpu::dev_devicereduce_temp_storage,
				NyxusGpu::dev_imat1,
				NyxusGpu::dev_imat2,
				NyxusGpu::gabor_linear_image,
				NyxusGpu::gabor_result,
				NyxusGpu::gabor_linear_kernel,
				NyxusGpu::gabor_energy_image))
			{
				std::cerr << "error in free_gpu_cache()\n";
				return 1;
			}
		}
#endif

		return 0; // success
	}

}