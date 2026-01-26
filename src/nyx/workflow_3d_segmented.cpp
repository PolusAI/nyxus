#include <fstream>
#include <future>
#include <string>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

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
#include "helpers/fsystem.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "raw_image_loader.h"
#include "save_option.h"

namespace Nyxus
{
	bool processIntSegImagePair_3D (Environment & env, const std::string& intens_fpath, const std::string& label_fpath, size_t filepair_index, size_t t_index, const std::vector<std::string>& z_indices)
	{
		std::vector<int> trivRois, nontrivRois;

		// Report the amount of free RAM
		unsigned long long freeRamAmt = Nyxus::getAvailPhysMemory();
		static unsigned long long initial_freeRamAmt = 0;
		if (initial_freeRamAmt == 0)
			initial_freeRamAmt = freeRamAmt;
		unsigned long long memDiff = std::max (freeRamAmt, initial_freeRamAmt) - std::min(freeRamAmt, initial_freeRamAmt);
		char diffSign = freeRamAmt < initial_freeRamAmt ? '-' : '+';
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << std::setw(15) << Nyxus::virguler_ulong(freeRamAmt) << " bytes free (" << "consumed=" << diffSign << Nyxus::virguler_ulong(memDiff) << ") ")

		// Phase 1: gather ROI metrics
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Gathering ROI metrics\n");
		bool okGather = false;
		if (z_indices.size())
			okGather = gatherRoisMetrics_25D (env, filepair_index, intens_fpath, label_fpath, z_indices);
		else
			okGather = gatherRoisMetrics_3D (env, filepair_index, intens_fpath, label_fpath, t_index);
		if (!okGather)
		{
			std::string msg = "Error gathering ROI metrics from " + intens_fpath + " / " + label_fpath + "\n";
#ifndef WITH_PYTHON_H
			throw (std::runtime_error(msg));
#endif
			std::cerr << msg;
			return false;
		}

		// are there any ROI to extract features from ?
		if (env.uniqueLabels.size() == 0)
		{
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "warning: no ROIs in I:" + intens_fpath + " M:" + label_fpath);
			return true;
		}

		// prepare each ROI's feature value buffer
		for (auto lab : env.uniqueLabels)
		{
			LR& r = env.roiData[lab];
			r.initialize_fvals();
		}

#ifndef WITH_PYTHON_H
		// Dump ROI metrics to the output directory
		VERBOSLVL2 (env.get_verbosity_level(), dump_roi_metrics(env.dim(), env.output_dir, env.get_ram_limit(), label_fpath, env.uniqueLabels, env.roiData))
#endif		

		// Support of ROI blacklist
		fs::path fp(label_fpath);
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
			if (size_t roiFootprint = r.get_ram_footprint_estimate_3D (env.uniqueLabels.size()),
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
				nontrivRois.push_back(lab);
			}
			else
				trivRois.push_back(lab);
		}

		// Phase 2: process trivial-sized ROIs
		if (trivRois.size())
		{
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Processing trivial ROIs\n";)
			if (z_indices.size())
				processTrivialRois_25D (env, trivRois, intens_fpath, label_fpath, env.get_ram_limit(), z_indices);
			else
				processTrivialRois_3D (env, filepair_index, t_index, trivRois, intens_fpath, label_fpath, env.get_ram_limit());
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRois.size())
		{
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Processing oversized ROIs\n";)
			processNontrivialRois (env, nontrivRois, intens_fpath, label_fpath);
		}

		return true;
	}

	int processDataset_3D_segmented (
		Environment & env,
		const std::vector <Imgfile3D_layoutA>& intensFiles,
		const std::vector <Imgfile3D_layoutA>& labelFiles,
		int numReduceThreads,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		//********************** prescan ***********************

		// slide properties
		VERBOSLVL1(env.get_verbosity_level(), std::cout << "phase 0 (3D prescanning)\n");
		size_t nf = intensFiles.size();
		env.dataset.reset_dataset_props();

		for (size_t i = 0; i < nf; i++)
		{
			// slide file names
			SlideProps& p = env.dataset.dataset_props.emplace_back (intensFiles[i].fdir + intensFiles[i].fname, labelFiles[i].fdir + labelFiles[i].fname);

			// slide metrics
			VERBOSLVL1(env.get_verbosity_level(), std::cout << "prescanning " << p.fname_int);
			if (! scan_slide_props(p, 3, env.anisoOptions, env.resultOptions.need_annotation()))
			{
				VERBOSLVL1(env.get_verbosity_level(), std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return 1;
			}
			VERBOSLVL1(env.get_verbosity_level(), std::cout << "\t " << p.slide_w << " W x " << p.slide_h << " H x " << p.volume_d << " D\tmax ROI " << p.max_roi_w << " x " << p.max_roi_h << "\tmin - max I " << Nyxus::virguler_real(p.min_preroi_inten) << " - " << Nyxus::virguler_real(p.max_preroi_inten) << "\t" << p.lolvl_slide_descr << "\n");
		}

		// update whole dataset's summary
		env.dataset.update_dataset_props_extrema();

		VERBOSLVL1(env.get_verbosity_level(), std::cout << "\t finished prescanning \n");

		//***** feature extraction

		// One-time initialization
		init_slide_rois (env.uniqueLabels, env.roiData);

		bool writeApache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize an Arrow writer if needed
		if (writeApache)
		{
			env.arrow_stream = ArrowOutputStream();
			auto [status, msg] = env.arrow_stream.create_arrow_file(
				saveOption,
				get_arrow_filename(outputPath, env.nyxus_result_fname, saveOption),
				Nyxus::get_header(env));

			if (!status)
			{
				std::cout << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		// iterate intensity-mask pairs
		for (size_t i=0; i<nf; i++)
		{
			// iterate time frames
			for (size_t t=0; t < env.dataset.dataset_props[i].inten_time; t++)
			{
				// Clear slide's ROI labels and cache allocated the previous image
				clear_slide_rois (env.uniqueLabels, env.roiData);

				auto& ifile = intensFiles[i],	// intensity
					& mfile = labelFiles[i];	// mask

			   // Do phased processing: prescan, trivial ROI processing, oversized ROI processing
			   // Expecting 2 cases of intensFiles[i].z_indices :
			   // -- non-empty indicating a 2.5D case (aka layoutA)
			   // -- empty indicating a 3D case (.nii, .dcm, etc)
			   
				// Display (1) dataset progress info and (2) file pair info
				int digits = 2, k = (int)std::pow(10.f, digits);
				float perCent = float(i * 100 * k / nf) / float(k);
				VERBOSLVL1(env.get_verbosity_level(), std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << ifile.fname << " SEG: " << mfile.fname << " T:" << t << "\n")
					
				bool ok = processIntSegImagePair_3D (env, ifile.fdir+ifile.fname, mfile.fdir+mfile.fname, i, t, intensFiles[i].z_indices);
				if (ok == false)
				{
					std::cerr << "processIntSegImagePair() returned an error code while processing file pair " << ifile.fname << " - " << mfile.fname << '\n';
					return 1;
				}

				// Output features
				if (writeApache)
				{
					auto [status, msg] = env.arrow_stream.write_arrow_file(Nyxus::get_feature_values(env.theFeatureSet, env.uniqueLabels, env.roiData, env.dataset));
					if (!status)
					{
						std::cout << "Error writing Arrow file: " << msg.value() << std::endl;
						return 2;
					}
				}
				else
					if (saveOption == SaveOption::saveCSV)
					{
						if (!save_features_2_csv(env, ifile.fname, mfile.fname, outputPath, t, env.resultOptions.need_aggregation()))
						{
							std::cout << "error saving results to CSV file, details: " << __FILE__ << ":" << __LINE__ << std::endl;
							return 2;
						}
					}
					else
					{
						if (!save_features_2_buffer(env.theResultsCache, env, t))
						{
							std::cout << "error saving results to a buffer, details: " << __FILE__ << ":" << __LINE__ << std::endl;
							return 2;
						}
					}

				// Save nested ROI related info of this image
				if (env.nestedOptions.defined())
					save_nested_roi_info(nestedRoiData, env.uniqueLabels, env.roiData, env.dataset);

				#ifdef WITH_PYTHON_H
				// Allow keyboard interrupt.
				if (PyErr_CheckSignals() != 0)
				{
					sureprint("\nAborting per user input\n");
					throw pybind11::error_already_set();
				}
				#endif
			} //- time frames
		} //- inten-mask pairs

		if (writeApache)
		{
			// close arrow file after use
			auto [status, msg] = env.arrow_stream.close_arrow_file();
			if (!status)
			{
				std::cout << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}

		return 0; // success
	}


}