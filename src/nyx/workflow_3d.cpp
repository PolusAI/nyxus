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
	bool processIntSegImagePair_3D(const std::string& intens_fpath, const std::string& label_fpath, size_t filepair_index, size_t tot_num_filepairs, const std::vector<std::string>& z_indices)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

		{ STOPWATCH("Image scan1/scan1/s1/#aabbcc", "\t=");
			// Report the amount of free RAM
			unsigned long long freeRamAmt = Nyxus::getAvailPhysMemory();
			static unsigned long long initial_freeRamAmt = 0;
			if (initial_freeRamAmt == 0)
				initial_freeRamAmt = freeRamAmt;
			double memDiff = double(freeRamAmt) - double(initial_freeRamAmt);
			VERBOSLVL1(std::cout << std::setw(15) << freeRamAmt << " bytes free (" << "consumed=" << memDiff << ") ")

			// Display (1) dataset progress info and (2) file pair info
			int digits = 2, k = (int) std::pow(10.f, digits);
			float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
			VERBOSLVL1(std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n")
		}

		{ STOPWATCH("Image scan2a/scan2a/s2a/#aabbcc", "\t=");
		// Phase 1: gather ROI metrics
		VERBOSLVL2(std::cout << "Gathering ROI metrics\n");
		bool okGather = gatherRoisMetrics_3D(intens_fpath, label_fpath, z_indices);
		if (!okGather)
		{
			std::string msg = "Error gathering ROI metrics from " + intens_fpath + " / " + label_fpath + "\n";
			std::cerr << msg;
			throw (std::runtime_error(msg));
			return false;
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
		fs::path fp(label_fpath);
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
			if (size_t roiFootprint = r.get_ram_footprint_estimate_3D(),
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

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			VERBOSLVL2(std::cout << "Processing trivial ROIs\n";)
				processTrivialRois_3D(trivRoiLabels, intens_fpath, label_fpath, theEnvironment.get_ram_limit(), z_indices);
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRoiLabels.size())
		{
			VERBOSLVL2(std::cout << "Processing oversized ROIs\n";)
				processNontrivialRois(nontrivRoiLabels, intens_fpath, label_fpath);
		}

		return true;
	}

	int processDataset_3D_segmented (
		const std::vector <Imgfile3D_layoutA>& intensFiles,
		const std::vector <Imgfile3D_layoutA>& labelFiles,
		int numReduceThreads,
		int min_online_roi_size,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
			Stopwatch::reset();
#endif		

		// One-time initialization
		init_slide_rois();

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize arrow writer if needed
		if (write_apache)
		{
			theEnvironment.arrow_stream = ArrowOutputStream();
			auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file(
				saveOption,
				get_arrow_filename(outputPath, theEnvironment.nyxus_result_fname, saveOption),
				Nyxus::get_header(theFeatureSet.getEnabledFeatures()));

			if (!status)
			{
				std::cout << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		bool ok = true;

		// Iterate intensity-mask image pairs
		size_t nf = intensFiles.size();
		for (size_t i = 0; i < nf; i++)
		{
#ifdef CHECKTIMING
			if (Stopwatch::exclusive())
				Stopwatch::reset();
#endif

			// Clear slide's ROI labels and cache allocated the previous image
			clear_slide_rois();

			auto& ifile = intensFiles[i],	// intensity
				& mfile = labelFiles[i];	// mask

			// Do phased processing: prescan, trivial ROI processing, oversized ROI processing
			ok = processIntSegImagePair_3D(ifile.fdir + ifile.fname, mfile.fdir + mfile.fname, i, nf, intensFiles[i].z_indices);
			if (ok == false)
			{
				std::cerr << "processIntSegImagePair() returned an error code while processing file pair " << ifile.fname << " - " << mfile.fname << '\n';
				return 1;
			}

			// Output features
			if (write_apache) {

				auto [status, msg] = theEnvironment.arrow_stream.write_arrow_file(Nyxus::get_feature_values());

				if (!status) {
					std::cout << "Error writing Arrow file: " << msg.value() << std::endl;
					return 2;
				}
			}
			else if (saveOption == SaveOption::saveCSV)
			{
				ok = save_features_2_csv(ifile.fname, mfile.fname, outputPath);

				if (ok == false)
				{
					std::cout << "save_features_2_csv() returned an error code" << std::endl;
					return 2;
				}
			}
			else {
				ok = save_features_2_buffer(theResultsCache);

				if (ok == false)
				{
					std::cout << "save_features_2_buffer() returned an error code" << std::endl;
					return 2;
				}
			}

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

		} //- pairs

#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
		{
			// Detailed timing - on the screen
			VERBOSLVL1(Stopwatch::print_stats());

			// Details - also to a file
			VERBOSLVL3(
				fs::path p(theSegFname);
			Stopwatch::save_stats(theEnvironment.output_dir + "/inclusive_nyxustiming.csv");
			);
		}
#endif

		if (write_apache)
		{
			// close arrow file after use
			auto [status, msg] = theEnvironment.arrow_stream.close_arrow_file();
			if (!status)
			{
				std::cout << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}

		return 0; // success
	}


}