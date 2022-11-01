//
// This file is a collection of drivers of tiled TIFF file scanning from the FastLoader side
//
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <map>
#include <array>
#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#endif
#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

// Sanity
#ifdef _WIN32
#include<windows.h>
#endif

namespace Nyxus
{
	bool processIntSegImagePair_2d (const std::string& intens_fpath, const std::string& label_fpath, int z, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

		// Report the amount of free RAM
		unsigned long long freeRamAmt = getAvailPhysMemory();
		static unsigned long long initial_freeRamAmt = 0;
		if (initial_freeRamAmt == 0)
			initial_freeRamAmt = freeRamAmt;
		double memDiff = double(freeRamAmt) - double(initial_freeRamAmt);
		VERBOSLVL1(std::cout << std::setw(15) << freeRamAmt << " bytes free (" << "consumed=" << memDiff << ") ";)

		// Display (1) dataset progress info and (2) file pair info
		int digits = 2, k = (int) std::pow(10.f, digits);
		float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
		VERBOSLVL1(std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n";)

		// Phase 1: gather ROI metrics
		VERBOSLVL1(std::cout << "Gathering 2D ROI metrics\n";)
		gatherRoisMetrics_2d (intens_fpath, label_fpath, num_FL_threads, z);	// Output - set of ROI labels, label-ROI cache mappings

		// Allocate each ROI's feature value buffer
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];
			r.initialize_fvals();
		}

		// Dump ROI metrics
		VERBOSLVL4(dump_roi_metrics(label_fpath))	// dumps to file in the output directory

		// Distribute ROIs among phases
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];
			size_t footprint = r.get_ram_footprint_estimate();
			if (footprint >= theEnvironment.get_ram_limit())
			{
				VERBOSLVL2(std::cout << ">>> Skipping non-trivial ROI " << lab << " (area=" << r.aux_area << " px, footprint=" << footprint << " b"
					<< " w=" << r.aabb.get_width() << " h=" << r.aabb.get_height() << " sz_Pixel2=" << sizeof(Pixel2)
					<< ")\n";)
					nontrivRoiLabels.push_back(lab);
			}
			else
				trivRoiLabels.push_back(lab);
		}

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			VERBOSLVL1(std::cout << "Processing trivial 2D ROIs\n";)
			processTrivialRois_2d (trivRoiLabels, intens_fpath, label_fpath, z, theEnvironment.get_ram_limit());
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRoiLabels.size())
		{
			VERBOSLVL1(std::cout << "Processing oversized ROIs\n";)
			processNontrivialRois (nontrivRoiLabels, intens_fpath, label_fpath, num_FL_threads);
		}

		return true;
	}

	bool processIntSegImagePair_3d (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

		// Report the amount of free RAM
		unsigned long long freeRamAmt = getAvailPhysMemory();
		static unsigned long long initial_freeRamAmt = 0;
		if (initial_freeRamAmt == 0)
			initial_freeRamAmt = freeRamAmt;
		double memDiff = double(freeRamAmt) - double(initial_freeRamAmt);
		VERBOSLVL1(std::cout << std::setw(15) << freeRamAmt << " bytes free (" << "consumed=" << memDiff << ") ");

		// Display (1) dataset progress info and (2) file pair info
		int digits = 2, k = (int) std::pow(10.f, digits);
		float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
		VERBOSLVL1(std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n";)

		// Phase 1: gather ROI metrics
		VERBOSLVL1(std::cout << "Gathering 3D ROI metrics\n";)
		gatherRoisMetrics_3d (intens_fpath, label_fpath, num_FL_threads);	// Output - set of ROI labels, label-ROI cache mappings

		// Allocate each ROI's feature value buffer
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];
			r.initialize_fvals();
		}

		// Dump ROI metrics
		VERBOSLVL4(dump_roi_metrics(label_fpath))	// dumps to file in the output directory

		// Distribute ROIs among phases
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];
			size_t footprint = r.get_ram_footprint_estimate();
			if (footprint >= theEnvironment.get_ram_limit())
			{
				VERBOSLVL2(std::cout << ">>> Skipping non-trivial ROI " << lab << " (area=" << r.aux_area << " px, footprint=" << footprint << " b"
					<< " w=" << r.aabb.get_width() << " h=" << r.aabb.get_height() << " sz_Pixel2=" << sizeof(Pixel2)
					<< ")\n";)
					nontrivRoiLabels.push_back(lab);
			}
			else
				trivRoiLabels.push_back(lab);
		}

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			VERBOSLVL1(std::cout << "Processing trivial 3D ROIs\n");
			processTrivialRois_3d (trivRoiLabels, intens_fpath, label_fpath, theEnvironment.get_ram_limit());
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRoiLabels.size())
		{
			VERBOSLVL1(std::cout << "Processing oversized ROIs\n";)
			processNontrivialRois(nontrivRoiLabels, intens_fpath, label_fpath, num_FL_threads);
		}

		return true;
	}

	int processDataset(
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		int numFastloaderThreads,
		int numSensemakerThreads,
		int numReduceThreads,
		int min_online_roi_size,
		bool save2csv,
		const std::string& csvOutputDir)
	{
		bool ok = true;

		auto nf = intensFiles.size();
		for (int i = 0; i < nf; i++)
		{
			// Clear ROI label list, ROI data, etc.
			clear_feature_buffers();

			auto& ifp = intensFiles[i],
				& lfp = labelFiles[i];

			// Cache the file names to be picked up by labels to know their file origin
			fs::path p_int(ifp), p_seg(lfp);
			theSegFname = p_seg.string(); 
			theIntFname = p_int.string(); 

			// Open the label-intensity image pair 
			ok = theImLoader.open (theIntFname, theSegFname);
			if (ok == false)
			{
				std::cout << "Error loading image data. Terminating\n";
				return 1;
			}

			// Reset the 2D and 3D flags in the featureset
			theFeatureSet.need_3d(false);
			theFeatureSet.need_3d(false);

			// Set the flag that 2D features were actually calculated based on the image dimensionality
			theFeatureSet.need_2d(true);

			// Extract 2D features
			ok = processIntSegImagePair_2d (ifp, lfp, theEnvironment.layerZ, numFastloaderThreads, i, nf);
			if (ok == false)
			{
				std::cout << "Error extracting 2D features of image pair " << ifp << " and " << lfp << "\n";
				return 1;
			}

			// Extract 3D features
			if (theImLoader.get_num_layers() > 1)
			{
				// Set the flag that 3D features were actually calculated based on the image dimensionality
				theFeatureSet.need_3d(true);

				ok = processIntSegImagePair_3d (ifp, lfp, numFastloaderThreads, i, nf);
				if (ok == false)
				{
					std::cout << "Error extracting 3D features of image pair " << ifp << " and " << lfp << "\n";
					return 1;
				}
			}

			// Save the result for this intensity-label file pair
			if (save2csv)
				ok = save_features_2_csv (ifp, lfp, csvOutputDir);
			else
				ok = save_features_2_buffer(theResultsCache);
			if (ok == false)
			{
				std::cout << "save_features_2_csv() returned an error code" << std::endl;
				return 2;
			}

			theImLoader.close();

			#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();
			#endif
		}

#ifdef CHECKTIMING
		// Detailed timing
		VERBOSLVL1(Stopwatch::print_stats();)
		VERBOSLVL1(Stopwatch::save_stats(theEnvironment.output_dir + "/nyxus_timing.csv");)
#endif

		return 0; // success
	}

	void dump_roi_metrics(const std::string & label_fpath)
	{
		fs::path pseg (label_fpath);
		std::string fpath = theEnvironment.output_dir + "/roi_metrics_" + pseg.stem().string() + ".csv";
		std::cout << "Dumping ROI metrics to " << fpath << " ...\n";

		std::ofstream f (fpath);

		// header
		f << "label, area, minx, miny, maxx, maxy, width, height, min_intens, max_intens, size_bytes, size_class, host_tiles \n";

		// sort labels
		std::vector<int>  sortedLabs { uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(sortedLabs.begin(), sortedLabs.end());
		// body
		for (auto lab : sortedLabs)
		{
			LR& r = roiData[lab];
			auto szb = r.get_ram_footprint_estimate();
			std::string ovsz = szb < theEnvironment.get_ram_limit() ? "T" : "OVERSIZE";
			f << lab << ", "
				<< r.aux_area << ", "
				<< r.aabb.get_xmin() << ", "
				<< r.aabb.get_ymin() << ", "
				<< r.aabb.get_xmax() << ", "
				<< r.aabb.get_ymax() << ", "
				<< r.aabb.get_width() << ", "
				<< r.aabb.get_height() << ", "
				<< r.aux_min << ", "
				<< r.aux_max << ", "
				<< szb << ", "
				<< ovsz << ", ";
			// host tile indices
			int ti = 0;
			for (auto tIdx : r.host_tiles)
			{
				if (ti++)
					f << "|";
				f << tIdx;
			}
			f << "\n";
		}

		f.flush();
		std::cout << "... done\n";
	}

} // namespace Nyxus
