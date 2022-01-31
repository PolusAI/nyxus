//
// This file is a collection of drivers of tiled TIFF file scanning from the FastLoader side
//

#include <string>
#include <vector>
#include <fast_loader/specialised_tile_loader/grayscale_tiff_tile_loader.h>
#include <map>
#include <array>
#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#endif
#include "virtual_file_tile_channel_loader.h"
#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

// Sanity
#ifdef _WIN32
#include<windows.h>
#endif

namespace Nyxus
{
	bool processIntSegImagePair (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		// Report the amount of free RAM
		unsigned long long freeRamAmt = getAvailPhysMemory();
		static unsigned long long initial_freeRamAmt = 0;
		if (initial_freeRamAmt == 0)
			initial_freeRamAmt = freeRamAmt;
		double memDiff = double(freeRamAmt) - double(initial_freeRamAmt);
		VERBOSLVL1(std::cout << std::setw(15) << freeRamAmt << " bytes free (" << "consumed=" << memDiff << ") ";)

		// Display (1) dataset progress info and (2) file pair info
		int digits = 2, k = std::pow(10.f, digits);
		float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
		VERBOSLVL1(std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n";)

		// Phase 1: gather ROI metrics
		VERBOSLVL1(std::cout << "Gathering ROI metrics\n";)
		gatherRoisMetrics (intens_fpath, label_fpath, num_FL_threads);	// Output - set of ROI labels, label-ROI cache mappings

		// Phase 2: process trivial-sized ROIs
		VERBOSLVL1(std::cout << "Processing trivial ROIs\n";)
		processTrivialRois (intens_fpath, label_fpath, num_FL_threads, theEnvironment.get_ram_limit());

		// Phase 3: process nontrivial (oversized) ROIs, if any
		VERBOSLVL1(std::cout << "Processing oversized ROIs\n";)
		processNontrivialRois (intens_fpath, label_fpath, num_FL_threads, theEnvironment.get_ram_limit());

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
			std::filesystem::path p_int(ifp), p_seg(lfp);
			theSegFname = p_seg.string(); 
			theIntFname = p_int.string(); 

			// Scan one label-intensity pair 
			theImLoader.open (theIntFname, theSegFname);

			{				
				STOPWATCH("Image scan/ImgScan/Scan/lightsteelblue", "\t=");
				ok = processIntSegImagePair (ifp, lfp, numFastloaderThreads, i, nf);		// Phased processing
			}

			if (ok == false)
			{
				std::cout << "processIntSegImagePair() returned an error code while processing file pair " << ifp << " and " << lfp << std::endl;
				return 1;
			}

			// Save the result for this intensity-label file pair
			if (save2csv)
				ok = save_features_2_csv (ifp, lfp, csvOutputDir);
			else
				ok = save_features_2_buffer(headerBuf, calcResultBuf, stringColBuf);
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

} // namespace Nyxus