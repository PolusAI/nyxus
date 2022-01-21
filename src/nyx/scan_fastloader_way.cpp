//
// This file is a collection of drivers of tiled TIFF file scanning from the FastLoader side
//

#include <string>
#include <vector>
#include <fast_loader/specialised_tile_loader/grayscale_tiff_tile_loader.h>
#include <map>
#include <array>
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
	bool scanFilePair (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		// Report the amount of free RAM
		unsigned long long freeRamAmt = getAvailPhysMemory();
		static unsigned long long initial_freeRamAmt = 0;
		if (initial_freeRamAmt == 0)
			initial_freeRamAmt = freeRamAmt;
		double memDiff = double(freeRamAmt) - double(initial_freeRamAmt);
		std::cout << std::setw(15) << freeRamAmt << " bytes free (" << "consumed=" << memDiff << ") ";

		// Display (1) dataset progress info and (2) file pair info
		int digits = 2, k = std::pow(10.f, digits);
		float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
		std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << intens_fpath << "\n";

		int lvl = 0;	// Pyramid level

		// File #1 (intensity)
		GrayscaleTiffTileLoader<uint32_t> I(num_FL_threads, intens_fpath);

		auto th = I.tileHeight(lvl),
			tw = I.tileWidth(lvl),
			td = I.tileDepth(lvl);
		auto tileSize = th * tw;

		auto fh = I.fullHeight(lvl);
		auto fw = I.fullWidth(lvl);
		auto fd = I.fullDepth(lvl);

		auto ntw = I.numberTileWidth(lvl);
		auto nth = I.numberTileHeight(lvl);
		auto ntd = I.numberTileDepth(lvl);

		// File #2 (labels)
		GrayscaleTiffTileLoader<uint32_t> L(num_FL_threads, label_fpath);

		// -- check whole file consistency
		if (fh != L.fullHeight(lvl) || fw != L.fullWidth(lvl) || fd != L.fullDepth(lvl))
		{
			std::cout << "\terror: mismatch in full height, width, or depth";
			return false;
		}

		// -- check tile consistency
		if (th != L.tileHeight(lvl) || tw != L.tileWidth(lvl) || td != L.tileDepth(lvl))
		{
			std::cout << "\terror: mismatch in tile height, width, or depth";
			return false;
		}

		// Read the TIFF tile by tile 
		// 
		// -- allocate the tile buffer
		std::shared_ptr<std::vector<uint32_t>> ptrI = std::make_shared<std::vector<uint32_t>>(tileSize);
		std::shared_ptr<std::vector<uint32_t>> ptrL = std::make_shared<std::vector<uint32_t>>(tileSize);

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntw; col++)
			{
#ifdef CHECKTIMING
				std::cout << "\tt." << row * ntw + col + 1 << "/" << nth * ntw;
#endif	

				// --Timing
				std::chrono::time_point<std::chrono::system_clock> start, end;
				start = std::chrono::system_clock::now();

				I.loadTileFromFile(ptrI, row, col, 0 /*layer*/, lvl);
				L.loadTileFromFile(ptrL, row, col, 0 /*layer*/, lvl);
				auto& dataI = *ptrI;
				auto& dataL = *ptrL;

				// --Timing
				end = std::chrono::system_clock::now();
				std::chrono::duration<double, std::micro> elapsedTile = end - start;

				// --Timing
				start = std::chrono::system_clock::now();

				for (unsigned long i = 0; i < tileSize; i++)
				{
					auto label = dataL[i];
					if (label != 0 && dataI[i] != 0)
					{
						int y = row * th + i / tw,
							x = col * tw + i % tw;

						// Collapse all the labels to one if single-ROI mde is requested
						if (theEnvironment.singleROI)
							label = 1;

						update_label(x, y, label, dataI[i]);
					}
				}

				// --Timing
				end = std::chrono::system_clock::now();
				std::chrono::duration<double, std::micro> elapsedCalc = end - start;

				// --Time ratio
#ifdef CHECKTIMING
				std::cout << " F/T: " << elapsedCalc.count() << " / " << elapsedTile.count() << " = " << elapsedCalc.count() / elapsedTile.count() << " x " << std::endl;
#endif

				// Show stayalive progress info
				if (cnt++ % 4 == 0)
					std::cout << "\t"
					//--Harmful in Python output scenarios-- << BEGINFORMAT_RED	
					<< int((row * nth + col) * 100 / float(nth * ntw) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs"
					//--Harmful in Python output scenarios-- << ENDFORMAT	
					<< "\n";

				totalImgScanTime += elapsedTile.count();
				totalFeatureReduceTime += elapsedCalc.count();

			}

		// Show stayalive progress info
		std::cout << "\t" << "100%\t" << uniqueLabels.size() << " ROIs\n";

		return true;
	}

	bool processIntSegImagePair (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		// Report the amount of free RAM
		unsigned long long freeRamAmt = getAvailPhysMemory();
		static unsigned long long initial_freeRamAmt = 0;
		if (initial_freeRamAmt == 0)
			initial_freeRamAmt = freeRamAmt;
		double memDiff = double(freeRamAmt) - double(initial_freeRamAmt);
		std::cout << std::setw(15) << freeRamAmt << " bytes free (" << "consumed=" << memDiff << ") ";

		// Display (1) dataset progress info and (2) file pair info
		int digits = 2, k = std::pow(10.f, digits);
		float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
		std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n";

		// Phase 1: gather ROI metrics
		std::cout << "Gathering ROI metrics\n";
		gatherRoisMetrics (intens_fpath, label_fpath, num_FL_threads);	// Output - set of ROI labels, label-ROI cache mappings

		// Phase 2: process trivial-sized ROIs
		std::cout << "Processing trivial ROIs\n";
		processTrivialRois (intens_fpath, label_fpath, num_FL_threads, theEnvironment.get_ram_limit());

		// Phase 3: process nontrivial (oversized) ROIs, if any
		std::cout << "Processing oversized ROIs\n";
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
				std::cout << "scanFilePair() returned an error code while processing file pair " << ifp << " and " << lfp << std::endl;
				return 1;
			}

			// --Timing
			std::chrono::time_point<std::chrono::system_clock> startRed, endRed;
			startRed = std::chrono::system_clock::now();

			// --Timing
			endRed = std::chrono::system_clock::now();
			std::chrono::duration<double, Stopwatch::Unit> elapsedRed = endRed - startRed;
			totalFeatureReduceTime += elapsedRed.count();

			// Save the result for this intensity-label file pair
			if (save2csv)
				ok = save_features_2_csv (ifp, lfp, csvOutputDir);
			else
				ok = save_features_2_buffer(calcResultBuf);
			if (ok == false)
			{
				std::cout << "save_features_2_csv() returned an error code" << std::endl;
				return 2;
			}

			theImLoader.close();
		}

#ifdef CHECKTIMING
		// General timing
		//---	end = std::chrono::system_clock::now();
		//---	std::chrono::duration<double, std::micro> elapsed = end - start;
		//---	double secs = elapsed.count() / 1e6;
		//---	std::cout << "Elapsed time (s) " << secs << std::endl;
		std::cout
			<< "Total image scan time [" << Stopwatch::UnitString << "]: " << totalImgScanTime
			<< "\n\t+\nTotal feature reduce time [" << Stopwatch::UnitString << "]: " << totalFeatureReduceTime
			<< "\n\t=\nScan to reduce ratio: " << totalImgScanTime / totalFeatureReduceTime
			<< std::endl;

		// Detailed timing
		Stopwatch::print_stats();
		Stopwatch::save_stats(theEnvironment.output_dir + "/nyxus_timing.csv");
#endif

		return 0; // success
	}

} // namespace Nyxus