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
#include "sensemaker.h"


// Sanity
#ifndef __unix
#include<windows.h>
#endif

// Timing
#include <chrono>
double totalTileLoadTime = 0.0, totalFeatureReduceTime = 0.0;

//--Harmful in Python output scenarios-- #define BEGINFORMAT_RED "\033[1;34m" 
//--Harmful in Python output scenarios-- #define ENDFORMAT "\033[0m"


bool scanFilePair (const std::string& intens_fpath, const std::string& label_fpath, int num_threads)
{
	std::cout << intens_fpath << "\n";

	int lvl = 0;	// Pyramid level

	// File #1 (intensity)
	GrayscaleTiffTileLoader<uint32_t> I (num_threads, intens_fpath);

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
	GrayscaleTiffTileLoader<uint32_t> L(num_threads, label_fpath);

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
	std::shared_ptr<std::vector<uint32_t>> ptrI = std::make_shared<std::vector<uint32_t>> (tileSize);
	std::shared_ptr<std::vector<uint32_t>> ptrL = std::make_shared<std::vector<uint32_t>> (tileSize);

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

			I.loadTileFromFile (ptrI, row, col, 0 /*layer*/, lvl);
			L.loadTileFromFile (ptrL, row, col, 0 /*layer*/, lvl);
			auto& dataI = *ptrI;
			auto& dataL = *ptrL;

			// --Timing
			end = std::chrono::system_clock::now();
			std::chrono::duration<double, std::milli> elapsedTile = end - start;
			
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

					update_label (x, y, label, dataI[i]);
				}
			}

			// --Timing
			end = std::chrono::system_clock::now();
			std::chrono::duration<double, std::milli> elapsedCalc = end - start;
			
			// --Time ratio
			#ifdef CHECKTIMING
			std::cout << " F/T: " << elapsedCalc.count() << " / " << elapsedTile.count() << " = " << elapsedCalc.count() / elapsedTile.count() << " x " << std::endl;
			#endif

			// Show stayalive progress info
			if (cnt++ % 4 == 0)
				std::cout << "\t" 
					//--Harmful in Python output scenarios-- << BEGINFORMAT_RED	
					<< int((row * nth + col) * 100 / float(nth * ntw) *100) / 100. << "%\t" << uniqueLabels.size() << " ULs"
					//--Harmful in Python output scenarios-- << ENDFORMAT	
					<< "\n";

			totalTileLoadTime += elapsedTile.count();
			totalFeatureReduceTime += elapsedCalc.count();

		}

	// Show stayalive progress info
	std::cout << "\t" 
		//--Harmful in Python output scenarios-- << BEGINFORMAT_RED
		<< "100%\t" << uniqueLabels.size() << " ULs"
		//--Harmful in Python output scenarios-- << ENDFORMAT
		<< "\n";

	return true;
}

int processDataset (
	const std::vector<std::string>& intensFiles, 
	const std::vector<std::string>& labelFiles, 
	int numFastloaderThreads, 
	int numSensemakerThreads, 
	int numReduceThreads, 
	int min_online_roi_size, 
	bool save2csv, 
	const std::string& csvOutputDir)
{
	// Temporarily disabled Haralick while I'm fixing its speed
	//		auto F = { TEXTURE_HARALICK2D };
	//		theFeatureSet.disableFeatures(F);
	//

	// Sanity
	std::chrono::time_point<std::chrono::system_clock> start, end;

	// Timing
	totalFeatureReduceTime = totalTileLoadTime = 0.0;
	start = std::chrono::system_clock::now();

	bool ok = true;

	auto nf = intensFiles.size();
	for (int i = 0; i < nf; i++)
	{
		// Clear label stats buffers
		clearLabelStats();

		auto &ifp = intensFiles[i], 
			&lfp = labelFiles[i];

		// Cache the file names to be picked up by labels to know their file origin
		std::filesystem::path p_int (ifp), p_seg (lfp);
		theSegFname = p_seg.filename().string();
		theIntFname = p_int.filename().string();

		// Scan a label-intensity pair and calculate features
		//---Option--- 
		if (numSensemakerThreads == 1)
			ok = scanFilePair (ifp, lfp, numFastloaderThreads);	// Sequential
		else
			ok = scanFilePairParallel (ifp, lfp, numFastloaderThreads, numSensemakerThreads);	// Parallel
		if (ok == false)
		{
			std::cout << "scanFilePair() returned an error code while processing file pair " << ifp << " and " << lfp << std::endl;
			return 1;
		}

		// --Timing
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		// Execute calculations requiring reduction
		reduce (numReduceThreads, min_online_roi_size);

		// --Timing
		end = std::chrono::system_clock::now();
		std::chrono::duration<double, std::milli> elapsed2 = end - start;
		
		#ifdef CHECKTIMING
		std::cout << "\tTiming of sensemaker::reduce [s] " << elapsed2.count() << std::endl;
		#endif

		totalFeatureReduceTime += elapsed2.count();

		// Save the result for this intensity-label file pair
		if (save2csv)
			ok = save_features_2_csv (ifp, csvOutputDir);
		else
			ok = save_features_2_buffer (calcResultBuf);
		if (ok == false)
		{
			std::cout << "save_features_2_csv() returned an error code" << std::endl;
			return 2;
		}
	}

	// Timing
	end = std::chrono::system_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;
	double elapsed_sec = elapsed.count() / 1000.;
	std::cout << "Elapsed time (s) " << elapsed.count() << std::endl;
	std::cout 
		<< "Total tile load time [s]: " << totalTileLoadTime 
		<< "\n\t+\nTotal feature calc time [s]: " << totalFeatureReduceTime 
		<< "\n\t=\nTotal time [s]: " << totalTileLoadTime + totalFeatureReduceTime 
		<< std::endl;

	return 0; // success
}

bool scanViaFastloader (const std::string & fpath, int num_threads)
{
	// Sanity check
	std::vector <uint32_t> histo; // histogram channels
	std::vector <uint32_t> counts; // channel counts
	//
	std::cout << std::endl << "Processing file " << fpath << std::endl;

	// Get the TIFF file info
	GrayscaleTiffTileLoader<uint32_t> fl (num_threads, fpath);
	
	auto th = fl.tileHeight(0),
		tw = fl.tileWidth(0),
		td = fl.tileDepth(0);
	auto tileSize = th*tw;

	// Just for information
	auto fh = fl.fullHeight(0);
	auto fw = fl.fullWidth(0);
	auto fd = fl.fullDepth(0);

	// Read the TIFF tile by tile
	auto ntw = fl.numberTileWidth(0);
	auto nth = fl.numberTileHeight(0);
	auto ntd = fl.numberTileDepth(0);

	// Allocate the tile buffer
	std::shared_ptr<std::vector<uint32_t>> data = std::make_shared<std::vector<uint32_t>>(tileSize);

	for (unsigned int row=0; row<nth; row++)
		for (unsigned int col = 0; col < ntw; col++)
		{
			std::cout << "\tt." << row*ntw+col+1 << " of " << nth*ntw << std::endl;
			fl.loadTileFromFile (data, row, col, 0 /*layer*/, 0 /*level*/);

			// Calculate statistics
			for (unsigned long i = 0; i < tileSize; i++)
			{
				// Sanity
				auto &v = *data;
				auto iter = std::find (histo.begin(), histo.end(), v[i]);
				if (iter == histo.end())
				{
					// The pixel value is not known, add it
					histo.push_back(v[i]);	// pixel value is not in the histogram so save it 
					counts.push_back(1);
				}
				else
				{
					// The pixel value is previously known, increment its counter
					auto index = iter - histo.begin();
					counts[index] = counts[index] + 1;
				}
			}
		}

	// Sanity
	std::cout << "\t"<< histo.size() << " histogram entries" << std::endl;
	// --Show a histogram fragment
	for (int i = 0; i < 100 && i < histo.size(); i++)
	{
		std::cout << i << "\t" << histo[i] << "\t" << counts[i] << std::endl;
	}

	return true;
}


bool TraverseViaFastloader1 (const std::string & fpath, int num_threads)
{
	// Sanity
#ifndef __unix
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof (statex);
	GlobalMemoryStatusEx (&statex);
	std::cout << "memory load, %=" << statex.dwMemoryLoad
		<< "\ntotal physical memory, Mb=" << statex.ullTotalPhys / 1048576
		<< "\ntotal available meory, Mb=" << statex.ullAvailPhys / 1048576
		<< std::endl;
#endif

	// Get the TIFF file info
	GrayscaleTiffTileLoader<uint32_t> gfl(num_threads, fpath);

	auto th = gfl.tileHeight(0),
		tw = gfl.tileWidth(0),
		td = gfl.tileDepth(0);
	auto tileSize = th * tw;

	// Just for information
	auto fh = gfl.fullHeight(0);
	auto fw = gfl.fullWidth(0);
	auto fd = gfl.fullDepth(0);

	// Read the TIFF tile by tile
	auto ntw = gfl.numberTileWidth(0);
	auto nth = gfl.numberTileHeight(0);
	auto ntd = gfl.numberTileDepth(0);

	// Radius
	const int radiusDepth = 1, radiusHeight = 1, radiusWidth = 1;

	// Instanciate a Tile loader
	auto tl = std::make_shared<VirtualFileTileChannelLoader>(
		num_threads,
		fh, fw, fd,
		th, tw, td,
		1 /*numberChannels*/);

	// Create the Fast Loader configuration
	auto options = std::make_unique<fl::FastLoaderConfiguration<fl::DefaultView<int>>>(tl);
	
	// Set the configuration
	options->radius(radiusDepth, radiusHeight, radiusWidth);
	options->ordered(true);
	options->borderCreatorConstant(0);
	// Create the Fast Loader Graph
	auto fl = fl::FastLoaderGraph<fl::DefaultView<int >>(std::move(options));
	// Execute the graph
	fl.executeGraph();
	// Request all the views in the graph
	fl.requestAllViews();
	// Indicate no other view will be requested
	fl.finishRequestingViews();

	// For each of the view
	while (auto view = fl.getBlockingResult()) {
		// Do stuff

		// Return the view to the Fast Loader Graph
		view->returnToMemoryManager();
	}
	// Wait for the graph to terminate
	fl.waitForTermination();

	return true;
}
