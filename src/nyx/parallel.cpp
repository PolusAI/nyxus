#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "virtual_file_tile_channel_loader.h"
#include "sensemaker.h"

#include <string>
#include <fast_loader/specialised_tile_loader/grayscale_tiff_tile_loader.h>
//#include <map>

#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif


// Global mutex locking requests to per-label mutexes
std::mutex glock;

// Parallel version of update_label_stats() 
void update_label_stats_parallel(int x, int y, int label, PixIntens intensity)
{
	std::mutex* mux;
	{
		std::lock_guard<std::mutex> lg(glock);

		auto itm = labelMutexes.find(label);
		if (itm == labelMutexes.end())
		{
			//=== Create a label-specific mutex
			itm = labelMutexes.emplace(label, std::make_shared <std::mutex>()).first;

			//=== Create a label record
			auto it = uniqueLabels.find(label);
			if (it != uniqueLabels.end())
				std::cout << "\n\tERROR\n";

			// Remember this label
			uniqueLabels.insert(label);

			// Initialize the label record
			LR lr;
			init_label_record (lr, x, y, label, intensity);
			labelData[label] = lr;
			
			// We're done processing the very first pixel of a label, return
			return;
		}
		
		// No need to create a mutex for this label. Let's consume the existing one:
		mux = itm->second.get();
	}

	// Calculate features updates for this "iteration"
	{
		// Lock guarding the call of function update_label_record()
		std::lock_guard<std::mutex> lock(*mux);

		#ifdef SIMULATE_WORKLOAD_FACTOR
		// Simulate a chunk of processing. 1K iterations cost ~300 mks
		for (long tmp = 0; tmp < SIMULATE_WORKLOAD_FACTOR * 1000; tmp++)
			auto start = std::chrono::system_clock::now();
		#endif

		// Update label's stats
		LR& lr = labelData[label];
		update_label_record (lr, x, y, label, intensity);
	}
}

// High-level handler of pixel features update. (update_label_stats_parallel() is the low-level handler.)
void processPixels (unsigned int start_idx_inclusive, unsigned int end_idx_exclusive, std::vector<uint32_t>* dataL, std::vector<uint32_t>* dataI, unsigned int tw)
{
	for (unsigned long i = start_idx_inclusive; i < end_idx_exclusive; i++)
	{
		auto label = (*dataL)[i];

		#ifdef SINGLE_ROI_TEST
		label = 1;
		#endif

		if (label != 0)
		{
			int y = i / tw,
				x = i % tw;
			update_label_stats_parallel(x, y, label, (*dataI)[i]);
		}
	}
}

// Function driving tiled processing a file pair - intensity and its mask
bool scanFilePairParallel (const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads)
{
	std::cout << std::endl << "Processing pair " << intens_fpath << " -- " << label_fpath << " with " << num_fastloader_threads << " threads" << std::endl;

	int lvl = 0;	// Pyramid level

	std::vector<std::future<void>> futures;

	// File #1 (intensity)
	GrayscaleTiffTileLoader<uint32_t> I(num_fastloader_threads, intens_fpath);

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
	GrayscaleTiffTileLoader<uint32_t> L(num_fastloader_threads, label_fpath);

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
			std::cout << "\tt." << row * ntw + col + 1 << "/" << nth * ntw; 

			// --Timing
			std::chrono::time_point<std::chrono::system_clock> start, end;
			start = std::chrono::system_clock::now();

			I.loadTileFromFile(ptrI, row, col, 0 /*layer*/, lvl);
			L.loadTileFromFile(ptrL, row, col, 0 /*layer*/, lvl);
			auto& dataI = *ptrI;
			auto& dataL = *ptrL;

			// --Timing
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed1 = end - start;

			// Calculate features

			// --Timing
			start = std::chrono::system_clock::now();

			{
				/*
				//--- Experimental: just 2 threads
				auto fu1 = std::async(std::launch::async, processPixels,
					0, tileSize / 2,
					&dataL, &dataI, tw);
				auto fu2 = std::async(std::launch::async, processPixels,
					tileSize / 2, tileSize,
					&dataL, &dataI, tw);
				*/

				int workPerThread = tileSize / num_sensemaker_threads;
				std::vector<std::future<void>> T;
				for (int t = 0; t < num_sensemaker_threads; t++)
				{
					int idxS = t * workPerThread,
						idxE = idxS + workPerThread;
					if (t == num_sensemaker_threads - 1)
						idxE = tileSize; // include the roundoff tail
					T.push_back(std::async(std::launch::async, processPixels, idxS, idxE, &dataL, &dataI, tw));
				}
			}
			// --Timing
			end = std::chrono::system_clock::now();
			std::chrono::duration<double, std::milli> elapsed2 = end - start;
			std::cout << " F/T: " << elapsed2.count() << " / " << elapsed1.count() << " = " << elapsed2.count() / elapsed1.count() << " x" << std::endl;
			totalTileLoadTime += elapsed1.count();
			totalPixStatsCalcTime += elapsed2.count();

			if (cnt++ % 4 == 0)
				std::cout << std::endl;
		}

	return true;
}

