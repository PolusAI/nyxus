#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "environment.h"
#include "globals.h"
#include "grayscale_tiff.h"
#include <string>
//#include <map>

#ifdef _WIN32
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif

namespace Nyxus
{
	// Global mutex locking requests to per-label mutexes
	std::mutex glock;

	// Parallel version of update_label_stats() 
	void update_label_parallel(int x, int y, int label, PixIntens intensity)
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
				init_label_record(lr, theSegFname, theIntFname, x, y, label, intensity);
				roiData[label] = lr;

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
			LR& lr = roiData[label];
			update_label_record(lr, x, y, label, intensity);
		}
	}

	// High-level handler of pixel features update. (update_label_parallel() is the low-level handler.)
	void processPixels(unsigned int start_idx_inclusive, unsigned int end_idx_exclusive, std::vector<uint32_t>* dataL, std::vector<uint32_t>* dataI, unsigned int tw)
	{
		for (unsigned long i = start_idx_inclusive; i < end_idx_exclusive; i++)
		{
			auto label = (*dataL)[i];
			if (label != 0)
			{
				int y = i / tw,
					x = i % tw;

				// Collapse all the labels to one if single-ROI mde is requested
				if (theEnvironment.singleROI)
					label = 1;

				update_label_parallel(x, y, label, (*dataI)[i]);
			}
		}
	}

	// Function driving tiled processing a file pair - intensity and its mask
	bool scanFilePairParallel(const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads, int filepair_index, int tot_num_filepairs)
	{
		// Display (1) dataset progress info and (2) file pair info
		int digits = 2, k = std::pow(10.f, digits);
		float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
		std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << intens_fpath << "\n";

		int lvl = 0;	// Pyramid level

		std::vector<std::future<void>> futures;

		// File #1 (intensity)
		NyxusGrayscaleTiffTileLoader<uint32_t> I(
			num_fastloader_threads, 
			intens_fpath, 
			true,
			Nyxus::theEnvironment.fpimageOptions.min_intensity(),
			Nyxus::theEnvironment.fpimageOptions.max_intensity(),
			Nyxus::theEnvironment.fpimageOptions.target_dyn_range());

		size_t th = I.tileHeight(lvl),
			tw = I.tileWidth(lvl),
			td = I.tileDepth(lvl);
		size_t tileSize = th * tw;

		size_t fh = I.fullHeight(lvl),
			fw = I.fullWidth(lvl),
			fd = I.fullDepth(lvl),
			ntw = I.numberTileWidth(lvl),
			nth = I.numberTileHeight(lvl), 
			ntd = I.numberTileDepth(lvl);

		// File #2 (labels)
		NyxusGrayscaleTiffTileLoader<uint32_t> L(
			num_fastloader_threads, 
			label_fpath,
			false,
			Nyxus::theEnvironment.fpimageOptions.min_intensity(),
			Nyxus::theEnvironment.fpimageOptions.max_intensity(),
			Nyxus::theEnvironment.fpimageOptions.target_dyn_range());

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
					int workPerThread = tileSize / num_sensemaker_threads;
					std::vector<std::future<void>> T;
					for (int t = 0; t < num_sensemaker_threads; t++)
					{
						int idxS = t * workPerThread,
							idxE = idxS + workPerThread;
						if (t == num_sensemaker_threads - 1)
							idxE = tileSize; // include the tail
						T.push_back(std::async(std::launch::async, processPixels, idxS, idxE, &dataL, &dataI, tw));
					}
				}

				if (cnt++ % 4 == 0)
					std::cout << std::endl;
			}

		return true;
	}

} // namespace Nyxus
