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


// Sanity
#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif


std::mutex glock;

void update_label_stats_parallel(int x, int y, int label, PixIntens intensity)
{
	// Create a mutex
	std::mutex* mux;
	{
		std::lock_guard<std::mutex> lg(glock);

		auto itm = labelMutexes.find(label);
		if (itm == labelMutexes.end())
		{
			//=== Create a label-specific mutex
			itm = labelMutexes.emplace(label, std::make_unique <std::mutex>()).first;

			//=== Create a label record
			auto it = uniqueLabels.find(label);
			if (it != uniqueLabels.end())
				std::cout << "\n\tERROR\n";

			// Remember this label
			uniqueLabels.insert(label);

			//--- New
			LR lr;

			lr.labelCount = 1;
			lr.labelPrevCount = 0;
			// Min
			lr.labelMins = intensity;
			// Max
			lr.labelMaxs = intensity;
			// Moments
			lr.labelMeans = intensity;
			lr.labelM2 = 0;
			lr.labelM3 = 0;
			lr.labelM4 = 0;
			// Median. Cache intensity values per label for the median calculation
			std::shared_ptr<std::unordered_set<PixIntens>> ptrUS = std::make_shared <std::unordered_set<PixIntens>>();
			//---time consuming---	ptrUS->reserve(N2R_2);
			ptrUS->insert(intensity);
			lr.labelUniqueIntensityValues = ptrUS;
			// Energy
			lr.labelMassEnergy = intensity;
			// Variance and standard deviation
			lr.labelVariance = 0.0;
			// Mean absolute deviation
			lr.labelMAD = 0;
			// Previous intensity
			lr.labelPrevIntens = intensity;
			// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
			lr.labelCentroid_x = StatsReal(x) + 1;
			lr.labelCentroid_y = StatsReal(y) + 1;
			// Histogram
			std::shared_ptr<Histo> ptrH = std::make_shared <Histo>();
			ptrH->add_observation(intensity);
			lr.labelHistogram = ptrH;
			//
			labelData[label] = lr;
		}

		mux = itm->second.get();
	}


	// Research
	if (intensityMin == -999.0 || intensityMin > intensity)
		intensityMin = intensity;
	if (intensityMax == -999.0 || intensityMax < intensity)
		intensityMax = intensity;

	// Calculate features updates for this "iteration"

	//else
	{
		std::lock_guard<std::mutex> lock(*mux);

#ifdef SIMULATE_WORKLOAD_FACTOR
		// Simulate a chunk of processing. 1K iterations cost ~300 mks
		for (long tmp = 0; tmp < SIMULATE_WORKLOAD_FACTOR * 1000; tmp++)
			auto start = std::chrono::system_clock::now();
#endif

		//--- New
		LR& lr = labelData[label];
		// Count of pixels belonging to the label
		auto prev_n = lr.labelCount;	// Previous count
		lr.labelPrevCount = prev_n;
		auto n = prev_n + 1;	// New count
		lr.labelCount = n;

		// Cumulants for moments calculation
		auto prev_mean = lr.labelMeans;
		auto delta = intensity - prev_mean;
		auto delta_n = delta / n;
		auto delta_n2 = delta_n * delta_n;
		auto term1 = delta * delta_n * prev_n;

		// Mean
		auto mean = prev_mean + delta_n;
		lr.labelMeans = mean;

		// Moments
		lr.labelM4 = lr.labelM4 + term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * lr.labelM2 - 4 * delta_n * lr.labelM3;
		lr.labelM3 = lr.labelM3 + term1 * delta_n * (n - 2) - 3 * delta_n * lr.labelM2;
		lr.labelM2 = lr.labelM2 + term1;

		// Median
		auto ptr = lr.labelUniqueIntensityValues;
		ptr->insert(intensity);

		// Min 
		auto tmp = std::min (1, 2);
		lr.labelMins = std::min(lr.labelMins, (StatsInt)intensity);

		// Max
		lr.labelMaxs = std::min(lr.labelMins, (StatsInt)intensity);

		// Energy
		lr.labelMassEnergy = lr.labelMassEnergy + intensity;

		// Variance and standard deviation
		if (n >= 2)
		{
			double s_prev = lr.labelVariance,
				diff = double(intensity) - prev_mean,
				diff2 = diff * diff;
			lr.labelVariance = (n - 2) * s_prev / (n - 1) + diff2 / n;
		}
		else
			lr.labelVariance = 0;

		// Mean absolute deviation
		lr.labelM2 = lr.labelM2 + sqrt(delta * (intensity - mean));
		lr.labelMAD = lr.labelM2 / n;

		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		lr.labelCentroid_x = lr.labelCentroid_x + StatsReal(x) + 1;
		lr.labelCentroid_y = lr.labelCentroid_y + StatsReal(y) + 1;

		// Histogram
		auto ptrH = lr.labelHistogram;
		ptrH->add_observation(intensity);

		// Previous intensity for succeeding iterations
		lr.labelPrevIntens = intensity;
	}
}

void processPixels(unsigned int start_idx_inclusive, unsigned int end_idx_exclusive, std::vector<uint32_t>* dataL, std::vector<uint32_t>* dataI, unsigned int tw)
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
	for (int row = 0; row < nth; row++)
		for (int col = 0; col < ntw; col++)
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
				//--- just 2 threads
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
					T.push_back(std::async(std::launch::async, processPixels, 0, tileSize / 4, &dataL, &dataI, tw));
				}
			}
			// --Timing
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed2 = end - start;
			std::cout << "\tT(featureScan) vs T(loadTile) [s]: " << elapsed2.count() << " / " << elapsed1.count() << " = " << elapsed2.count() / elapsed1.count() << " x" << std::endl;
			totalTileLoadTime += elapsed1.count();
			totalPixStatsCalcTime += elapsed2.count();

			if (cnt++ % 4 == 0)
				std::cout << std::endl;
		}

	return true;
}

