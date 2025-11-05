#pragma once

#include <unordered_map>
#include <thread>
#include <future>
#include "dataset.h"
#include "feature_settings.h"
#include "roi_cache.h"

namespace Nyxus
{
	/// @brief Defines a parallelizable function 
	typedef void (*functype) (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & f_settings, const Dataset & dataset);

	/// @brief Runs ROI data processing functions in parallel 
	/// @param f Global function or static class method
	/// @param nThr Number of threads
	/// @param workPerThread Number of ROIs per thread
	/// @param datasetSize Total of ROIs
	/// @param ptrLabels ROI labels "dictionary"
	/// @param ptrLabelData ROI data
	/// @param f_settings feature-specific settings
	inline void runParallel (
		functype f, 
		int nThr, 
		size_t workPerThread, 
		size_t datasetSize, 
		std::vector<int>* ptrLabels, 
		std::unordered_map <int, LR>* ptrLabelData,
		const Fsettings & f_settings,
		const Dataset & dataset)
	{
		std::vector<std::future<void>> T;
		for (int t = 0; t < nThr; t++)
		{
			size_t idxS = t * workPerThread,
				idxE = idxS + workPerThread;
			if (t == nThr - 1)
				idxE = datasetSize; // include the tail
			T.push_back(std::async(std::launch::async, f, idxS, idxE, ptrLabels, ptrLabelData, std::cref(f_settings), std::cref(dataset)));
		}
	}

	void parallelReduceConvHull (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & fst, const Dataset & ds);
}
