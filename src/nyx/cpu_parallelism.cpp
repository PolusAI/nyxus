#include "cpu_parallelism.h"

void runParallel (functype f, int nThr, size_t workPerThread, size_t datasetSize, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	std::vector<std::future<void>> T;
	for (int t = 0; t < nThr; t++)
	{
		size_t idxS = t * workPerThread,
			idxE = idxS + workPerThread;
		if (t == nThr - 1)
			idxE = datasetSize; // include the tail
		// Example:	T.push_back(std::async(std::launch::async, parallelReduceIntensityStats, idxS, idxE, &sortedUniqueLabels, &labelData));
		T.push_back(std::async(std::launch::async, f, idxS, idxE, &sortedUniqueLabels, &labelData));
	}
}

