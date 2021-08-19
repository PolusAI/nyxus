#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include "sensemaker.h"

bool save_features_2_buffer (std::vector<double> & resultBuf)
{
	resultBuf.clear();
	size_t bufLen = uniqueLabels.size() * featureSet.numEnabled();
	resultBuf.reserve(bufLen);

	// Sort the labels
	std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
	std::sort(L.begin(), L.end());

	unsigned int labelIdx = 0;
	for (auto l : L)
	{
		LR& r = labelData[l];
		for (int i = 0; i < AvailableFeatures::_COUNT_; i++)
		{
			if (featureSet.isEnabled(i))
			{
				auto fval = r.getValue((AvailableFeatures)i);
				resultBuf.push_back(fval);
			}
		}
	}

	return true;
}