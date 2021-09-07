#include "sensemaker.h"

FeatureSet featureSet;

std::unordered_set <int> uniqueLabels;
std::vector<int> sortedUniqueLabels;	// Populated in reduce()
std::unordered_map <int, LR> labelData;
std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;
std::vector<double> calcResultBuf;
