#include "environment.h"
#include "sensemaker.h"
#include "timing.h"

FeatureSet theFeatureSet;
std::string theSegFname, theIntFname;	
std::unordered_set <int> uniqueLabels;
std::vector<int> sortedUniqueLabels;	// Populated in reduce()
std::unordered_map <int, LR> labelData;
std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;
std::vector<double> calcResultBuf;
Environment theEnvironment;

std::map <std::string, double> Stopwatch::totals;