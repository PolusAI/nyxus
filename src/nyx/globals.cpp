#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

namespace Nyxus
{
	// Command line info and default values
	Environment theEnvironment;

	// Everything related to images
	ImageLoader theImLoader;
	std::string theSegFname, theIntFname;

	// Everything related to ROI data
	std::unordered_set <int> uniqueLabels;
	std::unordered_map <int, LR> roiData;
	std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;

	// Features
	FeatureSet theFeatureSet;
	FeatureManager theFeatureMgr;

	// Results cache serving Nyxus' CLI & Python API, NyxusHie's CLI & Python API
	ResultsCache theResultsCache;
}
