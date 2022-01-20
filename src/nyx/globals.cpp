#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

std::map <std::string, double> Stopwatch::totals;

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
	std::vector<double> calcResultBuf;

	// Timing
	double totalImgScanTime = 0.0, totalFeatureReduceTime = 0.0;	// Time counters external to class 'Stopwatch'

	// Features
	FeatureSet theFeatureSet;
	FeatureManager theFeatureMgr;

}
