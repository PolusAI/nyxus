#include "environment.h"
#include "globals.h"
#include "gpucache.h"
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
	size_t zero_background_area;
	std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;

	#ifdef USE_GPU
	// GPU cache of a ROI batch
	GpuCache<Pixel2> gpu_roiclouds_2d;
	GpuCache<size_t> gpu_roicontours_2d;
	#endif

	// Nested ROI
	std::unordered_map <std::string, NestableRois> nestedRoiData;

	// Features
	FeatureSet theFeatureSet;
	FeatureManager theFeatureMgr;

	// Results cache serving Nyxus' CLI & Python API, NyxusHie's CLI & Python API
	ResultsCache theResultsCache;

	// Shows a message in CLI or Python terminal 
	void sureprint (const std::string& msg, bool send_to_stderr)
	{
#ifdef WITH_PYTHON_H
		pybind11::print(msg);
#else
		if (send_to_stderr)
			std::cerr << msg;
		else
			std::cout << msg;
#endif
	}
}
