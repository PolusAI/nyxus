#include <algorithm>
#include "version.h"
#include "dirs_and_files.h"
#include "environment.h"
#include "globals.h"

#ifdef USE_GPU
	bool gpu_initialize(int dev_id); 
#endif

int main (int argc, char** argv)
{
	VERBOSLVL1(std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021-2023 Axle Informatics" << " Build of " << __TIMESTAMP__ << "\n";)

	bool parseOk = theEnvironment.parse_cmdline (argc, argv);
	if (! parseOk)
		return 1;

	VERBOSLVL1(theEnvironment.show_summary("\n"/*head*/, "\n"/*tail*/);)

	#ifdef USE_GPU
		if (theEnvironment.using_gpu())
		{
			int gpuDevNo = theEnvironment.get_gpu_device_choice();
			if (gpuDevNo >= 0 && gpu_initialize(gpuDevNo) == false)
			{
				std::cout << "Error: cannot use GPU device ID " << gpuDevNo << ". You can disable GPU usage via command line option " << USEGPU << "=false\n";
				return 1;
			}
		}
	#endif

	// Have the feature manager prepare the feature toolset reflecting user's selection
	if (!theFeatureMgr.compile())
	{
		std::cout << "Error: compiling feature methods failed\n";
		return 1;
	}
	theFeatureMgr.apply_user_selection();

	// Scan file names
	std::vector <std::string> intensFiles, labelFiles;
	int errorCode = Nyxus::read_dataset (
		theEnvironment.intensity_dir, 
		theEnvironment.labels_dir, 
		theEnvironment.get_file_pattern(),
		theEnvironment.output_dir, 
		theEnvironment.intSegMapDir, 
		theEnvironment.intSegMapFile, 
		true, 
		intensFiles, labelFiles);
	if (errorCode)
	{
		std::cout << "Dataset structure error\n";
		return 1; 
	}

	// Current time stamp #1
	auto startTS = getTimeStr();
	VERBOSLVL1(std::cout << "\n>>> STARTING >>> " << startTS << "\n";)

	// Process the image sdata
	int min_online_roi_size = 0;
	errorCode = processDataset (
		intensFiles, 
		labelFiles, 
		theEnvironment.n_loader_threads, 
		theEnvironment.n_pixel_scan_threads, 
		theEnvironment.n_reduce_threads,
		min_online_roi_size,
		true, // 'true' to save to csv
		theEnvironment.output_dir);

	// Check the error code 
	switch (errorCode)
	{
	case 0:		// Success
		break;
	case 1:		// Dataset structure error e.g. intensity-label file name mismatch
		std::cout << std::endl << "Input data error" << std::endl;
		break;
	case 2:		// Internal FastLoader error e.g. TIFF access error
		std::cout << std::endl << "Result output error" << std::endl;
		break;
	case 3:		// Memory error
		std::cout << std::endl << "Memory error" << std::endl;
		break;
	default:	// Any other error
		std::cout << std::endl << "Error #" << errorCode << std::endl;
		break;
	}

	// Current time stamp #2
	VERBOSLVL1(std::cout << "\n>>> STARTED >>>\t" << startTS << "\n>>> FINISHED >>>\t" << getTimeStr() << "\n";)

	int exitCode = errorCode;
	return exitCode;
}

void showCmdlineHelp()
{
	std::cout 
		<< "Command line format:" << std::endl 
		<< "\t" << PROJECT_NAME << " <label directory> <intensity directory> <output directory> [--minOnlineROI <# of pixels>] [--threads <# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


