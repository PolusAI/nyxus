#include <algorithm>
#include "version.h"
#include "environment.h"
#include "globals.h"

using namespace Nyxus;

int main (int argc, char** argv)
{
	std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021 Axle Informatics\n" << "Build of " << __TIMESTAMP__ << "\n";

	int parseRes = theEnvironment.parse_cmdline (argc, argv);
	if (parseRes)
		return parseRes;
	theEnvironment.show_summary ("\n"/*head*/, "\n"/*tail*/);

	// Scan file names
	std::vector <std::string> intensFiles, labelFiles;
	int errorCode = checkAndReadDataset (theEnvironment.intensity_dir, theEnvironment.labels_dir, theEnvironment.output_dir, true, intensFiles, labelFiles);
	if (errorCode)
	{
		std::cout << std::endl << "Dataset structure error" << std::endl;
		return 1; 
	}

	// Sort the dataset
	std::sort (intensFiles.begin(), intensFiles.end());
	std::sort (labelFiles.begin(), labelFiles.end());

	// One-time initialization
	init_feature_buffers();

	// Current time stamp #1
	auto startTS = getTimeStr();
	std::cout << "\n>>> STARTING >>> " << startTS << "\n";

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
		std::cout << std::endl << "Dataset structure error" << std::endl;
		break;
	case 2:		// Internal FastLoader error e.g. TIFF access error
		std::cout << std::endl << "Dataset structure error" << std::endl;
		break;
	case 3:		// Memory error
		std::cout << std::endl << "Dataset structure error" << std::endl;
		break;
	default:	// Any other error
		std::cout << std::endl << "Error #" << errorCode << std::endl;
		break;
	}

	// Current time stamp #2
	std::cout << "\n>>> STARTED >>>\t" << startTS << "\n>>> ENDING >>>\t" << getTimeStr() << "\n";

	int exitCode = errorCode;
	return exitCode;
}

void showCmdlineHelp()
{
	std::cout 
		<< "Command line format:" << std::endl 
		<< "\t" << PROJECT_NAME << " <label directory> <intensity directory> <output directory> [--minOnlineROI <# of pixels>] [--threads <# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


