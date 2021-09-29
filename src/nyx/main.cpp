// sensemaker1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "histogram.h"
#include "sensemaker.h"
#include "version.h"
#include "environment.h"

#if 0
//#define _CRT_SECURE_NO_WARNINGS
#include <ctime>

std::string getTimeStr() {
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

	std::string s(30, '\0');
	std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
	return s;
}
#endif

int main (int argc, char** argv)
{
	std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021 Axle Informatics\n";

	int parseRes = theEnvironment.parse_cmdline (argc, argv);
	if (parseRes)
		return parseRes;
	theEnvironment.show_summary ("\n"/*head*/, "\n"/*tail*/);

	#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
	std::cout << std::endl << "Attention! Will dump intensioities for label " << SANITY_CHECK_INTENSITIES_FOR_LABEL << ", otherwise undefine SANITY_CHECK_INTENSITIES_FOR_LABEL" << std::endl;
	#endif

	// Scan file names
	std::vector <std::string> intensFiles, labelFiles;
	int errorCode = checkAndReadDataset (theEnvironment.intensity_dir, theEnvironment.labels_dir, theEnvironment.output_dir, true, intensFiles, labelFiles);
	if (errorCode)
	{
		std::cout << std::endl << "Dataset structure error" << std::endl;
		return 1; 
	}

	// One-time initialization
	init_feature_buffers();

	// Current time #1

		// declaring argument of time()
	time_t my_time = time(NULL);
	// ctime() used to give the present time
	printf("Started at %s", ctime(&my_time));

	//std::cout << "Started at " << getTimeStr() << "\n";

	// Process the image sdata
	int min_online_roi_size = 0;
	errorCode = processDataset (
		intensFiles, 
		labelFiles, 
		theEnvironment.n_loader_threads, 
		theEnvironment.n_pixel_scan_threads, 
		theEnvironment.n_reduce_threads,
		min_online_roi_size,
		true /*save to csv*/, 
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

	// Current time #2
	//std::cout << "Ended at " << getTimeStr() << "\n";
	time_t my_time2 = time(NULL);
	// ctime() used to give the present time
	printf("Ended at %s", ctime(&my_time2));

	int exitCode = errorCode;
	return exitCode;
}

void showCmdlineHelp()
{
	std::cout 
		<< "Command line format:" << std::endl 
		<< "\t" << PROJECT_NAME << " <label directory> <intensity directory> <output directory> [--minOnlineROI <# of pixels>] [--threads <# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


