// sensemaker1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "histogram.h"
#include "sensemaker.h"
#include "version.h"


int main (int argc, char** argv)
{
	//??? DEBUG
	auto F = { TEXTURE_HARALICK2D };
	featureSet.enableAll();
	featureSet.disableFeatures (F);
	//


	// Quick tests:
	//		bool hOK = test_histogram();
	//		return 0;

	std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021 Axle Informatics" << std::endl;

	// Check the command line (it's primitive now)
	if (! (argc == 4 || argc == 7)) // 4 - only directories, 7 - threads info and threads info
	{
		showCmdlineHelp();
		return 1;
	}

	// Parse the command line
	std::string dirIntens = argv[1], 
		dirLabels = argv[2], 
		dirOut = argv[3];

	int n_tlt = 1 /*# of tile loader threads*/, n_fct = 1 /*# Sensemaker threads*/, min_online_roi_size = 0;	// Default values

	if (argc == 7)
	{
		/*
		if (strcmp(argv[4], CMDLN_OPT_THREADS) != 0)
		{
			std::cout << "Expecting " << argv[4] << " to be " << CMDLN_OPT_THREADS << std::endl;
			return 1;
		}
		*/

		char* stopPtr;

		// --parse the # of fastloader threads
		n_tlt = strtol (argv[4], &stopPtr, 10);
		if (*stopPtr || n_tlt<=0)
		{
			std::cout << "Command line error: expecting '" << argv[4] << "' to be a positive integer constant. Stopping" << std::endl;
			return 1;
		}

		// --parse the # of sensemaker threads
		n_fct = strtol(argv[5], &stopPtr, 10);
		if (*stopPtr || n_fct <= 0)
		{
			std::cout << "Command line error: expecting '" << argv[5] << "' to be a positive integer constant. Stopping" << std::endl;
			return 1;
		}

		// --parse the min online ROI size
		min_online_roi_size = strtol(argv[6], &stopPtr, 10);
		if (*stopPtr || min_online_roi_size <= 0)
		{
			std::cout << "Command line error: expecting '" << argv[6] << "' to be a positive integer constant. Stopping" << std::endl;
			return 1;
		}
	}

    std::cout << 
		"Using" << std::endl << "\t<intensity data directory> = " << dirIntens << std::endl << 
		"\t<labels data directory> = " << dirLabels << std::endl <<
		"\t<output directory> = " << dirOut << std::endl <<
		"\t" << n_tlt << " tile loader (currently FastLoader) threads" << std::endl <<
		"\t" << n_fct << " feature calculation threads" << std::endl << 
		"\t" << min_online_roi_size << " min online ROI size" << std::endl ;

	#ifdef SINGLE_ROI_TEST
	std::cout << std::endl << "Attention! Running the single-ROI test, otherwise undefine SINGLE_ROI_TEST" << std::endl;
	#endif

	#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
	std::cout << std::endl << "Attention! Will dump intensioities for label " << SANITY_CHECK_INTENSITIES_FOR_LABEL << ", otherwise undefine SANITY_CHECK_INTENSITIES_FOR_LABEL" << std::endl;
	#endif

	// Scan file names
	std::vector <std::string> intensFiles, labelFiles;
	int errorCode = checkAndReadDataset (dirIntens, dirLabels, dirOut, true, intensFiles, labelFiles);
	if (errorCode)
	{
		std::cout << std::endl << "Dataset structure error" << std::endl;
		return 1; 
	}

	// One-time initialization
	init_feature_buffers();

	// Process the image sdata
	errorCode = processDataset (intensFiles, labelFiles, n_tlt /*# of FastLoader threads*/, n_fct /*# Sensemaker threads*/, min_online_roi_size, true, dirOut);

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

	int exitCode = errorCode;
	return exitCode;
}

void showCmdlineHelp()
{
	std::cout 
		<< "Command line format:" << std::endl 
		<< "\t" << PROJECT_NAME << " <intensity directory> <label directory> <output directory> [--minOnlineROI <# of pixels>] [--threads <# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


