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
	std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021 Axle Informatics" << std::endl;

	// Check the command line (it's primitive now)
	if (! (argc == 4 || argc == 6))
	{
		showCmdlineHelp();
		return 1;
	}

	// Parse the command line
	std::string dirIntens = argv[1], 
		dirLabels = argv[2], 
		dirOut = argv[3];

	int n_tlt = 1 /*# of tile loader threads*/, n_fct = 1 /*# Sensemaker threads*/;	// Default values
	if (argc == 6)
	{
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
	}

    std::cout << 
		"Using" << std::endl << "\t<intensity data directory> = " << dirIntens << std::endl << 
		"\t<labels data directory> = " << dirLabels << std::endl <<
		"\t<output directory> = " << dirOut << std::endl <<
		"\t" << n_tlt << " tile loader (currently FastLoader) threads" << std::endl <<
		"\t" << n_fct << " feature calculation threads" << std::endl ;

	#ifdef SINGLE_ROI_TEST
	std::cout << std::endl << "Attention! Running the single-ROI test" << std::endl;
	#endif

	// Scan file names
	std::vector <std::string> intensFiles, labelFiles;
	int errorCode = checkAndReadDataset (dirIntens, dirLabels, dirOut, intensFiles, labelFiles);
	if (errorCode)
	{
		std::cout << std::endl << "Dataset structure error" << std::endl;
		return 1; 
	}

	// One-time initialization
	init_feature_buffers();

	// Process the image sdata
	errorCode = ingestDataset (intensFiles, labelFiles, n_tlt /*# of FastLoader threads*/, n_fct /*# Sensemaker threads*/, dirOut);

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
		<< "\t" << PROJECT_NAME << " <intensity directory> <label directory> <output directory> [<# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


