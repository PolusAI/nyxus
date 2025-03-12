#include <algorithm>
#include <iomanip>
#include "globals.h"
#include "environment.h"
#include "version.h"
#ifdef USE_GPU
	#include "gpu/gpu.h"
#endif

using namespace Nyxus;

int main (int argc, char** argv)
{
	VERBOSLVL1 (std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021-2025 Axle Research" << " Build of " << __TIMESTAMP__ << "\n")

	if (! theEnvironment.parse_cmdline(argc, argv))
		return 1;

	VERBOSLVL1 (theEnvironment.show_summary("\n"/*head*/, "\n"/*tail*/))

	#ifdef USE_GPU
	if (theEnvironment.using_gpu())
	{
		int devid = theEnvironment.get_gpu_device_choice();
		if (NyxusGpu::gpu_initialize(devid) == false)
		{
			std::cerr << "Error: cannot use GPU device ID " << devid << ". You can disable GPU usage via command line option " << USEGPU << "=false\n";
			return 1;
		}
	}
	#endif

	// Have the feature manager prepare the feature toolset reflecting user's selection
	if (!theFeatureMgr.compile())
	{
		std::cerr << "Error: compiling feature methods failed\n";
		return 1;
	}
	theFeatureMgr.apply_user_selection();

	// Current time stamp #1
	auto tsStart = Nyxus::getCurTime();
	VERBOSLVL1 (std::cout << "\n>>> STARTING >>> " << Nyxus::getTimeStr(tsStart) << "\n");

	// Initialize feature classes 
	if (! theFeatureMgr.init_feature_classes())
	{
		std::cerr << "Error: initializing feature classes failed\n";
		return 1;
	}
		
	if (theEnvironment.dim() == 2)
	{
		// Scan intensity and mask directories, apply fileppatern, make intensity-mask image file pairs (if not in wholeslide mode)
		std::vector <std::string> intensFiles, labelFiles;
		std::string ermsg;
		int errorCode = Nyxus::read_2D_dataset(
			theEnvironment.intensity_dir,
			theEnvironment.labels_dir,
			theEnvironment.get_file_pattern(),
			theEnvironment.output_dir,
			theEnvironment.intSegMapDir,
			theEnvironment.intSegMapFile,
			true,
			intensFiles, 
			labelFiles,
			ermsg);
		if (errorCode)
		{
			#ifdef WITH_PYTHON_H
				throw std::runtime_error (ermsg);
			#endif

			std::cerr << "Errors while reading the dataset:\n" << ermsg << "\n";
			return 1;
		}

		// Process the image data
		int min_online_roi_size = 0;

		if (theEnvironment.singleROI)
			errorCode = processDataset_2D_wholeslide (
				intensFiles,
				labelFiles,
				theEnvironment.n_reduce_threads,
				min_online_roi_size,
				theEnvironment.saveOption,
				theEnvironment.output_dir);
		else
			errorCode = processDataset_2D_segmented (
				intensFiles,
				labelFiles,
				theEnvironment.n_loader_threads,
				theEnvironment.n_pixel_scan_threads,
				theEnvironment.n_reduce_threads,
				min_online_roi_size,
				theEnvironment.saveOption,
				theEnvironment.output_dir);


		// Report feature extraction error, if any
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
		case 4:
			std::cout << std::endl << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
		default:	// Any other error
			std::cout << std::endl << "Error #" << errorCode << std::endl;
			break;
		}

		// Process nested ROIs
		if (theEnvironment.nestedOptions.defined())
		{
			bool mineOK = mine_segment_relations2(
				labelFiles,
				theEnvironment.get_file_pattern(),
				theEnvironment.nestedOptions.rawChannelSignature, //---.channel_signature(),
				theEnvironment.nestedOptions.parent_channel_number(),
				theEnvironment.nestedOptions.child_channel_number(),
				theEnvironment.output_dir,
				theEnvironment.nestedOptions.aggregation_method(),
				theEnvironment.get_verbosity_level());

			// Report nested ROI errors, if any
			if (!mineOK)
			{
				std::cerr << "Error minimg hierarchical relations\n";
				return 1;
			}
		}
	} // 2D
	else
		if (theEnvironment.dim() == 3)
		{
			// Scan intensity and mask directories, apply fileppatern, make intensity-mask image file pairs
			std::vector <Imgfile3D_layoutA> intensFiles, labelFiles;

			int errorCode = Nyxus::read_3D_dataset (
				theEnvironment.intensity_dir,
				theEnvironment.labels_dir,
				theEnvironment.file_pattern_3D,
				theEnvironment.output_dir,
				theEnvironment.intSegMapDir,
				theEnvironment.intSegMapFile,
				true,
				intensFiles, 
				labelFiles);
			if (errorCode)
			{
				std::cout << "Dataset error\n";
				return 1;
			}

			// Process the image data
			int min_online_roi_size = 0;

			errorCode = processDataset_3D_segmented (
				intensFiles,
				labelFiles,
				theEnvironment.n_loader_threads,
				theEnvironment.n_pixel_scan_threads,
				theEnvironment.n_reduce_threads,
				min_online_roi_size,
				theEnvironment.saveOption,
				theEnvironment.output_dir);


			// Report feature extraction error, if any
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
			case 4:
				std::cout << std::endl << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
			default:	// Any other error
				std::cout << std::endl << "Error #" << errorCode << std::endl;
				break;
			}

		} // 3D

	// Current time stamp #2
	auto tsEnd = Nyxus::getCurTime();
	VERBOSLVL1(
		std::cout << "\n";
		std::cout << ">>> STARTED  >>>\t" << getTimeStr(tsStart) << "\n";
		std::cout << ">>> FINISHED >>>\t" << getTimeStr(tsEnd) << "\n";
		std::cout << "\tGROSS ELAPSED [s]\t" << Nyxus::getTimeDiff(tsStart, tsEnd) << "\n"
		);

	return 0;
}

void showCmdlineHelp()
{
	std::cout 
		<< "Command line format:" << std::endl 
		<< "\t" << PROJECT_NAME << " <label directory> <intensity directory> <output directory> [--minOnlineROI <# of pixels>] [--threads <# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


