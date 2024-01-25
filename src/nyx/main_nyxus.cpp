#include <algorithm>
#include "version.h"
#include "dirs_and_files.h"
#include "environment.h"
#include "globals.h"
#include "save_option.h"
#include "arrow_output_stream.h"
#ifdef USE_GPU
	bool gpu_initialize(int dev_id); 
#endif

using namespace Nyxus;

namespace Nyxus {
	int processDataset(
		const std::vector<std::string>& intensFiles, 
		const std::vector<std::string>& labelFiles, 
		int numFastloaderThreads, 
		int numSensemakerThreads, 
		int numReduceThreads, 
		int min_online_roi_size, 
		const SaveOption saveOption, 
		const std::string& outputPath);

	int processDataset_3D(
		const std::vector <Imgfile3D_layoutA>& intensFiles,
		const std::vector <Imgfile3D_layoutA>& labelFiles,
		int numFastloaderThreads,
		int numSensemakerThreads,
		int numReduceThreads,
		int min_online_roi_size,
		const SaveOption saveOption,
		const std::string& outputPath);
};

int main (int argc, char** argv)
{
	VERBOSLVL1 (std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021-2023 Axle Informatics" << " Build of " << __TIMESTAMP__ << "\n")

	if (! theEnvironment.parse_cmdline(argc, argv))
		return 1;

	VERBOSLVL1 (theEnvironment.show_summary("\n"/*head*/, "\n"/*tail*/))

	#ifdef USE_GPU
	if (theEnvironment.using_gpu())
	{
		int gpuDevNo = theEnvironment.get_gpu_device_choice();
		if (gpuDevNo >= 0 && gpu_initialize(gpuDevNo) == false)
		{
			std::cerr << "Error: cannot use GPU device ID " << gpuDevNo << ". You can disable GPU usage via command line option " << USEGPU << "=false\n";
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

	// Current time stamp #1
	auto startTS = getTimeStr();
	VERBOSLVL1(std::cout << "\n>>> STARTING >>> " << startTS << "\n";)
		
	if (theEnvironment.dim() == 2)
	{
		// Scan intensity and mask directories, apply fileppatern, make intensity-mask image file pairs
		std::vector <std::string> intensFiles, labelFiles;
		int errorCode = Nyxus::read_2D_dataset(
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

		// Process the image data
		int min_online_roi_size = 0;

		errorCode = processDataset(
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

			errorCode = processDataset_3D (
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
	VERBOSLVL1(std::cout << "\n>>> STARTED >>>\t" << startTS << "\n>>> FINISHED >>>\t" << getTimeStr() << "\n";)

	return 0;
}

void showCmdlineHelp()
{
	std::cout 
		<< "Command line format:" << std::endl 
		<< "\t" << PROJECT_NAME << " <label directory> <intensity directory> <output directory> [--minOnlineROI <# of pixels>] [--threads <# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


