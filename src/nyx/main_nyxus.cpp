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
	Environment env;
	VERBOSLVL1 (env.get_verbosity_level(), std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021-2025 Axle Research" << " Build of " << __TIMESTAMP__ << "\n");

	if (! env.parse_cmdline(argc, argv))
		return 1;

	VERBOSLVL1 (env.get_verbosity_level(), env.show_summary());

	#ifdef USE_GPU
	if (env.using_gpu())
	{
		int devid = env.get_gpu_device_choice();
		if (NyxusGpu::gpu_initialize(devid) == false)
		{
			std::cerr << "Error: cannot use GPU device ID " << devid << ". You can disable GPU usage via command line option " << clo_USEGPU << "=false\n";
			return 1;
		}
	}
	#endif

	// Have the feature manager prepare the feature toolset reflecting user's selection
	if (!env.theFeatureMgr.compile())
	{
		std::cerr << "Error: compiling feature methods failed\n";
		return 1;
	}
	env.theFeatureMgr.apply_user_selection (env.theFeatureSet);

	// Current time stamp #1
	auto tsStart = Nyxus::getCurTime();
	VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\n>>> STARTING >>> " << Nyxus::getTimeStr(tsStart) << "\n");

	// Initialize feature classes 
	if (! env.theFeatureMgr.init_feature_classes())
	{
		std::cerr << "Error: initializing feature classes failed\n";
		return 1;
	}

	// prepare feature settings
	env.compile_feature_settings();
		
	if (env.dim() == 2)
	{
		// Scan intensity and mask directories, apply fileppatern, make intensity-mask image file pairs (if not in wholeslide mode)
		std::vector <std::string> intensFiles, labelFiles;
		auto maybeError = Nyxus::read_2D_dataset(
			env.intensity_dir,
			env.labels_dir,
			env.get_file_pattern(),
			env.output_dir,
			env.intSegMapDir,
			env.intSegMapFile,
			true,
			intensFiles, 
			labelFiles);
		if (maybeError.has_value())
		{
			#ifdef WITH_PYTHON_H
				throw std::runtime_error (*maybeError);
			#endif
			std::cerr << "Errors while reading the dataset:\n" << *maybeError << "\n";
			return 1;
		}

		// Process the image data
		int errorCode = 0;
		if (env.singleROI)
		{
			errorCode = processDataset_2D_wholeslide(
				env,
				intensFiles,
				labelFiles,
				env.n_reduce_threads,
				env.saveOption,
				env.output_dir);
		}
		else
		{
			errorCode = processDataset_2D_segmented(
				env,
				intensFiles,
				labelFiles,
				env.n_reduce_threads,
				env.saveOption,
				env.output_dir);
		}

		// Report feature extraction error, if any
		switch (errorCode)
		{
		case 0:		// Success
			break;
		case 1:		// Dataset structure error env.g. intensity-label file name mismatch
			std::cout << std::endl << "Input data error" << std::endl;
			break;
		case 2:		// Internal FastLoader error env.g. TIFF access error
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
		if (env.nestedOptions.defined())
		{
			bool mineOK = mine_segment_relations2(
				env,
				//xxxxxxxxxxxxxx		env.theFeatureSet,
				labelFiles
				//xxxxxxxxxxx		env.get_file_pattern(),
				//xxxxxxxxxxx		env.nestedOptions.rawChannelSignature, //---.channel_signature(),
				//xxxxxxxxxxx		env.nestedOptions.parent_channel_number(),
				//xxxxxxxxxxx		env.nestedOptions.child_channel_number(),
				//xxxxxxxxxxx		env.output_dir,
				//xxxxxxxxxxx		env.nestedOptions.aggregation_method(),
				//xxxxxxxxxxx		env.get_verbosity_level()
				);

			// Report nested ROI errors, if any
			if (!mineOK)
			{
				std::cerr << "Error minimg hierarchical relations\n";
				return 1;
			}
		}
	} // 2D
	else
		if (env.dim() == 3)
		{
			if (env.singleROI)
			{
				std::vector<std::string> ifiles;

				auto mayBerror = Nyxus::read_3D_dataset_wholevolume (
					env.intensity_dir,
					env.file_pattern_3D,
					env.output_dir,
					ifiles);
				if (mayBerror.has_value())
				{
					std::cout << "Dataset error: " + *mayBerror + "\n";
					return 1;
				}

				auto [ok, erm] = processDataset_3D_wholevolume (env, ifiles, env.n_reduce_threads, env.saveOption, env.output_dir);
				if (!ok)
				{
					std::cerr << *erm << "\n";
					return 1;
				}
			}
			else
			{
				// Scan intensity and mask directories, apply fileppatern, make intensity-mask image file pairs
				std::vector <Imgfile3D_layoutA> intensFiles, labelFiles;

				auto mayBerror = Nyxus::read_3D_dataset(
					env.intensity_dir,
					env.labels_dir,
					env.file_pattern_3D,
					env.output_dir,
					env.intSegMapDir,
					env.intSegMapFile,
					true,
					intensFiles,
					labelFiles);
				if (mayBerror.has_value())
				{
					std::cout << "Dataset error: " + *mayBerror + "\n";
					return 1;
				}

				int errorCode = processDataset_3D_segmented(
					env,
					intensFiles,
					labelFiles,
					env.n_reduce_threads,
					env.saveOption,
					env.output_dir);

				// Report feature extraction error, if any
				switch (errorCode)
				{
				case 0:		// Success
					break;
				case 1:		// Dataset structure error env.g. intensity-label file name mismatch
					std::cout << std::endl << "Input data error" << std::endl;
					break;
				case 2:		// Internal FastLoader error env.g. TIFF access error
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
			}

		} // 3D

	// Current time stamp #2
	auto tsEnd = Nyxus::getCurTime();
	VERBOSLVL1(env.get_verbosity_level(),
		std::cout << "\n"
			<< ">>> STARTED  >>>\t" << getTimeStr(tsStart) << "\n"
			<< ">>> FINISHED >>>\t" << getTimeStr(tsEnd) << "\n"
			<< "\tGROSS ELAPSED [s]\t" << Nyxus::getTimeDiff(tsStart, tsEnd) << "\n"
		);

	return 0;
}

void showCmdlineHelp()
{
	std::cout 
		<< "Command line format:" << std::endl 
		<< "\t" << PROJECT_NAME << " <label directory> <intensity directory> <output directory> [--minOnlineROI <# of pixels>] [--threads <# of FastLoader threads> <# of feature calculation threads>]" << std::endl;
}


