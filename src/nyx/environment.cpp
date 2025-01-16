#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>
#include <exception>
#include <iterator>
#include "environment.h"
#include "featureset.h"
#include "features/glcm.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "version.h"

#ifdef USE_GPU
	std::vector<std::map<std::string, std::string>> get_gpu_properties();
#endif

namespace Nyxus
{
	bool existsOnFilesystem(const std::string&);

	std::string strip_punctn_and_comment(const std::string& src)
	{
		std::string s = src;
		bool commentState = false;
		for (size_t i = 0; i < s.size(); i++)
		{
			if (s[i] == '#')
			{
				commentState = true;
				s.erase(i, 1); // remove ith char from string
				i--; // reduce i with one so you don't miss any char
				continue;
			}

			if (commentState)
			{
				s.erase(i, 1); // remove ith char from string
				i--; // reduce i with one so you don't miss any char
				continue;
			}

			if (s[i] == '\n')
			{
				commentState = false;
				s.erase(i, 1); // remove ith char from string
				i--; // reduce i with one so you don't miss any char
				continue;
			}

			if (!((std::isalnum(s[i]) || s[i] == '_') || s[i] == ','))
			{
				s.erase(i, 1); // remove ith char from string
				i--; // reduce i with one so you don't miss any char
				continue;
			}
		}

		return s;
	}
}

using namespace Nyxus;

bool Environment::ibsi_compliance = false;
std::string Environment::raw_ibsi_compliance = ""; // string for input

Environment::Environment() : BasicEnvironment()
{
	unsigned long long availMem = Nyxus::getAvailPhysMemory();
	ramLimit = availMem / 2;	// Safely require 50% of available memory
}

size_t Environment::get_ram_limit()
{
	return ramLimit;
}

int Environment::get_pixel_distance()
{
	return n_pixel_distance;
}

void Environment::set_pixel_distance(int pixelDistance)
{
	this->n_pixel_distance = pixelDistance;
}

void Environment::set_ibsi_compliance(bool skip) {
	this->ibsi_compliance = skip;
}

void Environment::show_cmdline_help()
{
	const char OPT[] = "(optional) ";
	std::cout
		<< "Usage:\t" << "nyxus"
		<< "\t" << FILEPATTERN << "=<file pattern regular expression> \n"
		<< "\t\t\tDefault: .* \n"
		<< "\t\t\tExample: " << FILEPATTERN << "=.* for all files, " << FILEPATTERN << "=*.tif for .tif files \n"
		<< "\t\t" << OUTPUTTYPE << "=<separatecsv or singlecsv for csv output. arrowipc or parquet for arrow output> \n"
		<< "\t\t\tDefault: separatecsv \n"
		<< "\t\t" << SEGDIR << "=<directory of segmentation mask images> \n"
		<< "\t\t" << INTDIR << "=<directory of intensity images> \n"
		<< "\t\t" << OUTDIR << "=<output directory> \n"
		<< "\t\t" << OPT << FEATURES << "=<specific feature or comma-separated features or feature group> \n"
		<< "\t\t\tDefault: " << Nyxus::theFeatureSet.findGroupNameByCode(Nyxus::Fgroup2D::FG2_ALL) << " \n"
		<< "\t\t\tExample 1: " << FEATURES << "=" << theFeatureSet.findFeatureNameByCode(Feature2D::PERIMETER) << " \n"
		<< "\t\t\tExample 2: " << FEATURES << "=" << theFeatureSet.findFeatureNameByCode(Feature2D::PERIMETER) << "," << theFeatureSet.findFeatureNameByCode(Feature2D::CIRCULARITY) << "," << theFeatureSet.findFeatureNameByCode(Feature2D::GABOR) << " \n"
		<< "\t\t\tExample 3: " << FEATURES << "=" << Nyxus::theFeatureSet.findGroupNameByCode(Nyxus::Fgroup2D::FG2_INTENSITY) << " \n"
		<< "\t\t\tExample 4: " << FEATURES << "=" << Nyxus::theFeatureSet.findGroupNameByCode(Nyxus::Fgroup2D::FG2_BASIC_MORPHOLOGY) << " \n"
		<< "\t\t\tExample 5: " << FEATURES << "=" << Nyxus::theFeatureSet.findGroupNameByCode(Nyxus::Fgroup2D::FG2_ALL) << " \n"
		<< "\t\t" << OPT << XYRESOLUTION << "=<number of pixels per centimeter, an integer or floating point number> \n"
		<< "\t\t" << OPT << EMBPIXSZ << "=<number> \n"
		<< "\t\t\tDefault: 0 \n"
		<< "\t\t" << OPT << LOADERTHREADS << "=<number of image loader threads> \n"
		<< "\t\t\tDefault: 1 \n"
		<< "\t\t" << OPT << REDUCETHREADS << "=<number of feature reduction threads> \n"
		<< "\t\t\tDefault: 1 \n"
		<< "\t\t" << OPT << PXLDIST << "=<number of pixels as neighbor features radius> \n"
		<< "\t\t\tDefault: 5 \n"
		<< "\t\t" << OPT << COARSEGRAYDEPTH << "=<custom number of grayscale levels> \n"
		<< "\t\t\tDefault: 64 \n"
		<< "\t\t" << OPT << GLCMANGLES << "=<one or more comma separated rotation angles from set {0, 45, 90, and 135}> \n"
		<< "\t\t\tDefault: 0,45,90,135 \n"
		<< "\t\t" << OPT << VERBOSITY << "=<levels of verbosity 0 (silence), 1 (minimum output), 2 (1 + timing), 3 (2 + roi metrics + more timing), 4 (3 + diagnostic information)> \n"
		<< "\t\t\tDefault: 0 \n"
		<< "\t\t" << OPT << IBSICOMPLIANCE << "=<false or true> Enable IBSI compliance mode \n"
		<< "\t\t\tDefault: false \n"
		<< "\t\t\tNote: " << IBSICOMPLIANCE << "=true may reduce performance \n"
		<< "\t\t" << OPT << RAMLIMIT << "=<megabytes> \n"
		<< "\t\t\tDefault: 50% of available memory \n"
		<< "\t\t" << OPT << TEMPDIR << "=<slash-terminating temporary directory path> \n"
		<< "\t\t\tDefault: default system temp directory \n"
		<< "\t\t" << OPT << SKIPROI << "=<ROI blacklist> \n"
		<< "\t\t\tDefault: void blacklist \n"
		<< "\t\t\tExample 1: " << SKIPROI << "=34,35,36 \n"
		<< "\t\t\tExample 2: " << SKIPROI << "=image1.ome.tif:34,35,36;image2.ome.tif:42,43 \n"
		<< "\t\t" << OPT << RESULTFNAME << "=<file name without extension> \n"
		<< "\t\t\tDefault: NyxusFeatures \n"
		<< "\t\t\tExample: " << RESULTFNAME << "=training_set_features";

#ifdef CHECKTIMING
	std::cout << "\t\t" << OPT << EXCLUSIVETIMING << "=<false or true> \n"
		<< "\t\t\tDefault: false \n"
		<< "\t\t\tUse " << EXCLUSIVETIMING << "=false to measure time of the whole image collection, " << EXCLUSIVETIMING << "=true to measure time per image pair \n";
#endif

	std::cout << "\t\t" << OPT << GABOR_FREQS << "=<comma separated denominators of \\pi> \n"
		<< "\t\t\tDefault: 1,2,4,8,16,32,64 \n"
		<< "\t\t" << OPT << GABOR_GAMMA << "=<anisotropy of the Gaussian> \n"
		<< "\t\t\tDefault: 0.1 \n"
		<< "\t\t" << OPT << GABOR_SIG2LAM << "=<spatial frequency bandwidth (sigma over lambda)> \n"
		<< "\t\t\tDefault: 0.8 \n"
		<< "\t\t" << OPT << GABOR_KERSIZE << "=<dimension of the 2D kernel> \n"
		<< "\t\t\tDefault: 16 \n"
		<< "\t\t" << OPT << GABOR_F0 << "=<frequency of the baseline lowpass filter as denominator of \\pi> \n"
		<< "\t\t\tDefault: 0.1 \n"
		<< "\t\t" << OPT << GABOR_THETA << "=<orientation of the Gaussian in degrees 0-180> \n"
		<< "\t\t\tDefault: 45 \n"
		<< "\t\t" << OPT << GABOR_THRESHOLD << "=<lower threshold of the filtered image to baseline ratio> \n"
		<< "\t\t\tDefault: 0.025 \n";

	std::cout << "\t\t" << OPT << NESTEDROI_CHNL_SIGNATURE << "=<comma separated denominators of \\pi> \n"
		<< "\t\t" << OPT << NESTEDROI_PARENT_CHNL << "=<number of the parent channel e.g. 1\n"
		<< "\t\t" << OPT << NESTEDROI_CHILD_CHNL << "=<number of the child channel e.g. 0\n"
		<< "\t\t" << OPT << NESTEDROI_AGGREGATION_METHOD << "=<SUM, MEAN, MIN, MAX, WMA, or NONE>\n"
		<< ""
		<< "\t\t\tDefault: 0.1 \n";

	std::cout << "\n"
		<< "\tnyxus -h\tDisplay help info\n"
		<< "\tnyxus --help\tDisplay help info\n";

#ifdef USE_GPU
	std::cout << " [" << USEGPU << "=<true or false>" << " [" << GPUDEVICEID << "=<comma separated GPU device ID>] ]\n";
#endif
}

void Environment::show_summary(const std::string& head, const std::string& tail)
{
	std::cout << head;
	std::cout << "Using " << get_ram_limit() << " bytes of memory\n\n";
	std::cout << "Work plan:\n"
		<< "\tdimensionality: " << dim() << "\n"
		<< "\tlabels\t" << labels_dir << "\n"
		<< "\tintensities\t" << intensity_dir << "\n"
		<< "\tintensities-to-segmentation map directory\t" << intSegMapDir << "\n"
		<< "\tintensities-to-segmentation map file\t" << intSegMapFile << "\n"
		<< "\toutput\t" << output_dir << "\n"
		<< "\tfile pattern\t" << rawFilePattern << "\n"
		<< "\tembedded pixel size\t" << embedded_pixel_size << "\n"
		<< "\toutput type\t" << rawOutpType << "\n"
		<< "\t# of image loader threads\t" << n_loader_threads << "\n"
		<< "\t# of pixel scanner threads\t" << n_pixel_scan_threads << "\n"
		<< "\t# of post-processing threads\t" << n_reduce_threads << "\n"
		<< "\tpixel distance\t" << n_pixel_distance << "\n"
		<< "\tverbosity level\t" << verbosity_level << "\n";

#ifdef USE_GPU
	std::cout << "\tusing GPU\t" << (gpuOptions.get_using_gpu() ? "yes" : "no") << "\n";
	if (gpuOptions.get_using_gpu())
		std::cout << "\trequested GPU device IDs: " << (gpuOptions.raw_requested_device_ids.empty() ? "(blank)" : gpuOptions.raw_requested_device_ids) << "\n"
			"\tbest GPU device ID: " << gpuOptions.get_single_device_id() << "\n";
#endif

	// Features
	std::cout << "\tfeatures\t";
	for (auto f : recognizedFeatureNames)
	{
		if (f != recognizedFeatureNames[0])
			std::cout << ", ";
		std::cout << f;
	}
	std::cout << "\n";

	// Resolution
	if (xyRes > 0.0)
		std::cout << "\tXY-resolution " << xyRes << "\n";

	// GLCM angles
	std::cout << "\tGLCM angles\t";
	for (auto ang : GLCMFeature::angles)
	{
		if (ang != GLCMFeature::angles[0])
			std::cout << ", ";
		std::cout << ang;
	}
	std::cout << "\n";

	// Oversized ROI limit
	std::cout << "\tbatch and oversized ROI lower limit " << theEnvironment.get_ram_limit() << " bytes\n";

	// Temp directory
	std::cout << "\ttemp directory " << theEnvironment.get_temp_dir_path() << "\n";

	// Blacklisted ROIs
	if (roiBlacklist.defined())
		std::cout << "\tblacklisted ROI " << roiBlacklist.get_summary_text() << "\n";

	// Timing mode
#if CHECKTIMING
	std::cout << "\t#CHECKTIMING / exclusive mode of timing " << (Stopwatch::exclusive() ? "TRUE" : "FALSE") << "\n";
#endif

	if (!gaborOptions.empty())
		std::cout << "\tGabor feature options: " << gaborOptions.get_summary_text() << "\n";

	// Real valued TIFF
	if (!fpimageOptions.empty())
		std::cout << "\tImage-wide expected \n" << fpimageOptions.get_summary_text() << "\n";
	std::cout << tail;
}

bool Environment::find_string_argument(std::vector<std::string>::iterator& i, const char* arg, std::string& arg_value)
{
	std::string actualArgName = *i;

	// Syntax #1 <arg> <value>
	std::string a = arg;
	if (actualArgName == a)
	{
		if (arg_value.length())
			std::cerr << "Warning: " << a << " already has value \'" << arg_value << "\'\n";

		arg_value = *++i;
		recognizedArgs.push_back({ a, arg_value });
		return true;
	}
	else
	{
		// Syntax #2 <arg>=<value>
		a.append("=");
		auto pos = actualArgName.find(a);
		if (pos != std::string::npos)
		{
			if (arg_value.length())
				std::cerr << "Warning: " << a << " already has value \'" << arg_value << "\'\n";

			arg_value = actualArgName.substr(a.length());
			recognizedArgs.push_back({ a, arg_value });
			return true;
		}
	}

	// Argument was not recognized
	return false;
}

bool Environment::find_int_argument(std::vector<std::string>::iterator& i, const char* arg, int& arg_value)
{
	// Syntax #1
	std::string a = arg;
	if (*i == a)
	{
		std::string val = *++i;
		// string -> integer
		if (sscanf(val.c_str(), "%d", &arg_value) != 1)
			return true;
	}
	else
	{
		// Syntax #2
		a.append("=");
		auto pos = (*i).find(a);
		if (pos != std::string::npos)
		{
			std::string val = (*i).substr(a.length());
			// string -> integer
			if (sscanf(val.c_str(), "%d", &arg_value) != 1)
				return true;
		}
	}

	// Argument was not recognized
	return false;
}



/**
 * @brief Parses the command line. Caller needn't display command line help screen to the user after call
 *
 * @return true - success, false - error, execution should not continue
 */
bool Environment::parse_cmdline(int argc, char** argv)
{
	// Program being run without any flags and options?
	if (argc == 1)
	{
		std::cerr << "Error: missing command line arguments \n";
		show_cmdline_help();
		return false;
	}

	std::vector<std::string> args(argv + 1, argv + argc);
	std::vector<std::string> unrecognizedArgs;

	//==== Gather command line arguments as raw strings
	for (auto i = args.begin(); i != args.end(); ++i)
	{
		if (*i == "-h" || *i == "--help")
		{
			show_cmdline_help();
			return false;	// Option is valid but we return false to stop execution
		}

		if (!(
			find_string_argument(i, INTDIR, intensity_dir) ||
			find_string_argument(i, SEGDIR, labels_dir) ||
			find_string_argument(i, OUTDIR, output_dir) ||
			find_string_argument(i, INTSEGMAPDIR, intSegMapDir) ||
			find_string_argument(i, INTSEGMAPFILE, intSegMapFile) ||
			find_string_argument(i, FEATURES, rawFeatures) ||
			find_string_argument(i, XYRESOLUTION, rawXYRes) ||
			find_string_argument(i, FILEPATTERN, rawFilePattern) ||
			find_string_argument(i, OUTPUTTYPE, rawOutpType) ||
			find_string_argument(i, EMBPIXSZ, embedded_pixel_size) ||
			find_string_argument(i, LOADERTHREADS, loader_threads) ||
			find_string_argument(i, PXLSCANTHREADS, pixel_scan_threads) ||
			find_string_argument(i, REDUCETHREADS, reduce_threads) ||
			find_string_argument(i, GLCMANGLES, glcmOptions.rawAngles) ||
			find_string_argument(i, GLCMOFFSET, glcmOptions.rawOffs) ||
			find_string_argument(i, PXLDIST, pixel_distance) ||
			find_string_argument(i, COARSEGRAYDEPTH, raw_coarse_grayscale_depth) ||
			find_string_argument(i, VERBOSITY, rawVerbosity) ||
			find_string_argument(i, IBSICOMPLIANCE, raw_ibsi_compliance) ||
			find_string_argument(i, RAMLIMIT, rawRamLimit) ||
			find_string_argument(i, TEMPDIR, rawTempDir) ||
			find_string_argument(i, SKIPROI, rawBlacklistedRois) ||
			find_string_argument(i, GABOR_FREQS, gaborOptions.rawFreqs) ||
			find_string_argument(i, GABOR_GAMMA, gaborOptions.rawGamma) ||
			find_string_argument(i, GABOR_SIG2LAM, gaborOptions.rawSig2lam) ||
			find_string_argument(i, GABOR_KERSIZE, gaborOptions.rawKerSize) ||
			find_string_argument(i, GABOR_F0, gaborOptions.rawF0) ||
			find_string_argument(i, GABOR_THETA, gaborOptions.rawTheta) ||
			find_string_argument(i, GABOR_THRESHOLD, gaborOptions.rawGrayThreshold)
			|| find_string_argument(i, NESTEDROI_CHNL_SIGNATURE, nestedOptions.rawChannelSignature)
			|| find_string_argument(i, NESTEDROI_PARENT_CHNL, nestedOptions.rawParentChannelNo)
			|| find_string_argument(i, NESTEDROI_CHILD_CHNL, nestedOptions.rawChildChannelNo)
			|| find_string_argument(i, NESTEDROI_AGGREGATION_METHOD, nestedOptions.rawAggregationMethod)
			|| find_string_argument(i, FPIMAGE_TARGET_DYNRANGE, fpimageOptions.raw_target_dyn_range)
			|| find_string_argument(i, FPIMAGE_MIN, fpimageOptions.raw_min_intensity)
			|| find_string_argument(i, FPIMAGE_MAX, fpimageOptions.raw_max_intensity)
			|| find_string_argument(i, RESULTFNAME, nyxus_result_fname)
			|| find_string_argument(i, CLI_DIM, raw_dim)

#ifdef CHECKTIMING
			|| find_string_argument(i, EXCLUSIVETIMING, rawExclusiveTiming)
#endif

#ifdef USE_GPU
			|| find_string_argument(i, USEGPU, gpuOptions.raw_use_gpu)
			|| find_string_argument(i, GPUDEVICEID, gpuOptions.raw_requested_device_ids)
#endif
			))
			unrecognizedArgs.push_back(*i);
	}

	//==== Show the user recognized and unrecognized command line elements

	// --include the raw command line
	std::stringstream rawCL;
	rawCL << "\nCommand line:\n" << argv[0] << " ";
	std::copy(args.begin(), args.end(), std::ostream_iterator<std::string>(rawCL, " ")); // vector of strings -> string
	rawCL << "\n\n";

	std::cout << "Recognized command line arguments:\n";
	for (auto& m : recognizedArgs)
		std::cout << "\t" << std::get<0>(m) << " : " << std::get<1>(m) << "\n";
	
	std::cout << "\n";

	// --what's not recognized?
	if (unrecognizedArgs.size() > 0)
	{
		std::cerr << "\nError - unrecognized arguments:\n";
		for (auto& u : unrecognizedArgs)
			std::cerr << "\t" << u << "\n";
		return false;
	}

	//==== Check mandatory parameters

	// -- file pattern

	if (rawFilePattern == "")
	{
		std::cerr << "Error: Missing argument " << FILEPATTERN << "\n";
		return false;
	}

	// -- dimensionality

	if (!raw_dim.empty())
	{
		// string -> integer
		if (sscanf(raw_dim.c_str(), "%d", &dim_) != 1 || !(dim_==2 || dim_==3))
		{
			std::cerr << "Error: " << CLI_DIM << "=" << raw_dim << ": expecting 2 or 3\n";
			return false;
		}
	}
	else
		set_dim(2);	// No user's input about dimensionality, so default it to 2D
	switch (dim())
	{
		case 2:
			if (check_2d_file_pattern(rawFilePattern) == false)
			{
				std::cerr << "Error: invalid file pattern '" << rawFilePattern << "' \n";
				return false;
			}
			break;
		case 3:
			if (check_3d_file_pattern(rawFilePattern) == false)
			{
				std::cerr << "Error: invalid 3D file pattern " << rawFilePattern << " : " << this->file_pattern_3D.get_ermsg() << '\n';
				return false;
			}
			break;
	}

	// -- directories

	if (labels_dir == "")
	{
		std::cerr << "Error: Missing argument " << SEGDIR << "\n";
		return false;
	}
	if (intensity_dir == "")
	{
		std::cerr << "Error: Missing argument " << INTDIR << "\n";
		return false;
	}
	if (output_dir == "")
	{
		std::cerr << "Error: Missing argument " << OUTDIR << "\n";
		return false;
	}

	//==== whole-slide mode
	if (Nyxus::toupper(labels_dir) == Nyxus::toupper(intensity_dir))
	{
		singleROI = true;
	}


	if (rawOutpType == "")
	{
		std::cerr << "Error: Missing argument " << OUTPUTTYPE << "\n";
		return false;
	}

	if (rawFeatures == "")
	{
		if (dim() == 3)
			rawFeatures = theFeatureSet.findGroupNameByCode(Fgroup3D::FG3_ALL);
		else
			rawFeatures = theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_ALL);
		std::cerr << "Warning: " << FEATURES << "=<empty string>, defaulting to " << rawFeatures << '\n';
	}

	//==== Parse optional result file name
	if (nyxus_result_fname == "")
	{
		std::cerr << "Error: void argument " << RESULTFNAME << "\n";
		return false;
	}

	//==== Output type
	auto rawOutpTypeUC = Nyxus::toupper(rawOutpType);
	if (!((rawOutpTypeUC == Nyxus::toupper(OT_SINGLECSV)) ||
		(rawOutpTypeUC == Nyxus::toupper(OT_SEPCSV)) ||
		(rawOutpTypeUC == Nyxus::toupper(OT_ARROWIPC)) ||
		(rawOutpTypeUC == Nyxus::toupper(OT_PARQUET))
		))
	{
		std::cerr << "Error: valid values of " << OUTPUTTYPE << " are " << OT_SEPCSV << ", "
			<< OT_SINGLECSV << ", "
#ifdef USE_ARROW
			<< OT_ARROWIPC << ", or" << OT_PARQUET <<
#endif 
			"."  "\n";
		return false;
	}

	SaveOption saveOption = [&rawOutpTypeUC]() {
		if (rawOutpTypeUC == Nyxus::toupper(OT_ARROWIPC)) {
			return SaveOption::saveArrowIPC;
		}
		else if (rawOutpTypeUC == Nyxus::toupper(OT_PARQUET)) {
			return SaveOption::saveParquet;
		}
		else {
			return SaveOption::saveCSV;
		}
	}();

#ifndef USE_ARROW // no Apache support
	if (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet) {
		std::cerr << "Error: Nyxus must be built with Apache Arrow enabled to use Arrow output types. Please rebuild with the flag USEARROW=ON." << std::endl;
		return false;
	}
#endif

	if (saveOption == SaveOption::saveCSV) {
		separateCsv = rawOutpTypeUC == Nyxus::toupper(OT_SEPCSV);
	}

	//==== Check numeric parameters
	if (!loader_threads.empty())
	{
		// string -> integer
		if (sscanf(loader_threads.c_str(), "%d", &n_loader_threads) != 1 || n_loader_threads <= 0)
		{
			std::cerr << "Error: " << LOADERTHREADS << "=" << loader_threads << ": expecting a positive integer constant\n";
			return false;
		}
	}

	if (!pixel_scan_threads.empty())
	{
		// string -> integer
		if (sscanf(pixel_scan_threads.c_str(), "%d", &n_pixel_scan_threads) != 1 || n_pixel_scan_threads <= 0)
		{
			std::cerr << "Error: " << PXLSCANTHREADS << "=" << pixel_scan_threads << ": expecting a positive integer constant\n";
			return false;
		}
	}

	if (!reduce_threads.empty())
	{
		// string -> integer
		if (sscanf(reduce_threads.c_str(), "%d", &n_reduce_threads) != 1 || n_reduce_threads <= 0)
		{
			std::cerr << "Error: " << REDUCETHREADS << "=" << reduce_threads << ": expecting a positive integer constant\n";
			return false;
		}
	}

	if (!pixel_distance.empty())
	{
		// string -> integer
		if (sscanf(pixel_distance.c_str(), "%d", &n_pixel_distance) != 1 || n_pixel_distance <= 0)
		{
			std::cerr << "Error: " << PXLDIST << "=" << pixel_distance << ": expecting a positive integer constant\n";
			return false;
		}
	}

	// parse COARSEGRAYDEPTH
	if (!raw_coarse_grayscale_depth.empty())
	{
		// string -> integer
		if (sscanf(raw_coarse_grayscale_depth.c_str(), "%d", &coarse_grayscale_depth) != 1)
		{
			std::cerr << "Error: " << COARSEGRAYDEPTH << "=" << raw_coarse_grayscale_depth << ": expecting an integer constant\n";
			return false;
		}
	}

	if (!rawVerbosity.empty())
	{
		// string -> integer
		if (sscanf(rawVerbosity.c_str(), "%d", &verbosity_level) != 1 || verbosity_level < 0)
		{
			std::cerr << "Error: " << VERBOSITY << "=" << reduce_threads << ": expecting a positive integer constant\n";
			return false;
		}
	}

	//==== Parse the RAM limit (in megabytes)
	if (!rawRamLimit.empty())
	{
		// string -> integer
		int value = 0;
		auto scanfResult = sscanf(rawRamLimit.c_str(), "%d", &value);
		if (scanfResult != 1 || value < 0)
		{
			std::cerr << "Error: " << RAMLIMIT << "=" << rawRamLimit << ": expecting a non-negative integer constant (RAM limit in megabytes)\n";
			return false;
		}

		auto success = set_ram_limit(value);

		// Check if it over the actual limit
		unsigned long long actualRam = Nyxus::getAvailPhysMemory();
		if (!success)
		{
			return false;
		}
	}

	//==== Parse the temp directory
	if (!rawTempDir.empty())
	{
		// Check the path
		if (!existsOnFilesystem(rawTempDir))
		{
			std::cerr << "Error :" << TEMPDIR << "=" << rawTempDir << ": nonexisting directory\n";
			return false;
		}

		// Modify the temp directory path
		this->temp_dir_path = rawTempDir + "\\";
	}

	//==== Parse ROI blacklist
	if (!rawBlacklistedRois.empty())
	{
		std::string ermsg;
		if (!this->parse_roi_blacklist_raw_string(rawBlacklistedRois, ermsg))
		{
			std::cerr << ermsg << "\n";
			return false;
		}
	}

	//==== Parse Gabor options
	if (!gaborOptions.empty())
	{
		std::string ermsg;
		if (!this->parse_gabor_options_raw_inputs(ermsg))
		{
			std::cerr << ermsg << "\n";
			return false;
		}
	}

	//==== Parse GLCM options
	if (!glcmOptions.empty())
	{
		std::string ermsg;
		if (!this->parse_glcm_options_raw_inputs(ermsg))
		{
			std::cerr << ermsg << "\n";
			return false;
		}
	}

	//==== Parse floating point image options
	if (!fpimageOptions.empty())
	{
		std::string ermsg;
		if (!this->parse_fpimage_options_raw_inputs(ermsg))
		{
			std::cerr << ermsg << "\n";
			return false;
		}
	}

	//==== Parse nested ROI options
	if (!nestedOptions.empty())
	{
		std::string ermsg;
		if (!this->parse_nested_options_raw_inputs(ermsg))
		{
			std::cerr << ermsg << "\n";
			return false;
		}
	}

	//==== Parse exclusive-inclusive timing
	#ifdef CHECKTIMING
	if (!rawExclusiveTiming.empty())
	{
		std::transform(rawExclusiveTiming.begin(), rawExclusiveTiming.end(), rawExclusiveTiming.begin(), ::tolower);
		if (rawExclusiveTiming == "true" || rawExclusiveTiming == "1" || rawExclusiveTiming == "on")
			Stopwatch::set_inclusive(false);
		else
			Stopwatch::set_inclusive(true);
	}
	#endif

	//==== Using GPU
	#ifdef USE_GPU
	if (!gpuOptions.empty())
	{
		std::string ermsg;
		if (!gpuOptions.parse_input(ermsg))
		{
			std::cerr << ermsg << "\n";
			return false;
		}
	}
	#endif

	//==== Parse desired features

	// --Try to pick up features from a text file treating 'rawFeatures' as a feature file path-name
	if (rawFeatures.length() > 0 && Nyxus::existsOnFilesystem(rawFeatures))
	{
		std::ifstream file(rawFeatures);
		std::string ln, featureList;
		while (std::getline(file, ln))
		{
			// Strip punctuation and comment text
			std::string pureLn = strip_punctn_and_comment(ln);

			// Skip empty strings
			if (pureLn.length() == 0)
				continue;

			// Consume the purified feature name
			// --insert comma after 1st item
			if (featureList.length() > 0)
				// --no need for inserted comma if the item came with its own comma
				if (pureLn[pureLn.size() - 1] != ',')
					featureList += ",";
			featureList += pureLn;
		}

		std::cout << "Using features [" << featureList << "] from file " << rawFeatures << "\n";

		// Modify the input string
		rawFeatures = featureList;
	}

	// -- uppercase it (class FeatureSet understands uppercase names)
	rawFeatures = Nyxus::toupper (rawFeatures);
	
	// --Make sure all the feature names are correct
	if (!spellcheck_raw_featurelist(rawFeatures, recognizedFeatureNames))
	{
		std::cerr << "Stopping due to errors while parsing user requested features\n";
		return false;
	}

	// --Feature names are ok, now expand feature group nicknames and enable each feature in 'theFeatureSet'
	try
	{
		expand_featuregroups();
	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return false;
	}

	// -- check if any feature is enablesd as a result of expanding user's choice
	if (theFeatureSet.numOfEnabled(dim()) == 0)
	{
		std::cerr << "Error: no features are selected. Stopping \n";
		return false;
	}

	//==== Parse resolution
	if (rawXYRes.length() > 0)
	{
		// string -> number
		if (sscanf(rawXYRes.c_str(), "%f", &xyRes) != 1 || xyRes <= 0)
		{
			std::cerr << "Error: " << XYRESOLUTION << "=" << xyRes << ": expecting a positive numeric constant\n";
			return false;
		}
		// pixel size
		pixelSizeUm = 1e-2f / xyRes / 1e-6f; // 1 cm in meters / pixels per cm / micrometers
	}

	//==== Parse IBSI compliance mode
	std::transform(raw_ibsi_compliance.begin(), raw_ibsi_compliance.end(), raw_ibsi_compliance.begin(), ::tolower);
	if (raw_ibsi_compliance == "true" || raw_ibsi_compliance == "1" || raw_ibsi_compliance == "on")
	{
		ibsi_compliance = true;
	}
	else
	{
		ibsi_compliance = false;
	}

	// Success
	return true;
}


int Environment::get_floating_point_precision()
{
	return floating_point_precision;
}

int Environment::get_coarse_gray_depth()
{
	return coarse_grayscale_depth;
}

void Environment::set_coarse_gray_depth(unsigned int new_depth)
{
	coarse_grayscale_depth = new_depth;
}

bool Environment::set_ram_limit(size_t megabytes) {

	// Megabytes to bytes
	size_t requestedCeiling = megabytes * 1048576;

	// Check if it over the actual limit
	unsigned long long actualRam = Nyxus::getAvailPhysMemory();
	if (requestedCeiling > actualRam)
	{
		std::cerr << "Error: RAM limit " << megabytes << " megabytes (=" << requestedCeiling << " bytes) exceeds the actual amount of available RAM " << actualRam << " bytes\n";
		return false;
	}

	// Set the member variable
	ramLimit = requestedCeiling;
	return true;
}

bool Environment::gpu_is_available() {
#ifdef USE_GPU
	return get_gpu_properties().size() > 0 ? true : false;
#else
	return false;
#endif
}

bool Environment::parse_roi_blacklist_raw_string(const std::string& rbs, std::string& error_message)
{
	if (!roiBlacklist.parse_raw_string(rbs))
	{
		error_message = roiBlacklist.get_last_er_msg();
		return false;
	}
	return true;
}

void Environment::clear_roi_blacklist()
{
	roiBlacklist.clear();
}

bool Environment::roi_is_blacklisted(const std::string& fname, int label)
{
	bool retval = roiBlacklist.check_label_blacklisted(fname, label);
	return retval;
}

void Environment::get_roi_blacklist_summary(std::string& response)
{
	response = roiBlacklist.get_summary_text();
}

bool Environment::parse_gabor_options_raw_inputs(std::string& error_message)
{
	if (dim() != 2)
	{
		error_message = "Error: Gabor options are not applicable to dimensionality " + std::to_string(dim());
		return false;
	}

	if (!gaborOptions.parse_input())
	{
		error_message = gaborOptions.get_last_er_msg();
		return false;
	}
	return true;
}

bool Environment::parse_glcm_options_raw_inputs (std::string& error_message)
{

	if (!glcmOptions.parse_input())
	{
		error_message = glcmOptions.get_last_er_msg();
		return false;
	}
	return true;
}

bool Environment::parse_fpimage_options_raw_inputs(std::string& error_message)
{
	if (!fpimageOptions.parse_input())
	{
		error_message = fpimageOptions.get_last_er_msg();
		return false;
	}
	return true;
}

bool Environment::parse_nested_options_raw_inputs(std::string& error_message)
{
	if (dim() != 2)
	{
		error_message = "Error: nested options are not applicable to dimensionality " + std::to_string(dim());
		return false;
	}

	if (!nestedOptions.parse_input())
	{
		error_message = nestedOptions.get_last_er_msg();
		return false;
	}
	return true;
}

bool Environment::arrow_is_enabled()
{
#ifdef USE_ARROW
	return true;
#else
	return false;
#endif
}


#ifdef USE_GPU

void Environment::set_gpu_device_id (int choice) 
{
	auto prp = get_gpu_properties();
	auto n= prp.size();
	if (n == 0) 
	{
		std::cerr << "Error: no GPU devices available \n";
		return;
	}

	if (choice >= n) 
	{
		std::cerr << "Warning: GPU choice (" << choice << ") is out of range. Defaulting to device 0 \n";
		gpuOptions.set_single_device_id (0);
	}
	else
		gpuOptions.set_single_device_id (choice);
}

int Environment::get_gpu_device_choice()
{
	if (using_gpu())
		return gpuOptions.get_single_device_id();
	else
		return -1;  // GPU was not requested so return an invalid device ID -1
}

void Environment::set_using_gpu (bool yes)
{
	gpuOptions.set_using_gpu (yes);
}

bool Environment::using_gpu()
{
	return gpuOptions.get_using_gpu();
}

std::vector<std::map<std::string, std::string>> Environment::get_gpu_properties() 
{
	int n_devices;
	std::vector<std::map<std::string, std::string>> props;

	cudaGetDeviceCount(&n_devices);

	for (int i = 0; i < n_devices; ++i) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		std::map<std::string, std::string> temp;

		temp["Device number"] = std::to_string(i);
		temp["Device name"] = prop.name;
		temp["Memory"] = std::to_string(prop.totalGlobalMem / pow(2, 30)) + " GB";
		temp["Capability"] = std::to_string(prop.major) + std::to_string(prop.minor);

		props.push_back(temp);
	}

	return props;
}

#endif

