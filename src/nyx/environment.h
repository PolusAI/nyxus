#pragma once

#include <map>
#include <string>
#include <vector>
#include "environment_basic.h"
#include "roi_blacklist.h"
#include "cli_gabor_options.h"

#ifdef USE_GPU
	#include <cuda_runtime.h>
#endif

// Command line arguments
#define SEGDIR "--segDir"						// Environment :: labels_dir
#define INTDIR "--intDir"						// Environment :: intensity_dir
#define OUTDIR "--outDir"						// Environment :: output_dir
#define INTSEGMAPDIR "--intSegMapDir"			// get_int_seg_map_dir()
#define INTSEGMAPFILE "--intSegMapFile"			// get_int_seg_map_file()
#define FEATURES "--features"					// Environment :: features	-- Example: (1) --features=area,kurtosis,neighbors (2) --features=featurefile.txt
#define FILEPATTERN "--filePattern"				// Environment :: file_pattern
#define OUTPUTTYPE "--csvFile"					// Environment :: bool separateCsv <= valid values "separatecsv" or "singlecsv"
#define EMBPIXSZ "--embeddedpixelsize"			// Environment :: embedded_pixel_size
#define LOADERTHREADS "--loaderThreads"			// Environment :: n_loader_threads
#define PXLSCANTHREADS "--pxlscanThreads"		// Environment :: n_pixel_scan_threads
#define REDUCETHREADS "--reduceThreads"			// Environment :: n_reduce_threads
#define GLCMANGLES "--glcmAngles"					// Environment :: rotAngles
#define VERBOSITY "--verbose"					// Environment :: verbosity_level	-- Example: --verbosity=3
#define ONLINESTATSTHRESH "--onlineStatsThresh" // Environment :: onlineStatsThreshold	-- Example: --onlineStatsThresh=150
#define XYRESOLUTION "--pixelsPerCentimeter"	// pixels per centimeter
#define PXLDIST "--pixelDistance"				// used in neighbor features
#define COARSEGRAYDEPTH "--coarseGrayDepth"		// Environment :: raw_coarse_grayscale_depth
#define RAMLIMIT "--ramLimit"					// Optional. Limit for treating ROIs as non-trivial and for setting the batch size of trivial ROIs. Default - amount of available system RAM
#define TEMPDIR "--tempDir"						// Optional. Used in processing non-trivial features. Default - system temp directory
#define IBSICOMPLIANCE "--ibsi" // skip binning for grey level and grey tone features
#define SKIPROI "--skiproi"		// Optional. Skip ROIs having specified labels. Sybtax: --skiproi <label[,label,label,...]>

#ifdef CHECKTIMING
	#define EXCLUSIVETIMING "--exclusivetiming"
#endif

#ifdef USE_GPU
	#define USEGPU "--useGpu"					// Environment::rawUseGpu, "true" or "false"
	#define GPUDEVICEID "--gpuDeviceID"		// Environment::rawGpuDeviceID
#endif

// Gabor feature CLI arguments
#define GABOR_FREQS "--gaborfreqs"		// Example: "2,4,8,72"
#define GABOR_GAMMA "--gaborgamma"		// Example: "0.1"
#define GABOR_SIG2LAM "--gaborsig2lam"	// Example: "0.8"
#define GABOR_KERSIZE "--gaborkersize"	// Example: "20"
#define GABOR_F0 "--gaborf0"			// Example: "0.1"
#define GABOR_THETA "--gabortheta"		// Example: "60"
#define GABOR_THRESHOLD "--gaborthold"	// Example: "0.025"

// Feature group nicknames
#define FEA_NICK_ALL "*ALL*"
#define FEA_NICK_ALL_INTENSITY "*ALL_INTENSITY*"
#define FEA_NICK_ALL_MORPHOLOGY "*ALL_MORPHOLOGY*"
#define FEA_NICK_BASIC_MORPHOLOGY "*BASIC_MORPHOLOGY*"
#define FEA_NICK_ALL_GLCM "*ALL_GLCM*"
#define FEA_NICK_ALL_GLRLM "*ALL_GLRLM*"
#define FEA_NICK_ALL_GLSZM "*ALL_GLSZM*"
#define FEA_NICK_ALL_GLDM "*ALL_GLDM*"
#define FEA_NICK_ALL_NGTDM "*ALL_NGTDM*"
#define FEA_NICK_ALL_BUT_GABOR "*ALL_BUT_GABOR*"
#define FEA_NICK_ALL_BUT_GLCM "*ALL_BUT_GLCM*"
#define FEA_NICK_ALL_EASY "*ALL_EASY*"	// Equivalent to *ALL* minus GABOR, GLCM, and 2D moments
#define FEA_NICK_ALL_NEIG "*ALL_NEIGHBOR*"	

// Valid values of 'OUTPUTTYPE'
#define OT_SEPCSV "separatecsv"
#define OT_SINGLECSV "singlecsv"

// Verbosity levels (combinable via binary and)
#define VERBOSITY_TIMING 2
#define VERBOSITY_ROI_INFO 4
#define VERBOSITY_DETAILED 8

/// @brief Class encapsulating the the feature extraction environment - command line option values, default values, etc. Use it to add a parseable command line parameter.
class Environment: public BasicEnvironment
{
public:
	Environment();
	bool parse_cmdline(int argc, char **argv);
	void show_cmdline_help();
	void show_featureset_help(); 
	void show_summary(const std::string &head, const std::string &tail);

	std::string labels_dir = "",
				intensity_dir = "",
				output_dir = "",
				intSegMapDir = "",
				intSegMapFile = "";

	bool singleROI = false; // is set to 'true' parse_cmdline() if labels_dir==intensity_dir

	std::string embedded_pixel_size = "";

	std::string rawFeatures;
	std::vector<std::string> recognizedFeatureNames;

	std::string loader_threads = "";
	int n_loader_threads = 1;

	std::string pixel_scan_threads = "";
	int n_pixel_scan_threads = 1;

	std::string reduce_threads = "";
	int n_reduce_threads = 4;

	std::string pixel_distance = "";
	int n_pixel_distance = 5;

	std::string rawGlcmAngles = "";
	std::vector<int> glcmAngles = {0, 45, 90, 135};

	std::string verbosity = "";	// 'verbosity_level' is inherited from BasicEnvironment

	std::string rawOnlineStatsThresh = "";
	int onlineStatsTreshold = 0;

	std::string rawOutpType = ""; // Valid values: "separatecsv" or "singlecsv"
	bool separateCsv = true;

	// x- and y- resolution in pixels per centimeter
	std::string rawXYRes = "";
	float xyRes = 0.0,
		  pixelSizeUm = 0.0;

	int get_pixel_distance();
	void set_pixel_distance(int pixelDistance);
	size_t get_ram_limit();
	void process_feature_list();

	static bool gpu_is_available();

	static bool ibsi_compliance;
	static std::string raw_ibsi_compliance; // string for input

	void set_ibsi_compliance(bool skip);

#ifdef USE_GPU
	/// @brief Returns GPU device ID of choice
	/// @return 0-based GPU device ID (default: 0) or -1 not to use GPU even if it is available
	int get_gpu_device_choice();
	void set_gpu_device_id(int choice);
	void set_use_gpu(bool yes);
	bool using_gpu();	
	static std::vector<std::map<std::string, std::string>> get_gpu_properties();
#endif

	int get_floating_point_precision();

	unsigned int get_coarse_gray_depth();
	void set_coarse_gray_depth(unsigned int new_depth);

	// implementation of SKIPROI
	bool roi_is_blacklisted (const std::string& fname, int roi_label);
	bool parse_roi_blacklist_raw_string (const std::string& raw_blacklist_string, std::string& error_message);
	void clear_roi_blacklist ();
	void get_roi_blacklist_summary(std::string& response);

	// implementation of Gabor feature options
	bool parse_gabor_options_raw_inputs (std::string& error_message);
	GaborOptions gaborOptions;

private:

	std::vector<std::tuple<std::string, std::string>> recognizedArgs;	// Accepted command line arguments

	bool find_string_argument (std::vector<std::string>::iterator &i, const char *arg, std::string &arg_value);
	bool find_int_argument(std::vector<std::string>::iterator &i, const char *arg, int &arg_value);

	std::string rawTempDirPath = "";

#ifdef USE_GPU
	std::string rawUseGpu = "";		// boolean
	bool use_gpu_ = false;
	std::string rawGpuDeviceID = "";		// integer
	int gpu_device_id_ = -1;
	std::vector<std::map<std::string, std::string>> gpu_props_;
#endif

	int floating_point_precision = 10;	

	unsigned int coarse_grayscale_depth = 256;
	std::string raw_coarse_grayscale_depth = "";

	// data members implementing RAMLIMIT
	std::string rawRamLimit = "";
	size_t ramLimit = 0;

	// data members implementing TEMPDIR
	std::string rawTempDir = "";

	// implementation of SKIPROI
	std::string rawBlacklistedRois = "";
	RoiBlacklist roiBlacklist;

	// data members implementing exclusive-inclusive timing switch
	#ifdef CHECKTIMING
		std::string rawExclusiveTiming = "";
	#endif
};

namespace Nyxus
{
	extern Environment theEnvironment;
}

#define VERBOSLVL1(stmt) if(Nyxus::theEnvironment.get_verbosity_level()>=1){stmt;}
#define VERBOSLVL2(stmt) if(Nyxus::theEnvironment.get_verbosity_level()>=2){stmt;}
#define VERBOSLVL3(stmt) if(Nyxus::theEnvironment.get_verbosity_level()>=3){stmt;}
#define VERBOSLVL4(stmt) if(Nyxus::theEnvironment.get_verbosity_level()>=4){stmt;}	
