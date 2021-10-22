#pragma once

#include <string>
#include <vector>

// Command line arguments
#define SEGDIR	"--segDir"	// Environment :: labels_dir
#define INTDIR	"--intDir"	// Environment :: intensity_dir
#define OUTDIR	"--outDir"	// Environment :: output_dir
#define FEATURES	"--features"	// Environment :: features	-- Example: (1) --features=area,kurtosis,neighbors (2) --features=featurefile.txt
#define FILEPATTERN	"--filePattern"	// Environment :: file_pattern
#define OUTPUTTYPE	"--csvFile"	// Environment :: bool separateCsv <= valid values "separatecsv" or "singlecsv" 
#define EMBPIXSZ	"--embeddedpixelsize"	// Environment :: embedded_pixel_size
#define LOADERTHREADS	"--loaderThreads"	// Environment :: n_loader_threads
#define PXLSCANTHREADS	"--pxlscanThreads"	// Environment :: n_pixel_scan_threads
#define REDUCETHREADS	"--reduceThreads"	// Environment :: n_reduce_threads
#define ROTATIONS	"--rotations"	// Environment :: rotAngles
#define VERBOSITY	"--verbosity"	// Environment :: verbosity_level	-- Example: --verbosity=3
#define ONLINESTATSTHRESH	"--onlineStatsThresh"	// Environment :: onlineStatsThreshold	-- Example: --onlineStatsThresh=150

// Feature group nicknames
#define FEA_NICK_ALL "*ALL*"
#define FEA_NICK_ALL_INTENSITY "*ALL_INTENSITY*"
#define FEA_NICK_ALL_MORPHOLOGY "*ALL_MORPHOLOGY*"
#define FEA_NICK_ALL_GLCM "*ALL_GLCM*"
#define FEA_NICK_ALL_GLRLM "*ALL_GLRLM*"
#define FEA_NICK_ALL_GLSZM "*ALL_GLSZM*"
#define FEA_NICK_ALL_GLDM "*ALL_GLDM*"
#define FEA_NICK_ALL_NGTDM "*ALL_NGTDM*"

// Valid values of 'OUTPUTTYPE'
#define OT_SEPCSV "separatecsv"
#define OT_SINGLECSV "singlecsv"

// Verbosity levels (combinable via binary and)
#define VERBOSITY_TIMING	2
#define VERBOSITY_ROI_INFO	4
#define VERBOSITY_DETAILED	8

class Environment
{
public:

	Environment() {}
	int parse_cmdline (int argc, char** argv);
	void show_help();
	void show_summary(const std::string& head, const std::string& tail);

	std::string labels_dir = "",
		intensity_dir = "",
		output_dir = "";

	bool singleROI = false;	// is set to 'true' parse_cmdline() if labels_dir==intensity_dir

	std::string file_pattern = ".*";
	std::string embedded_pixel_size = "";

	std::string features;
	std::vector<std::string> desiredFeatures;
	std::string featureFileName = "";

	std::string loader_threads = "";
	int n_loader_threads = 2;

	std::string pixel_scan_threads = "";
	int n_pixel_scan_threads = 1;

	std::string reduce_threads = "";
	int n_reduce_threads = 16;

	std::string rotations = "";
	std::vector<float> rotAngles = {0, 45, 90, 135};

	std::string verbosity = "";
	int verbosity_level = 0;	// 0 = silent

	std::string rawOnlineStatsThresh = "";
	int onlineStatsTreshold = 0;

	std::string rawOutpType = "";	// Valid values: "separatecsv" or "singlecsv"
	bool separateCsv = true;

protected:

	bool check_file_pattern (const std::string & pat);

	std::vector<std::tuple<std::string, std::string>> memory;
	void show_memory(const std::string& head, const std::string& tail);

	bool find_string_argument (std::vector<std::string>::iterator& i, const char* arg, std::string& arg_value);
	bool find_int_argument (std::vector<std::string>::iterator& i, const char* arg, int& arg_value);
};

extern Environment theEnvironment;