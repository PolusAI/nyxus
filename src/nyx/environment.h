#pragma once

#include <string>
#include <vector>

// Command line arguments
#define SEGDIR	"--segDir"	// Environment :: labels_dir
#define INTDIR	"--intDir"	// Environment :: intensity_dir
#define OUTDIR	"--outDir"	// Environment :: output_dir
#define FEATURES	"--features"	// Environment :: features	-- Example: (1) --features=area,kurtosis,neighbors (2) --features=featurefile.txt
#define FILEPATTERN	"--filePattern"	// Environment :: file_pattern
#define CSVFILE	"--outputType"	// Environment :: csv_file
#define EMBPIXSZ	"--embeddedpixelsize"	// Environment :: embedded_pixel_size
#define LOADERTHREADS	"--loaderThreads"	// Environment :: n_loader_threads
#define PXLSCANTHREADS	"--pxlscanThreads"	// Environment :: n_pixel_scan_threads
#define REDUCETHREADS	"--reduceThreads"	// Environment :: n_reduce_threads
#define ROTATIONS	"--rotations"	// Environment :: rotAngles
#define VERBOSITY	"--verbosity"	// Environment :: verbosity_level	-- Example: --verbosity=3

// Verbosity levels (combinable via binary and)
#define VERBOSITY_TIMING	2
#define VERBOSITY_ROI_INFO	4

class Environment
{
public:

	Environment() {}
	int parse_cmdline (int argc, char** argv);
	void show_help();
	void show_summary(const std::string& head, const std::string& tail);

	std::string labels_dir = "",
		intensity_dir = "",
		output_dir = "",
		features = "all",
		file_pattern = "",
		embedded_pixel_size = "",
		csv_file = "separatecsv";

	std::string loader_threads = "";
	int n_loader_threads = 1;

	std::string pixel_scan_threads = "";
	int n_pixel_scan_threads = 1;

	std::string reduce_threads = "";
	int n_reduce_threads = 1;

	std::string rotations = "";
	std::vector<float> rotAngles = {0, 45, 90, 135};

	std::string verbosity = "";
	int verbosity_level = 0;	// 0 = silent

protected:

	std::vector<std::tuple<std::string, std::string>> memory;
	void show_memory(const std::string& head, const std::string& tail);

	void find_string_argument (std::vector<std::string>::iterator& i, const char* arg, std::string& arg_value);
	bool find_int_argument (std::vector<std::string>::iterator& i, const char* arg, int& arg_value);
};

extern Environment theEnvironment;