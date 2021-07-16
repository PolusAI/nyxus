#pragma once

//---	#define SINGLE_ROI_TEST

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "histogram.h"

bool datasetDirsOK (std::string & dirIntens, std::string & dirLab, std::string & dirOut);
bool directoryExists (const std::string & dir);
void readDirectoryFiles (const std::string & dir, std::vector<std::string> & files);
bool scanViaFastloader (const std::string & fpath, int num_threads);
bool scanFilePair (const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads);
bool scanFilePairParallel (const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads);
bool TraverseViaFastloader1 (const std::string& fpath, int num_threads);
std::string getPureFname(std::string fpath);
int ingestDataset (std::vector<std::string> & intensFiles, std::vector<std::string> & labelFiles, int numFastloaderThreads, int numSensemakerThreads, std::string outputDir);
bool save_features (std::string inputFpath, std::string outputDir);
void showCmdlineHelp();
int checkAndReadDataset(
	// input
	std::string& dirIntens, std::string& dirLabels, std::string& dirOut,
	// output
	std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles);

using PixIntens = unsigned int;
using StatsInt = unsigned long;
using StatsReal = double;
using Histo = OnlineHistogram<PixIntens>;

void init_feature_buffers();
void update_label_stats (int x, int y, int label, PixIntens intensity);
void update_label_stats_parallel (int x, int y, int label, PixIntens intensity);
void print_label_stats();
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8); 
void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);
void clearLabelStats();
void do_partial_stats_reduction();

// The following label data relates to a single intensity-label file pair
extern std::unordered_set <int> uniqueLabels;


// Label record - structure aggregating label's running statistics and sums
struct LR
{
	StatsInt labelCount;
	StatsInt labelPrevCount;
	StatsInt labelPrevIntens;
	StatsReal labelMeans;
	std::shared_ptr<std::unordered_set<PixIntens>> labelUniqueIntensityValues;
	StatsInt labelMedians;
	StatsInt labelMins;
	StatsInt labelMaxs;
	StatsInt labelMassEnergy;
	StatsReal labelVariance;
	StatsReal labelStddev;	
	StatsReal labelCentroid_x;
	StatsReal labelCentroid_y;
	StatsReal labelM2;
	StatsReal labelM3;
	StatsReal labelM4;
	StatsReal labelSkewness;
	StatsReal labelKurtosis;
	StatsReal labelMAD;
	StatsReal labelRMS;
	std::shared_ptr<Histo> labelHistogram;
	StatsReal labelP10;
	StatsReal labelP25;
	StatsReal labelP75;
	StatsReal labelP90;
	StatsReal labelIQR;
	StatsReal labelEntropy;
	StatsReal labelMode;
	StatsReal labelUniformity;
	StatsReal labelRMAD;
};

extern std::unordered_map <int, LR> labelData;

extern std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;

// Research
extern StatsReal intensityMin, intensityMax;

// Timing
extern double totalTileLoadTime, totalPixStatsCalcTime;
double test_containers1();
double test_containers2();
