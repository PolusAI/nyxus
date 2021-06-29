#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

bool datasetDirsOK (std::string & dirIntens, std::string & dirLab, std::string & dirOut);
bool directoryExists (const std::string & dir);
void readDirectoryFiles (const std::string & dir, std::vector<std::string> & files);
bool scanViaFastloader (const std::string & fpath, int num_threads);
bool TraverseViaFastloader1 (const std::string& fpath, int num_threads);
std::string getPureFname(std::string fpath);
int ingestDataset (std::vector<std::string> & intensFiles, std::vector<std::string> & labelFiles, int numFastloaderThreads, std::string outputDir);
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

void update_label_stats (int x, int y, int label, PixIntens intensity);
void print_label_stats();
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8); 
void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);
void clearLabelStats();
void do_partial_stats_reduction();

// The following label data relates to a single intensity-label file pair
extern std::unordered_set<int> uniqueLabels;
extern std::unordered_map <int, StatsInt> labelCount;
extern std::unordered_map <int, StatsInt> labelPrevIntens;
extern std::unordered_map <int, StatsReal> labelMeans;
extern std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
extern std::unordered_map <int, StatsInt> labelMedians;
extern std::unordered_map <int, StatsInt> labelMins;
extern std::unordered_map <int, StatsInt> labelMaxs;
extern std::unordered_map <int, StatsInt> labelMassEnergy;
extern std::unordered_map <int, StatsReal> labelVariance;
extern std::unordered_map <int, StatsReal> labelCentroid_x;
extern std::unordered_map <int, StatsReal> labelCentroid_y;
extern std::unordered_map <int, StatsReal> labelM2;
extern std::unordered_map <int, StatsReal> labelM3;
extern std::unordered_map <int, StatsReal> labelM4;
extern std::unordered_map <int, StatsReal> labelSkewness;
extern std::unordered_map <int, StatsReal> labelKurtosis;
extern std::unordered_map <int, StatsReal> labelMAD;

