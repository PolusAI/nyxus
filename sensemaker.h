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
int ingestDataset (std::vector<std::string> & intensFiles, std::vector<std::string> & labelFiles, int numFastloaderThreads);
void showCmdlineHelp();
int checkAndReadDataset(
	// input
	std::string& dirIntens, std::string& dirLabels, std::string& dirOut,
	// output
	std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles);

using PixIntens = unsigned int;
using StatsInt = unsigned long;

void update_label_stats (int label, PixIntens intensity);
void print_label_stats();
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8); 
void clearLabelStats();
void do_partial_stats_reduction();

extern std::unordered_map <int, StatsInt> labelCounts;
extern std::unordered_map <int, StatsInt> labelMeans;
extern std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
extern std::unordered_map <int, StatsInt> labelMedians;
extern std::unordered_map <int, StatsInt> labelMins;
extern std::unordered_map <int, StatsInt> labelMaxs;
extern std::unordered_map <int, StatsInt> labelEnergy;
