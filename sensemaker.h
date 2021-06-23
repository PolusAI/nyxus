#pragma once

#include <string>
#include <vector>

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

void updateLabelStats (int label, int intensity);
void printLabelStats();