#pragma once

#include <climits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "featureset.h"
#include "feature_method.h"
#include "feature_mgr.h"
#include "image_loader.h"
#include "roi_cache.h"

namespace Nyxus
{
	extern FeatureManager theFeatureMgr;
	extern ImageLoader theImLoader;

	bool datasetDirsOK(const std::string& dirIntens, const std::string& dirLab, const std::string& dirOut, bool mustCheckDirOut);
	bool directoryExists(const std::string& dir);
	void readDirectoryFiles(const std::string& dir, std::vector<std::string>& files);
	bool scanFilePairParallel(const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads, int filepair_index, int tot_num_filepairs);
	std::string getPureFname(std::string fpath);
	int processDataset(const std::vector<std::string>& intensFiles, const std::vector<std::string>& labelFiles, int numFastloaderThreads, int numSensemakerThreads, int numReduceThreads, int min_online_roi_size, bool save2csv, const std::string& csvOutputDir);
	bool gatherRoisMetrics(const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads);
	bool processTrivialRois (const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, size_t memory_limit);
	bool processNontrivialRois (const std::vector<int>& nontrivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads);
	void dump_roi_metrics();

	// 2 scenarios of saving a result of feature calculation of a label-intensity file pair: saving to a CSV-file and saving to a matrix to be later consumed by a Python endpoint
	bool save_features_2_csv (std::string intFpath, std::string segFpath, std::string outputDir);
	bool save_features_2_buffer (std::vector<std::string>& headerBuf, std::vector<double>& resultMatrix, std::vector<std::string>& stringColBuf);

	int read_dataset (
		// input
		const std::string& dirIntens, 
		const std::string& dirLabels, 
		const std::string& dirOut, 
		const std::string& intLabMappingDir,
		const std::string& intLabMappingFile,		
		bool mustCheckDirOut,
		// output
		std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles);

	void init_feature_buffers();
	void clear_feature_buffers();	

	void update_label(int x, int y, int label, PixIntens intensity);
	void update_label_parallel(int x, int y, int label, PixIntens intensity);

	void print_label_stats();
	void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8);
	void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);

	void reduce_by_feature (int nThr, int min_online_roi_size);
	void reduce_by_roi (int nThr, int min_online_roi_size);
	void reduce_trivial_rois (std::vector<int>& PendingRoisLabels);
	void reduce_trivial_rois_manual (std::vector<int>& PendingRoisLabels);
	void reduce_neighbors();

	void init_label_record(LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity);
	void init_label_record_2(LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity, unsigned int tile_index);
	void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity);
	void update_label_record_2(LR& lr, int x, int y, int label, PixIntens intensity, unsigned int tile_index);
	void reduce_neighbors(int labels_collision_radius);

	// Label data
	extern std::string theSegFname, theIntFname;	// Cached file names while iterating a dataset
	extern std::unordered_set<int> uniqueLabels;
	extern std::unordered_map <int, LR> roiData;
	extern std::vector<double> calcResultBuf;	// [# of labels X # of features]
	extern std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;

	// Ugly hack fix me.
	extern std::vector<std::string> stringColBuf, headerBuf;
	extern size_t totalNumFeatures, totalNumLabels;

	// System resources
	unsigned long long getAvailPhysMemory();

} // namespace Nyxus

