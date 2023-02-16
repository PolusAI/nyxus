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
#include "results_cache.h"
#include "roi_cache.h"

namespace Nyxus
{
	extern FeatureManager theFeatureMgr;
	extern ImageLoader theImLoader;

	bool scanFilePairParallel(const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads, int filepair_index, int tot_num_filepairs);
	std::string getPureFname(const std::string& fpath);
	int processDataset(const std::vector<std::string>& intensFiles, const std::vector<std::string>& labelFiles, int numFastloaderThreads, int numSensemakerThreads, int numReduceThreads, int min_online_roi_size, bool save2csv, const std::string& csvOutputDir);
	bool gatherRoisMetrics(const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads);
	bool processTrivialRois (const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, size_t memory_limit);
	bool processNontrivialRois (const std::vector<int>& nontrivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads);
	void dump_roi_metrics(const std::string & label_fpath);

	// 2 scenarios of saving a result of feature calculation of a label-intensity file pair: saving to a CSV-file and saving to a matrix to be later consumed by a Python endpoint
	bool save_features_2_csv (std::string intFpath, std::string segFpath, std::string outputDir);
	bool save_features_2_buffer (ResultsCache& results_cache);		

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
	void reduce_neighbors_and_dependencies_manual ();

	void init_label_record(LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity);
	void init_label_record_2(LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity, unsigned int tile_index);
	void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity);
	void update_label_record_2(LR& lr, int x, int y, int label, PixIntens intensity, unsigned int tile_index);

	void allocateTrivialRoisBuffers(const std::vector<int>& Pending);
	void freeTrivialRoisBuffers(const std::vector<int>& Pending);

	// Label data
	extern std::string theSegFname, theIntFname;	// Cached file names while iterating a dataset
	extern std::unordered_set<int> uniqueLabels;
	extern std::unordered_map <int, LR> roiData;
	extern size_t zero_background_area;
	extern std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;

	/// @brief Feeds a pixel to image measurement object to gauge the image RAM footprint without caching the pixel. Updates 'uniqueLabels' and 'roiData'.
	/// @param x -- x-coordinate of the pixel in the image
	/// @param y -- y-coordinate of the pixel in the image
	/// @param label -- label of pixel's segment 
	/// @param intensity -- pixel's intensity
	/// @param tile_index -- index of pixel's tile in the image
	void feed_pixel_2_metrics(int x, int y, PixIntens intensity, int label, unsigned int tile_index);

	/// @brief Copies a pixel to the ROI's cache. 
	/// @param x -- x-coordinate of the pixel in the image
	/// @param y -- y-coordinate of the pixel in the image
	/// @param label -- label of pixel's segment 
	/// @param intensity -- pixel's intensity
	void feed_pixel_2_cache(int x, int y, PixIntens intensity, int label);

	// System resources
	unsigned long long getAvailPhysMemory();

} // namespace Nyxus

