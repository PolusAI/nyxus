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
#include "save_option.h"

#include "nested_feature_aggregation.h" // Nested ROI

#include "cli_nested_roi_options.h"

#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

namespace Nyxus
{
	// Permanent column names of the feature output table
	const char colname_intensity_image[] = "intensity_image",
		colname_mask_image[] = "mask_image",
		colname_roi_label[] = "ROI_label";

	// Global instances
	extern FeatureManager theFeatureMgr;
	extern ImageLoader theImLoader;

	bool scanFilePairParallel(const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads, int filepair_index, int tot_num_filepairs);
	std::string getPureFname(const std::string& fpath);
//	int processDataset(const std::vector<std::string>& intensFiles, const std::vector<std::string>& labelFiles, int numFastloaderThreads, int numSensemakerThreads, int numReduceThreads, int min_online_roi_size, const SaveOption saveOption, const std::string& outputPath);
//	int processDataset_3D(const std::vector <Imgfile3D_layoutA>& intensFiles, const std::vector <Imgfile3D_layoutA>& labelFiles, int numFastloaderThreads, int numSensemakerThreads, int numReduceThreads, int min_online_roi_size, const SaveOption saveOption, const std::string& outputPath);
	bool gatherRoisMetrics(const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads);
	bool gatherRoisMetrics_3D(const std::string& intens_fpath, const std::string& label_fpath, const std::vector<std::string>& z_indices);	
	bool processTrivialRois(const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, size_t memory_limit);
	bool processTrivialRois_3D(const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit, const std::vector<std::string>& z_indices);
	bool processNontrivialRois (const std::vector<int>& nontrivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads);
	void dump_roi_metrics(const std::string & label_fpath);
	void dump_roi_pixels(const std::vector<int> & batch_labels, const std::string & label_fpath);

	// Shows a message in CLI ('send_to_stderr': stdout or stderr) or Python terminal
	void sureprint(const std::string& msg, bool send_to_stderr=false);

	// in memory functions
#ifdef WITH_PYTHON_H
	bool gatherRoisMetricsInMemory (const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_image, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_image, int start_idx);
	bool processIntSegImagePairInMemory (const std::string& intens_fpath, const std::string& label_fpath, int filepair_index, const std::string& intens_name, const std::string& seg_name);
	int processMontage(const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intensFiles, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& labelFiles, int numReduceThreads, const std::vector<std::string>& intensity_names,
		const std::vector<std::string>& seg_names, std::string& error_message, const SaveOption saveOption,  const std::string& outputPath="");
	bool scanTrivialRois (const std::vector<int>& batch_labels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_images, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images, int start_idx);
	bool processTrivialRoisInMemory (const std::vector<int>& trivRoiLabels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_fpath, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_fpath, int start_idx, size_t memory_limit);
#endif

	// 2 scenarios of saving a result of feature calculation of a label-intensity file pair: saving to a CSV-file and saving to a matrix to be later consumed by a Python endpoint
	std::string get_feature_output_fname(const std::string& intFpath, const std::string& segFpath);
	extern const std::vector<std::string> mandatory_output_columns;
	bool save_features_2_csv (const std::string & intFpath, const std::string & segFpath, const std::string & outputDir);
	bool save_features_2_buffer (ResultsCache& results_cache);	
	
	std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>> get_feature_values();	
	std::vector<std::string> get_header(const std::vector<std::tuple<std::string, int>>& F );
	std::string get_arrow_filename(const std::string& output_path, const std::string& default_filename, const SaveOption& arrow_file_type);

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
	void init_label_record_3D (LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int z, int label, PixIntens intensity, unsigned int tile_index);
	void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity);
	void update_label_record_2(LR& lr, int x, int y, int label, PixIntens intensity, unsigned int tile_index);
	void update_label_record_3D (LR& lr, int x, int y, int z, int label, PixIntens intensity, unsigned int tile_index);

	void allocateTrivialRoisBuffers(const std::vector<int>& roi_labels);
	void allocateTrivialRoisBuffers_3D(const std::vector<int>& roi_labels);
	void freeTrivialRoisBuffers(const std::vector<int>& roi_labels);
	void freeTrivialRoisBuffers_3D(const std::vector<int>& roi_labels);

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
	void feed_pixel_2_metrics_3D (int x, int y, int z, PixIntens intensity, int label, unsigned int tile_index);

	/// @brief Copies a pixel to the ROI's cache. 
	/// @param x -- x-coordinate of the pixel in the image
	/// @param y -- y-coordinate of the pixel in the image
	/// @param label -- label of pixel's segment 
	/// @param intensity -- pixel's intensity
	void feed_pixel_2_cache(int x, int y, PixIntens intensity, int label);
	void feed_pixel_2_cache_3D(int x, int y, int z, PixIntens intensity, int label);

	// System resources
	unsigned long long getAvailPhysMemory();

	// Nested ROI

	using NestableRois = std::unordered_map<int, NestedLR>;
	extern std::unordered_map <std::string, NestableRois> nestedRoiData;
	void save_nested_roi_info(std::unordered_map <std::string, NestableRois>& dst_nestedRoiData, const std::unordered_set<int>& src_labels, std::unordered_map <int, LR>& src_roiData);

	bool mine_segment_relations2(
		const std::vector <std::string>& label_files,
		const std::string& file_pattern,
		const std::string& channel_signature,
		const int parent_channel,
		const int child_channel,
		const std::string& outdir,
		const NestedRoiOptions::Aggregations& aggr,
		int verbosity_level);

} // namespace Nyxus