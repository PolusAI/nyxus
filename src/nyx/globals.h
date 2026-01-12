#pragma once

#include <climits>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "cache.h"
#include "dataset.h"
#include "dirs_and_files.h"
#include "environment.h"
#include "featureset.h"
#include "image_loader.h"
#include "results_cache.h"
#include "roi_cache.h"
#include "save_option.h"
#include "arrow_output_stream.h"
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
	using Uniqueids = std::unordered_set<int>;
	using Roidata = std::unordered_map<int, LR>;

	// Permanent column names of the feature output table
	const char colname_intensity_image[] = "intensity_image",
		colname_mask_image[] = "mask_image",
		colname_roi_label[] = "ROI_label",
		colname_t_index[] = "t_index";

	// segmented 2D workflow
	int processDataset_2D_segmented(
		Environment& env,
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		int numReduceThreads,
		const SaveOption saveOption,
		const std::string& outputPath);

	// single-segment 2D workflow
	int processDataset_2D_wholeslide(
		Environment & env,
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		int n_threads,
		const SaveOption saveOption,
		const std::string& outputPath);

	// segmented 3D workflow
	int processDataset_3D_segmented(
		Environment & env, 
		const std::vector <Imgfile3D_layoutA>& intensFiles,
		const std::vector <Imgfile3D_layoutA>& labelFiles,
		int numReduceThreads,
		const SaveOption saveOption,
		const std::string& outputPath);

	// single-segment 3D workflow
	std::tuple<bool, std::optional<std::string>> processDataset_3D_wholevolume(
		Environment & env,
		const std::vector <std::string>& intensFiles,
		int n_threads,
		const SaveOption saveOption,
		const std::string& outputPath);

	std::string getPureFname(const std::string& fpath);
	bool gatherRoisMetrics(int slide_idx, const std::string& intens_fpath, const std::string& label_fpath, Environment & env, ImageLoader & L);
	bool gather_wholeslide_metrics(const std::string& intens_fpath, ImageLoader& L, LR& roi);
	bool gatherRoisMetrics_25D (Environment& env, size_t sidx, const std::string& intens_fpath, const std::string& mask_fpath, const std::vector<std::string>& z_indices);
	bool gatherRoisMetrics_3D (Environment& env, size_t sidx, const std::string& intens_fpath, const std::string& mask_fpath, size_t t_index);
	bool processTrivialRois (Environment& env, const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit);
	bool processTrivialRois_25D (Environment & env, const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit, const std::vector<std::string>& z_indices);
	bool processTrivialRois_3D (Environment & env, size_t sidx, size_t t_index, const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit);
	bool processNontrivialRois (Environment& env, const std::vector<int>& nontrivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath);
	bool scan_trivial_wholeslide (LR& vroi, const std::string& intens_fpath, ImageLoader& ldr);	// reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi'
	bool scan_trivial_wholeslide_anisotropic (LR& vroi, const std::string& intens_fpath, ImageLoader& ldr, double aniso_x, double aniso_y);

	bool scan_trivial_wholevolume (LR& vroi, const std::string& intens_fpath, ImageLoader& ldr);	
	bool scan_trivial_wholevolume_anisotropic (LR& vroi, const std::string& intens_fpath, ImageLoader& ldr, double aniso_x, double aniso_y, double aniso_z);

	bool scanTrivialRois_3D (Environment& env, const std::vector<int>& batch_labels, const std::string& intens_fpath, const std::string& label_fpath, size_t t_index);
	void dump_roi_metrics (const int dim, const std::string& output_dir, const size_t ram_limit, const std::string& seg_fpath, const Uniqueids& uniqueLabels, const Roidata& roiData);
	void dump_roi_pixels (const int dim, const std::string& output_dir, const std::vector<int>& batch_labels, const std::string& seg_fpath, const Uniqueids& uniqueLabels, const Roidata& roiData);
	void dump_2d_image_with_halfcontour(
		const std::vector<PixIntens>& I, // border image
		const std::list<Pixel2>& unordered, // unordered contour pixels
		const std::vector<Pixel2>& ordered, // already ordered pixels
		const Pixel2& pxTip, // tip of ordering
		const int W,
		const int H,
		const std::string& head,
		const std::string& tail);
	void dump_2d_image_with_vertex_chain(
		const std::vector<PixIntens>& I,
		const std::vector<Pixel2>& V,
		const int W,
		const int H,
		const std::string& head,
		const std::string& tail);
	void dump_2d_image_with_vertex_set (
		const std::vector<PixIntens>& I,
		const std::list<Pixel2>& U,
		const int W,
		const int H,
		const std::string& head,
		const std::string& tail);
	void dump_2d_image_1d_layout(const std::vector<PixIntens>& I, const int W, const int H, const std::string& head, const std::string& tail);
	// Shows a message in CLI ('send_to_stderr': stdout or stderr) or Python terminal
	void sureprint(const std::string& msg, bool send_to_stderr=false);

	// in-memory functions
#ifdef WITH_PYTHON_H
	bool gatherRoisMetricsInMemory (Environment& env, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_image, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_image, int start_idx);
	bool processIntSegImagePairInMemory (const std::string& intens_fpath, const std::string& label_fpath, int filepair_index, const std::string& intens_name, const std::string& seg_name);
	std::optional<std::string> processMontage (Environment& env, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intensFiles, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& labelFiles, int numReduceThreads, const std::vector<std::string>& intensity_names,
		const std::vector<std::string>& seg_names, const SaveOption saveOption,  const std::string& outputPath="");
	bool processTrivialRoisInMemory (Environment& env, const std::vector<int>& trivRoiLabels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_fpath, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_fpath, int start_idx, size_t memory_limit);
	bool scanTrivialRoisInMemory (const std::vector<int>& batch_labels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_images, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images, int start_idx, Environment & env);
#endif

	// 2 scenarios of saving a result of feature calculation of a label-intensity file pair: saving to a CSV-file and saving to a matrix to be later consumed by a Python endpoint
	std::string get_feature_output_fname (Environment& env, const std::string& intFpath, const std::string& segFpath);
	extern const std::vector<std::string> mandatory_output_columns;
	bool save_features_2_csv (Environment & env, const std::string & intFpath, const std::string & segFpath, const std::string & outputDir, size_t t_index, bool need_aggregation);
	bool save_features_2_csv_wholeslide (Environment & env, const LR & r, const std::string & ifpath, const std::string & mfpath, const std::string & outdir, size_t t_index);
	bool save_features_2_buffer (ResultsCache& results_cache, Environment& env);
	bool save_features_2_buffer_wholeslide(
		ResultsCache& rescache,
		Environment& env, 
		const LR& r,
		const std::string& ifpath,
		const std::string& mfpath);
	std::tuple<bool, std::optional<std::string>> save_features_2_apache_wholeslide (Environment & env, const LR & wsi_roi, const std::string & wsi_path);

	std::vector<FTABLE_RECORD> get_feature_values (
		const FeatureSet & user_selected_features, 
		const Uniqueids & uniqueLabels, 
		const Roidata & roiData,
		const Dataset & dataset);

	std::vector<FTABLE_RECORD> get_feature_values_roi (
		const FeatureSet & fset,
		const LR& r,
		const std::string & ifpath,
		const std::string & mfpath);

	std::vector<std::string> get_header (Environment & env);
	std::string get_arrow_filename(const std::string& output_path, const std::string& default_filename, const SaveOption& arrow_file_type);

	void init_slide_rois (std::unordered_set<int> & uniqueLabels, std::unordered_map<int, LR> & roiData);
	void clear_slide_rois (Uniqueids& L, Roidata& D);

	void update_label(int x, int y, int label, PixIntens intensity);
	void update_label_parallel(int x, int y, int label, PixIntens intensity);

	void print_label_stats();
	void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8);
	void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);

	void reduce_by_feature (Environment & env, int nThr, int min_online_roi_size);
	void reduce_trivial_rois (std::vector<int>& PendingRoisLabels);
	void reduce_trivial_rois_manual (std::vector<int>& PendingRoisLabels, Environment & env);
	void reduce_trivial_wholeslide (Environment & env, LR & slideroi);
	void reduce_trivial_3d_wholevolume (Environment & env, LR& r);
	void reduce_neighbors_and_dependencies_manual (Environment & env);

	void init_label_record_hierarchical (LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity);
	void init_label_record_3(LR& lr, int x, int y, PixIntens intensity);
	void init_label_record_3D (
		/*out*/
		LR& roi, 
		/*in*/
		int x, int y, int z, int label, PixIntens intensity);
	void update_label_record_2(LR& lr, int x, int y, int label, PixIntens intensity);
	void update_label_record_3(LR& lr, int x, int y, PixIntens intensity);
	void update_label_record_3D (LR& lr, int x, int y, int z, int label, PixIntens intensity);

	void allocateTrivialRoisBuffers (const std::vector<int>& roi_labels, Roidata& roi_data, CpusideCache& hostside);
	void allocateTrivialRoisBuffers_3D (const std::vector<int>& roi_labels, Roidata& roiData, CpusideCache& hostside);
	void freeTrivialRoisBuffers (const std::vector<int>& roi_labels, Roidata& roi_data);
	void freeTrivialRoisBuffers_3D (const std::vector<int>& roi_labels, Roidata& roi_data);

	// Label data
	/// @brief Feeds a pixel to image measurement object to gauge the image RAM footprint without caching the pixel. Updates 'uniqueLabels' and 'roiData'.
	/// @param x -- x-coordinate of the pixel in the image
	/// @param y -- y-coordinate of the pixel in the image
	/// @param label -- label of pixel's segment 
	/// @param intensity -- pixel's intensity
	void feed_pixel_2_metrics (
		// modified
		std::unordered_set<int> & uniqueLabels,
		std::unordered_map <int, LR> & roiData,
		// in
		int x, int y, PixIntens intensity, int label, int slide_idx);
	void feed_pixel_2_metrics_3D (
		// modified
		std::unordered_set<int> & uniqueLabels,
		std::unordered_map <int, LR> & roiData,
		// in
		int x, int y, int z, PixIntens intensity, int label, int sidx);

	/// @brief Copies a pixel to the ROI's cache. 
	/// @param x -- x-coordinate of the pixel in the image
	/// @param y -- y-coordinate of the pixel in the image
	/// @param label -- label of pixel's segment 
	/// @param intensity -- pixel's intensity
	void feed_pixel_2_cache_LR (int x, int y, PixIntens intensity, LR& r);
	void feed_pixel_2_cache_3D_LR (int x, int y, int z, PixIntens intensity, LR& r);

	// Nested ROI

	using NestableRois = std::unordered_map<int, NestedLR>;
	extern std::unordered_map <std::string, NestableRois> nestedRoiData;
	void save_nested_roi_info(std::unordered_map <std::string, NestableRois>& dst_nestedRoiData, const std::unordered_set<int>& src_labels, std::unordered_map <int, LR>& src_roiData, const Dataset& ds);

	bool mine_segment_relations2 (Environment& env, const std::vector <std::string>& label_files);

} // namespace Nyxus
