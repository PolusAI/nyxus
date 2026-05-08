#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include "constants.h"
#include "globals.h"
#include "environment.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/zernike.h"
#include "helpers/fsystem.h"

namespace Nyxus
{
	static std::mutex mx1;

	/// @brief Writes feature header columns for all enabled features to a ResultsCache.
	/// Uses get_feature_column_names() as the single source of truth.
	void write_feature_header_buffer (
		ResultsCache & rescache,
		Environment & env,
		const std::vector<std::tuple<std::string, int>> & F)
	{
		auto cols = get_feature_column_names(env, F);
		for (const auto& c : cols)
			rescache.add_to_header(c);
	}

	/// @brief Writes all feature values for a single ROI to a ResultsCache.
	/// Uses collect_feature_values() as the single source of truth.
	void write_feature_values_buffer (
		ResultsCache & rescache,
		const LR & r,
		const std::vector<std::tuple<std::string, int>> & F,
		Environment & env)
	{
		auto vals = collect_feature_values(r, F);
		for (auto v : vals)
			rescache.add_numeric(Nyxus::force_finite_number(v, env.resultOptions.noval()));
	}

	bool save_features_2_buffer_wholeslide (
		// out
		ResultsCache & rescache, 
		// in
		Environment & env,
		const LR & r,
		const std::string& ifpath,
		const std::string& mfpath)
	{
		std::lock_guard<std::mutex> lg (mx1);

		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		// We only fill in the header once.
		// We depend on the caller to manage headerBuf contents and clear it appropriately...
		bool fill_header = rescache.get_headerBuf().size() == 0;

		// -- Header

		if (fill_header)
		{
			rescache.add_to_header({ Nyxus::colname_intensity_image, Nyxus::colname_mask_image, Nyxus::colname_roi_label, Nyxus::colname_t_index });
			write_feature_header_buffer(rescache, env, F);
		}

		// -- Values

		rescache.inc_num_rows();

		// - slide info
		rescache.add_string (ifpath);
		rescache.add_string (mfpath);
		rescache.add_numeric (r.label);
		rescache.add_numeric (DEFAULT_T_INDEX);

		// - features
		write_feature_values_buffer(rescache, r, F, env);

		return true;
	}

	/// @brief Copies ROIs' feature values into a ResultsCache structure that will then shape them as a table
	bool save_features_2_buffer (ResultsCache& rescache, Environment & env, size_t t_index)
	{
		std::vector<int> L{ env.uniqueLabels.begin(), env.uniqueLabels.end() };
		std::sort(L.begin(), L.end());
		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		// We only fill in the header once.
		// We depend on the caller to manage headerBuf contents and clear it appropriately...
		bool fill_header = rescache.get_headerBuf().size() == 0;

		// -- Header
		if (fill_header)
		{
			rescache.add_to_header({ Nyxus::colname_intensity_image, Nyxus::colname_mask_image, Nyxus::colname_roi_label, Nyxus::colname_t_index });
			write_feature_header_buffer(rescache, env, F);
		}

		// -- Values
		for (auto l : L)
		{
			LR& r = env.roiData[l];

			// Skip blacklisted ROI
			if (r.blacklisted)
				continue;

			rescache.inc_num_rows();

			// Tear off pure file names from segment and intensity file paths
			const SlideProps & slide = env.dataset.dataset_props[r.slide_idx];
			fs::path pseg(slide.fname_seg),
				pint(slide.fname_int);
			std::string segfname = pseg.filename().string(),
				intfname = pint.filename().string();

			rescache.add_string (intfname);
			rescache.add_string (segfname);
			rescache.add_numeric (l);
			rescache.add_numeric (t_index);

			write_feature_values_buffer(rescache, r, F, env);
		}

		return true;
	}

	/// @brief Assembles child ROI features into spatial arrays for one parent ROI.
	/// Produces one FmapArrayResult with n_features arrays of shape (map_d, map_h, map_w).
	/// Map dimensions are parent_dim - kernel_size + 1 (the valid convolution output size).
	/// Positions with no valid child ROI remain NaN.
	void save_features_2_fmap_arrays (
		ResultsCache & rescache,
		Environment & env,
		const std::string & intens_name,
		const std::string & seg_name,
		int parent_label,
		int parent_xmin, int parent_ymin, int parent_zmin,
		int parent_w, int parent_h, int parent_d,
		int kernel_size,
		const std::unordered_set<int> & childLabels,
		const std::unordered_map<int, LR> & childRoiData,
		const std::unordered_map<int, FmapChildInfo> & childToParentMap)
	{
		int half = kernel_size / 2;
		// Map dimensions = valid convolution output size
		int map_w = parent_w - kernel_size + 1;
		int map_h = parent_h - kernel_size + 1;
		int map_d = parent_d - kernel_size + 1;

		if (map_w <= 0 || map_h <= 0 || map_d <= 0)
			return;

		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();
		std::vector<std::string> feature_names = get_feature_column_names(env, F);
		size_t n_features = feature_names.size();
		size_t map_size = (size_t)map_d * map_h * map_w;

		// Initialize with NaN
		std::vector<double> feature_data(n_features * map_size, std::numeric_limits<double>::quiet_NaN());

		// Fill in values from child ROIs
		for (auto l : childLabels)
		{
			auto it_roi = childRoiData.find(l);
			if (it_roi == childRoiData.end())
				continue;
			const LR& r = it_roi->second;
			if (r.blacklisted)
				continue;

			auto it_map = childToParentMap.find(l);
			if (it_map == childToParentMap.end())
				continue;

			const FmapChildInfo& info = it_map->second;

			// Convert global center coords to map indices
			int map_col = info.center_x - parent_xmin - half;
			int map_row = info.center_y - parent_ymin - half;
			int map_z   = info.center_z - parent_zmin - half;

			if (map_col < 0 || map_col >= map_w ||
				map_row < 0 || map_row >= map_h ||
				map_z < 0 || map_z >= map_d)
				continue;

			size_t voxel_idx = (size_t)map_z * map_h * map_w + (size_t)map_row * map_w + map_col;

			// Collect feature values for this child
			auto vals = collect_feature_values(r, F);
			for (size_t fi = 0; fi < n_features && fi < vals.size(); fi++)
				feature_data[fi * map_size + voxel_idx] = force_finite_number(vals[fi], env.resultOptions.noval());
		}

		// Store result
		FmapArrayResult result;
		result.parent_label = parent_label;
		result.intens_name = intens_name;
		result.seg_name = seg_name;
		result.map_w = map_w;
		result.map_h = map_h;
		result.map_d = map_d;
		result.origin_x = parent_xmin + half;
		result.origin_y = parent_ymin + half;
		result.origin_z = parent_zmin + half;
		result.feature_names = std::move(feature_names);
		result.feature_data = std::move(feature_data);
		rescache.get_fmapArrayResults().push_back(std::move(result));
	}

}
