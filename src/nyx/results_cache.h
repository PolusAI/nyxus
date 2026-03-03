#pragma once
#include <cmath>
#include <limits>
#include <string>
#include <vector>

/// @brief Holds a spatial feature map for one parent ROI: one 2D/3D array per feature.
struct FmapArrayResult
{
	int parent_label;
	std::string intens_name;
	std::string seg_name;
	int map_w, map_h;        // dimensions of the feature map arrays
	int map_d = 1;           // depth dimension (1 for 2D feature maps)
	int origin_x, origin_y;  // global image coords of the map's top-left pixel
	int origin_z = 0;        // global z-coord of the map's first slice (unused in 2D)
	std::vector<std::string> feature_names;
	// Flat storage: n_features contiguous blocks of (map_d * map_h * map_w) doubles each.
	// 2D access: feature_data[f * map_h * map_w + row * map_w + col]
	// 3D access: feature_data[f * map_d * map_h * map_w + z * map_h * map_w + row * map_w + col]
	std::vector<double> feature_data;
};

class ResultsCache
{
public:

	ResultsCache() {}

	void clear()
	{
		headerBuf_.clear();
		stringColBuf_.clear();
		calcResultBuf_.clear();
		totalNumLabels_ = 0;
		fmapArrayResults_.clear();
	}

	std::vector<std::string>& get_headerBuf() { return headerBuf_; }
	std::vector<std::string>& get_stringColBuf() { return stringColBuf_; }
	std::vector<double>& get_calcResultBuf() { return calcResultBuf_; }

	void add_to_header(std::initializer_list<std::string> cols)
	{
		for (auto c : cols)
			add_to_header(c);
	}
	void add_to_header(const std::string& col)
	{
		headerBuf_.push_back(col);
	}

	void add_string (const std::string& s) { stringColBuf_.push_back(s); }
	void add_numeric(double n) { calcResultBuf_.push_back(n); }
	void inc_num_rows() { totalNumLabels_++; }
	size_t get_num_rows() { return totalNumLabels_; }

	std::vector<FmapArrayResult>& get_fmapArrayResults() { return fmapArrayResults_; }

private:

	std::vector<double> calcResultBuf_;
	size_t totalNumLabels_ = 0;
	std::vector<std::string> stringColBuf_, headerBuf_;
	std::vector<FmapArrayResult> fmapArrayResults_;
};
