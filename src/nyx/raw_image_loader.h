#pragma once

#include <array>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "raw_tiff.h"

class RawImageLoader
{
public:

	RawImageLoader();
	bool open(const std::string& int_fpath, const std::string& seg_fpath);
	void close();
	bool load_tile(size_t tile_idx);
	bool load_tile(size_t tile_row, size_t tile_col);

	uint32_t get_cur_tile_seg_pixel(size_t pixel_idx);
	double get_cur_tile_dpequiv_pixel(size_t idx);

	size_t get_tile_size();
	size_t get_num_tiles_vert();
	size_t get_num_tiles_hor();
	size_t get_tile_height();
	size_t get_tile_width();
	size_t get_tile_x(size_t pixel_col);
	size_t get_tile_y(size_t pixel_row);
	size_t get_within_tile_idx(size_t pixel_row, size_t pixel_col);
	size_t get_full_width();
	size_t get_full_height();
	std::string get_slide_descr();

private:
	
	RawFormatLoader* segFL = nullptr, * intFL = nullptr;	// RawTiffTileLoader, RawOmezarrLoader, RawDicomLoader

	// Tile height, width, and depth
	size_t th,
		tw,
		td;

	// 2D tile size
	size_t tileSize;

	// Full height, width, and depth
	size_t fh,
		fw,
		fd;

	// Number of tiles along width, height, and depth
	size_t ntw,
		nth,
		ntd;

	int lvl = 0,	// Pyramid level
		lyr = 0;	//	Layer

};

