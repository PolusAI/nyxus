#pragma once

#include <array>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "abs_tile_loader.h"

/// @brief Incapsulates access to an intensity and mask image file pair
class ImageLoader1x
{
public:
	ImageLoader1x();
	bool open(const std::string& fpath);
	void close();
	bool load_tile(size_t tile_idx);
	bool load_tile(size_t tile_row, size_t tile_col);
	const std::vector<uint32_t>& get_tile_buffer();
	size_t get_tile_size();
	size_t get_num_tiles_vert();
	size_t get_num_tiles_hor();
	size_t get_tile_height();
	size_t get_tile_width();
	size_t get_tile_x(size_t pixel_col);
	size_t get_tile_y(size_t pixel_row);
	size_t get_within_tile_idx(size_t pixel_row, size_t pixel_col);
private:
	std::unique_ptr<AbstractTileLoader<uint32_t>> FL = nullptr;
	std::shared_ptr<std::vector<uint32_t>> ptr = nullptr;
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

	size_t lvl = 0,	// Pyramid level
		lyr = 0;	//	Layer
};
