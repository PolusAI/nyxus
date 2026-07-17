#pragma once

#include <array>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "abs_tile_loader.h"
#include "cli_fpimage_options.h"
#include "slideprops.h"

/// @brief Incapsulates access to an intensity and mask image file pair
class ImageLoader
{
public:

	ImageLoader();
	bool open (SlideProps & p, const FpImageOptions& fpopts);
	void close();
	bool load_tile (size_t tile_idx);
	bool load_tile (size_t tile_row, size_t tile_col);

	// Assemble the whole X*Y*Z volume for one (channel, timeframe) into an
	// internal buffer by looping over Z-planes. This is what lets plane-by-plane
	// loaders (OME-Zarr, multi-page/OME-TIFF) feed the volumetric pipeline, which
	// otherwise assumes the whole volume arrives in one read (the NIfTI model).
	bool load_volume (size_t channel, size_t timeframe);
	const std::vector<uint32_t>& get_int_volume_buffer() const { return vol_int_; }
	const std::vector<uint32_t>& get_seg_volume_buffer() const { return vol_seg_; }

	const std::vector<uint32_t>& get_int_tile_buffer();
	const std::vector<uint32_t>& get_seg_tile_buffer();
	const std::shared_ptr<std::vector<uint32_t>>& get_seg_tile_sptr();
	size_t get_tile_size();
	size_t get_num_tiles_vert();
	size_t get_num_tiles_hor();
	size_t get_tile_height();
	size_t get_tile_width();
	size_t get_tile_x (size_t pixel_col);
	size_t get_tile_y (size_t pixel_row);
	size_t get_within_tile_idx (size_t pixel_row, size_t pixel_col);
	size_t get_full_width();
	size_t get_full_height();
	size_t get_full_depth();
	size_t get_inten_time();
	size_t get_mask_time();

	// Select which channel (C) / timeframe (T) plane subsequent load_tile() calls
	// read. Default 0/0 preserves the single-channel, single-timepoint behavior.
	void set_channel (size_t c) { cur_channel = c; }
	void set_timeframe (size_t t) { cur_timeframe = t; }
	size_t get_channel() const { return cur_channel; }
	size_t get_timeframe() const { return cur_timeframe; }

private:

	AbstractTileLoader<uint32_t> *segFL = nullptr, *intFL = nullptr; 
	std::shared_ptr<std::vector<uint32_t>> ptrI = nullptr; 
	std::shared_ptr<std::vector<uint32_t>> ptrL = nullptr; 

	// Tile height, width, depth, and number of time frames
	size_t th,
		tw,
		td,
		tt;

	// 2D tile size
	size_t tileSize;	

	// Full height, width, depth, and number of time frames
	size_t fh,
		fw,
		fd,
		ft;

	// Number of tiles along width, height, and depth
	size_t ntw,
		nth,
		ntd;

	int lvl = 0,	// Pyramid level
		lyr = 0;	//	Layer

	size_t cur_channel = 0,		// Currently selected channel (C) plane
		cur_timeframe = 0;		// Currently selected timeframe (T) plane

	// Whole-volume (X*Y*Z) assembly buffers filled by load_volume()
	std::vector<uint32_t> vol_int_, vol_seg_;
};

