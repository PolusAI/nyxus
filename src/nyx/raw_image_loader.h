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
	bool open(const std::string& int_fpath, const std::string& seg_fpath, bool preserve_hu = false);
	void close();
	bool load_tile(size_t tile_idx);
	bool load_tile(size_t tile_row, size_t tile_col);
	void free_tile_buffers();

	uint32_t get_cur_tile_seg_pixel(size_t pixel_idx);
	double get_cur_tile_dpequiv_pixel(size_t idx);

	// FIX: assemble the whole X*Y*Z volume by looping Z, into packed (stride = full width)
	// buffers. The 3D prescan used to do ONE load_tile(0,0) and then index W*H*D voxels off
	// the tile buffer -- but a per-plane loader (OME-TIFF / OME-Zarr) fills only ONE Z-plane,
	// so everything past the first slice read out of bounds and corrupted the slide min/max.
	// (It also assumed the tile stride equals the full width.) The volumetric counterpart of
	// ImageLoader::load_volume; NIfTI-style whole-4D loaders are slabbed out via frameBase.
	bool load_volume (size_t channel = 0, size_t timeframe = 0);
	double get_voxel_dpequiv (size_t idx) const { return vol_int_[idx]; }
	uint32_t get_voxel_seg (size_t idx) const { return vol_seg_[idx]; }

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
	size_t get_full_depth();
	size_t get_inten_time();
	size_t get_mask_time();
	size_t get_inten_channels();		// FIX (IO): number of intensity channels (>=1)
	double get_physical_size_x();		// FIX (IO): physical voxel spacing (1.0 if uncalibrated)
	double get_physical_size_y();
	double get_physical_size_z();
	std::string get_physical_size_unit();

	// Select which channel (C) / timeframe (T) plane subsequent load_tile() calls
	// read. Default 0/0 preserves the single-channel, single-timepoint behavior.
	void set_channel (size_t c) { cur_channel = c; }
	void set_timeframe (size_t t) { cur_timeframe = t; }
	size_t get_channel() const { return cur_channel; }
	size_t get_timeframe() const { return cur_timeframe; }

	std::string get_slide_descr();
	bool get_fp_phys_pixvoxels();

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

	size_t cur_channel = 0,		// Currently selected channel (C) plane
		cur_timeframe = 0;		// Currently selected timeframe (T) plane

	// Whole-volume (X*Y*Z) assembly buffers filled by load_volume(), packed at full-width
	// stride. vol_seg_ stays empty in whole-slide (no mask) mode.
	std::vector<double> vol_int_;
	std::vector<uint32_t> vol_seg_;
};

