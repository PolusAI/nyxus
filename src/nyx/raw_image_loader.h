#pragma once

#include <array>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "raw_tiff.h"
#include "mask_plane.h"
#include "volume_walk.h"

class RawImageLoader
{
public:

	RawImageLoader();
	// close() is idempotent, so the loaders are released on every exit path, including a
	// prescan that refuses the file or throws. The loaders are owned, so copying is disabled.
	~RawImageLoader() { close(); }
	RawImageLoader (const RawImageLoader&) = delete;
	RawImageLoader& operator= (const RawImageLoader&) = delete;
	bool open(const std::string& int_fpath, const std::string& seg_fpath);
	void close();
	bool load_tile(size_t tile_idx);
	bool load_tile(size_t tile_row, size_t tile_col);
	void free_tile_buffers();

	uint32_t get_cur_tile_seg_pixel(size_t pixel_idx);
	double get_cur_tile_dpequiv_pixel(size_t idx);

	// Visit every voxel of one (channel, timeframe) volume without materializing it. The 3D
	// prescan only needs running extrema plus mask-driven ROI geometry, so streaming tile by
	// tile costs one tile of RAM where a W*H*D staging buffer of doubles would cost 4x the raw
	// uint16 volume.
	//
	// The volume is walked by Nyxus::walk_volume, the walk ImageLoader::stream_volume_planes reads
	// featurization's volume with, so both see the same voxels. The mask is read at the plane
	// Nyxus::mask_plane_for() pairs with (channel, timeframe), and each loader's timeframe slab
	// is taken at its own frame base (a 1-frame mask can serve a whole-4D intensity).
	// fn is invoked for EVERY voxel, as fn(x, y, z, intensity, msk) -- msk may be 0 for an
	// off-ROI voxel in segmented mode (the caller decides what, if anything, to do with it);
	// in whole-slide mode every voxel is in-mask with msk == 1.
	template <typename F>
	bool for_each_voxel (size_t channel, size_t timeframe, F&& fn)
	{
		const bool haveSeg = (segFL != nullptr);

		size_t maskChannel = 0, maskTimeframe = 0;
		if (haveSeg)
			Nyxus::mask_plane_for (channel, timeframe, intFL->fullTimestamps (lvl),
				segFL->numberChannels(), segFL->fullTimestamps (lvl), maskChannel, maskTimeframe);

		const Nyxus::VolumeGrid grid = Nyxus::volume_grid_of (*intFL, lvl);
		const size_t intBase = grid.frame_base (timeframe),
			segBase = haveSeg ? Nyxus::volume_grid_of (*segFL, lvl).frame_base (maskTimeframe) : 0;

		Nyxus::walk_volume (grid,
			[&](size_t tr, size_t tc, size_t lz)
			{
				intFL->loadTileFromFile (tr, tc, lz, channel, timeframe, lvl);
				if (haveSeg)
					segFL->loadTileFromFile (tr, tc, lz, maskChannel, maskTimeframe, lvl);
			},
			[&](size_t src, size_t x0, size_t y, size_t z, size_t n)
			{
				for (size_t k = 0; k < n; k++)
				{
					const uint32_t msk = haveSeg ? segFL->get_uint32_pixel (segBase + src + k) : (uint32_t)1;
					fn (x0 + k, y, z, intFL->get_dpequiv_pixel (intBase + src + k), msk);
				}
			},
			[&]()
			{
				// the raw TIFF loaders malloc their tile buffer on each read (no-op elsewhere)
				intFL->free_tile();
				if (haveSeg)
					segFL->free_tile();
			});

		return true;
	}

	size_t get_tile_size();
	// Tiles down a column (the number of tile ROWS) and tiles across a row (the number
	// of tile COLUMNS). A grid walk bounds its row by the first and its column by the
	// second, which is the order load_tile (row, col) takes.
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
	size_t get_inten_channels();		// number of intensity channels (>=1)
	double get_physical_size_x();		// physical voxel spacing (1.0 if uncalibrated)
	double get_physical_size_y();
	double get_physical_size_z();
	std::string get_physical_size_unit();

	std::string get_slide_descr();
	bool get_fp_phys_pixvoxels();
	bool get_integer_rescale (double & slope, double & intercept);

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

