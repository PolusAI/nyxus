#pragma once

#include <array>
#include <algorithm>
#include <functional>
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

	// Stream the whole X*Y*Z volume for one (channel, timeframe) to 'sink', Z-plane by Z-plane in
	// ascending Z, without materializing it: each tile layer (the tileDepth Z-planes a read
	// delivers) is assembled into reused plane buffers, so peak memory is one tile layer of
	// intensity and mask rather than the whole cube. Every volumetric consumer -- the phase-1 ROI
	// metrics, the phase-2 voxel caches and the out-of-core voxel clouds -- reads the volume
	// through here. The mask is read at the plane Nyxus::mask_plane_for() pairs with
	// (channel, timeframe); 'seg_plane' is empty when the pair has no mask.
	void stream_volume_planes (size_t channel, size_t timeframe,
		const std::function<void(size_t z, const std::vector<uint32_t>& int_plane, const std::vector<uint32_t>& seg_plane)>& sink);

	// True when a read of this pair delivers less than the volume a streaming pass walks, so the
	// pass can hold a slab instead of the whole thing. A pass walks one (channel, timeframe)
	// volume, and a read delivers tileDepth planes of it -- except on a loader that keeps the
	// whole time series in its tile (NIfTI), where one read is the entire x*y*z*t blob. The
	// phases that hold the volume anyway can afford such a loader; the out-of-core paths, whose
	// entire purpose is a bounded footprint, cannot.
	bool streams_bounded() const;

	// The loader of this pair that has no bounded streaming path, if either has: sets 'planes' to
	// the planes one read of it delivers and 'of_mask' to which input that is, so a refusal
	// reports the file the user has to re-chunk. Returns false when the pair streams -- it is the
	// same question streams_bounded() answers, off the same numbers.
	bool unstreamable_read (size_t& planes, bool& of_mask) const;

	const std::vector<uint32_t>& get_int_tile_buffer();
	const std::vector<uint32_t>& get_seg_tile_buffer();
	const std::shared_ptr<std::vector<uint32_t>>& get_seg_tile_sptr();
	size_t get_tile_size();
	// Tiles down a column (the number of tile ROWS) and tiles across a row (the number
	// of tile COLUMNS). A grid walk bounds its row by the first and its column by the
	// second, which is the order load_tile (row, col) takes.
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

	// Channel (C) / timeframe (T) plane that load_tile() reads. open() sets it to 0/0 (the
	// single-channel, single-timepoint case); stream_volume_planes() sets it to the plane it reads.
	size_t cur_channel = 0,
		cur_timeframe = 0;

	// Assemble tile layer 'lz' of one loader's (channel, timeframe) volume into 'planes', one
	// X*Y buffer per Z-plane of the layer, honoring that loader's own tile grid, tileDepth and
	// tileTimestamps (per-plane vs whole-4D). Returns the number of planes the layer holds (the
	// last layer may be partial).
	size_t assemble_tile_layer (AbstractTileLoader<uint32_t>* fl,
		std::shared_ptr<std::vector<uint32_t>>& ptr,
		std::vector<std::vector<uint32_t>>& planes, size_t lz, size_t channel, size_t timeframe);
};

