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

	// Assemble the whole X*Y*Z volume for one (channel, timeframe) into an
	// internal buffer by looping over Z-planes. This is what lets plane-by-plane
	// loaders (OME-Zarr, multi-page/OME-TIFF) feed the volumetric pipeline, which
	// otherwise assumes the whole volume arrives in one read (the NIfTI model).
	// The mask may live on a different timeframe than the intensity (the 1-mask :
	// N-intensity case), so `mask_timeframe` is separable; the 2-arg overload uses
	// the same frame for both.
	bool load_volume (size_t channel, size_t timeframe, size_t mask_timeframe);
	bool load_volume (size_t channel, size_t timeframe) { return load_volume(channel, timeframe, timeframe); }
	const std::vector<uint32_t>& get_int_volume_buffer() const { return vol_int_; }
	const std::vector<uint32_t>& get_seg_volume_buffer() const { return vol_seg_; }

	// Stream the whole X*Y*Z volume for one (channel, timeframe) WITHOUT materializing it:
	// one Z-plane is assembled into a reused buffer and handed to 'sink', so peak memory is
	// two planes (intensity + mask) rather than the whole cube. This is what lets an oversized
	// volumetric ROI be featurized out-of-core. Applies the same mask channel/timeframe clamp as
	// load_volume (the mask is usually channel-agnostic). Returns false for a whole-4D loader
	// (NIfTI, tileDepth>1) which delivers the entire cube in one read and cannot be streamed here.
	bool stream_volume_planes (size_t channel, size_t timeframe, size_t mask_timeframe,
		const std::function<void(size_t z, const std::vector<uint32_t>& int_plane, const std::vector<uint32_t>& seg_plane)>& sink);

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

	// Channel (C) / timeframe (T) plane that load_tile() reads. Set by load_volume() /
	// stream_volume_planes(), which take the plane as an argument; 0/0 (the single-channel,
	// single-timepoint case) until then.
	size_t cur_channel = 0,
		cur_timeframe = 0;

	// Whole-volume (X*Y*Z) assembly buffers filled by load_volume()
	std::vector<uint32_t> vol_int_, vol_seg_;

	// Assemble one loader's X*Y*Z volume (for the given channel/timeframe) into dst,
	// honoring that loader's own tileDepth/tileTimestamps (per-plane vs whole-4D).
	void assemble_volume (AbstractTileLoader<uint32_t>* fl,
		std::shared_ptr<std::vector<uint32_t>>& ptr,
		std::vector<uint32_t>& dst, size_t channel, size_t timeframe);

	// Assemble a single global Z-plane 'gz' (X*Y) into 'dst_plane' for a per-plane loader
	// (tileDepth==1). The streaming counterpart of assemble_volume's inner plane fill.
	void assemble_one_plane (AbstractTileLoader<uint32_t>* fl,
		std::shared_ptr<std::vector<uint32_t>>& ptr,
		std::vector<uint32_t>& dst_plane, size_t gz, size_t channel, size_t timeframe);
};

