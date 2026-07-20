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

	// FIX: visit every voxel of one (channel, timeframe) volume WITHOUT materializing it.
	// The 3D prescan used to do ONE load_tile(0,0) and then index W*H*D voxels off the tile
	// buffer -- but a per-plane loader (OME-TIFF / OME-Zarr) fills only ONE Z-plane, so
	// everything past the first slice read out of bounds and corrupted the slide min/max.
	// (It also assumed the tile stride equals the full width.) Streaming rather than filling
	// a W*H*D staging buffer matters because the prescan repeats this for every (c,t) and
	// only needs running min/max plus mask-driven ROI geometry: a double buffer would cost
	// 4x the raw uint16 volume in RAM per pass and buy nothing.
	//
	// Walks the tile GRID of each plane, so planes spanning several tiles/chunks are handled;
	// NIfTI-style whole-4D loaders (tileDepth == full depth) are slabbed out via frameBase.
	// fn is invoked ONLY for in-mask voxels (msk != 0), as fn(x, y, z, intensity, msk); in
	// whole-slide mode every voxel is in-mask with msk == 1.
	template <typename F>
	bool for_each_voxel (size_t channel, size_t timeframe, F&& fn)
	{
		const bool haveSeg = (segFL != nullptr);

		// Use each loader's OWN layout: per-plane loaders (OME-Zarr, multi-page/OME-TIFF)
		// deliver one Z-plane per read (tileDepth == 1); a whole-4D loader delivers the entire
		// x*y*z*t blob in one read and ignores the layer arg. Mirrors ImageLoader::assemble_volume.
		const size_t ltd = intFL->tileDepth (lvl),
			ltt = intFL->tileTimestamps (lvl),
			lntd = intFL->numberTileDepth (lvl),
			lnth = intFL->numberTileHeight (lvl),
			lntw = intFL->numberTileWidth (lvl),
			frameStride = ltd * th * tw,
			frameBase = (ltt > 1) ? timeframe * frameStride : 0;

		// The mask is channel-agnostic unless it genuinely has that many channels
		const size_t maskChannel = (haveSeg && channel < segFL->numberChannels()) ? channel : 0;

		for (size_t lz = 0; lz < lntd; lz++)
		{
			for (size_t tr = 0; tr < lnth; tr++)
			for (size_t tc = 0; tc < lntw; tc++)
			{
				intFL->loadTileFromFile (tr, tc, lz, channel, timeframe, lvl);
				if (haveSeg)
					segFL->loadTileFromFile (tr, tc, lz, maskChannel, timeframe, lvl);

				const size_t row0 = tr * th, col0 = tc * tw;
				if (row0 < fh && col0 < fw)
				{
					const size_t validH = (std::min) (th, fh - row0),
						validW = (std::min) (tw, fw - col0);

					for (size_t pz = 0; pz < ltd && (lz * ltd + pz) < fd; pz++)
					{
						const size_t gz = lz * ltd + pz;
						for (size_t row = 0; row < validH; row++)
							for (size_t col = 0; col < validW; col++)
							{
								const size_t src = frameBase + (pz * th + row) * tw + col;
								const uint32_t msk = haveSeg ? segFL->get_uint32_pixel (src) : (uint32_t)1;
								if (!msk)
									continue;		// off-ROI: skip before paying for the intensity read
								fn (col0 + col, row0 + row, gz, intFL->get_dpequiv_pixel (src), msk);
							}
					}
				}

				// the raw TIFF loaders malloc their tile buffer on each read (no-op elsewhere)
				intFL->free_tile();
				if (haveSeg)
					segFL->free_tile();
			}
		}

		return true;
	}

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

};

