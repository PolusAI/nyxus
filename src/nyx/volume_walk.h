#pragma once

#include <algorithm>
#include <cstddef>

namespace Nyxus
{
	/// @brief The tile grid of one loader's volume at one pyramid level.
	struct VolumeGrid
	{
		std::size_t full_w = 0, full_h = 0, full_d = 1;
		std::size_t tile_w = 0, tile_h = 0, tile_d = 1, tile_t = 1;
		std::size_t n_tiles_w = 0, n_tiles_h = 0, n_tiles_d = 0;

		/// @brief Offset of timeframe t's slab in a tile buffer. A whole-4D loader (NIfTI)
		/// delivers every timeframe in one read (tile_t > 1) and the frame is slabbed out of it;
		/// a per-plane loader reads the requested frame alone, so its slab starts at 0.
		std::size_t frame_base (std::size_t t) const
		{
			return (tile_t > 1) ? t * tile_d * tile_h * tile_w : 0;
		}
	};

	/// @brief The grid of a loader of either stack (AbstractTileLoader or RawFormatLoader).
	template <class Loader>
	VolumeGrid volume_grid_of (const Loader& fl, std::size_t lvl)
	{
		VolumeGrid g;
		g.full_w = fl.fullWidth (lvl);
		g.full_h = fl.fullHeight (lvl);
		g.full_d = fl.fullDepth (lvl);
		g.tile_w = fl.tileWidth (lvl);
		g.tile_h = fl.tileHeight (lvl);
		g.tile_d = fl.tileDepth (lvl);
		g.tile_t = fl.tileTimestamps (lvl);
		g.n_tiles_w = fl.numberTileWidth (lvl);
		g.n_tiles_h = fl.numberTileHeight (lvl);
		g.n_tiles_d = fl.numberTileDepth (lvl);
		return g;
	}

	/// @brief Walk one tile layer 'lz' of a volume tile by tile: every tile of its tile grid, and
	/// every plane of a tile deeper than one plane.
	///
	/// read(tile_row, tile_col, tile_layer) loads one tile. row(src, x0, y, z, n) then receives
	/// each row of that tile's valid extent: the n samples at tile-buffer offset src (within one
	/// frame, see VolumeGrid::frame_base) are voxels x0..x0+n-1 of row y of plane z. Edge tiles
	/// are partial, so only their valid extent is handed on. done() runs after each tile.
	template <class Read, class Row, class Done>
	void walk_tile_layer (const VolumeGrid& g, std::size_t lz, Read&& read, Row&& row, Done&& done)
	{
		for (std::size_t tr = 0; tr < g.n_tiles_h; tr++)
			for (std::size_t tc = 0; tc < g.n_tiles_w; tc++)
			{
				read (tr, tc, lz);

				const std::size_t row0 = tr * g.tile_h,
					col0 = tc * g.tile_w;
				if (row0 < g.full_h && col0 < g.full_w)
				{
					const std::size_t valid_h = (std::min) (g.tile_h, g.full_h - row0),
						valid_w = (std::min) (g.tile_w, g.full_w - col0);

					for (std::size_t pz = 0; pz < g.tile_d && lz * g.tile_d + pz < g.full_d; pz++)
						for (std::size_t r = 0; r < valid_h; r++)
							row ((pz * g.tile_h + r) * g.tile_w, col0, row0 + r, lz * g.tile_d + pz, valid_w);
				}

				done();
			}
	}

	/// @brief Walk a whole X*Y*Z volume, tile layer by tile layer (see walk_tile_layer).
	/// Featurization (ImageLoader::stream_volume_planes, tile layer by tile layer) and the prescan
	/// (RawImageLoader::for_each_voxel) all walk through here, so they see the same voxels.
	template <class Read, class Row, class Done>
	void walk_volume (const VolumeGrid& g, Read&& read, Row&& row, Done&& done)
	{
		for (std::size_t lz = 0; lz < g.n_tiles_d; lz++)
			walk_tile_layer (g, lz, read, row, done);
	}
}
