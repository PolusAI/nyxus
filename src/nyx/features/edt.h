#pragma once

#include <cstdint>
#include <vector>
#include "pixel.h"

namespace Nyxus
{
	/// @brief Exact squared Euclidean distance transform of a 2D raster seeded at a set of sites.
	///
	/// For every cell of the W x H raster whose origin is (xmin, ymin) in absolute image coordinates,
	/// computes the squared Euclidean distance to the nearest site. Sites outside the raster are
	/// ignored, so the caller sizes the raster to contain every site it wants considered.
	///
	/// The result is exact, not an approximation: a cell's value is the same integer
	/// min over sites of (dx*dx + dy*dy) that an exhaustive scan over the sites returns. The cost is
	/// O(W*H) and is independent of the number of sites, against O(W*H*sites) for the scan.
	///
	/// Two passes after Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions": a
	/// per-column nearest-site distance, then a per-row lower envelope of the parabolas it defines.
	/// Both are integer arithmetic, the envelope's breakpoints included, which are compared by
	/// cross-multiplying rationals rather than by dividing. The widest intermediate is bounded by
	/// 2*max(W,H)^2 * 2*max(W,H), so int64_t covers any raster up to 2^20 cells per side.
	///
	/// The transform runs in the output buffer, so it holds one int64_t per raster cell and a further
	/// O(W). The caller supplies the raster, which is what makes this unsuitable for a ROI too large
	/// to raster - there, the exhaustive scan needs no buffer at all.
	///
	/// @param sites Seed pixels, in absolute image coordinates.
	/// @param xmin Absolute x of raster column 0.
	/// @param ymin Absolute y of raster row 0.
	/// @param W Raster width in cells.
	/// @param H Raster height in cells.
	/// @param sqdist Receives W*H squared distances in row-major order, cell (x,y) at index y*W + x.
	///	   Every cell is finite when at least one site lies inside the raster; with none, every cell
	///	   is -1, no site being reachable.
	void exact_sqedt (
		const std::vector<Pixel2>& sites,
		StatsInt xmin,
		StatsInt ymin,
		size_t W,
		size_t H,
		std::vector<int64_t>& sqdist);
}
