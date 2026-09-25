#pragma once

// Shared OME-Zarr array layout for the two Zarr loaders (omezarr.h's NyxusOmeZarrLoader and
// raw_omezarr.h's RawOmezarrLoader). Both open the same containers and need the same answers --
// which storage dimension carries which axis role, the X/Y/Z extents and chunking, the C/T
// extents, the physical voxel spacing and the pixel type -- and both read a chunk the same
// way, so the resolution, the sample-type dispatch and the read window live here once
// instead of in both headers.
//
// Guarded by OMEZARR_SUPPORT because it depends on z5 (and, through ome_zarr_meta.h, on
// nlohmann::json vendored with z5).

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/types/types.hxx"

#include "ome_axes.h"
#include "ome_zarr_meta.h"

namespace Nyxus
{
	// z5 Datatype -> the "<u2"-style numpy dtype string. z5 uses '|u1'/'|i1' for the
	// single-byte types; normalize to '<u1'/'<i1' so pixel_type_from_zarr_dtype() resolves
	// both Zarr v2 (.zarray dtype) and Zarr v3 (zarr.json data_type) uniformly.
	inline std::string zarr_dtype_string_of (z5::types::Datatype dt)
	{
		switch (dt)
		{
			case z5::types::uint8:   return "<u1";
			case z5::types::uint16:  return "<u2";
			case z5::types::uint32:  return "<u4";
			case z5::types::uint64:  return "<u8";
			case z5::types::int8:    return "<i1";
			case z5::types::int16:   return "<i2";
			case z5::types::int32:   return "<i4";
			case z5::types::int64:   return "<i8";
			case z5::types::float32: return "<f4";
			case z5::types::float64: return "<f8";
			default:                 return "<u2";
		}
	}

	// Call f with a value of the C++ type that stores a sample of pixel type t, so a loader's
	// typed read is written once as a generic lambda. An unrecognized type reads as uint16.
	template <class F>
	inline void with_zarr_sample_type (PixelType t, F&& f)
	{
		switch (t)
		{
			case PixelType::UInt8:   f (std::uint8_t{});  break;
			case PixelType::UInt16:  f (std::uint16_t{}); break;
			case PixelType::UInt32:  f (std::uint32_t{}); break;
			case PixelType::UInt64:  f (std::uint64_t{}); break;
			case PixelType::Int8:    f (std::int8_t{});   break;
			case PixelType::Int16:   f (std::int16_t{});  break;
			case PixelType::Int32:   f (std::int32_t{});  break;
			case PixelType::Int64:   f (std::int64_t{});  break;
			case PixelType::Float32: f (float{});         break;
			case PixelType::Float64: f (double{});        break;
			default:                 f (std::uint16_t{}); break;
		}
	}

	// One chunk-aligned block to read: its valid extent (the last chunk along an axis may be
	// partial) and the z5 read window in storage-dimension order.
	struct ZarrBlock
	{
		std::size_t depth = 1, height = 0, width = 0;
		z5::types::ShapeType shape, offset;
	};

	// Everything a Zarr loader needs to address the array, resolved from its metadata.
	struct ZarrLayout
	{
		// Storage-dimension index of each axis role (-1 if the axis is absent)
		int ix = -1, iy = -1, iz = -1, ic = -1, it = -1;
		std::size_t ndim = 0;             ///< Number of on-disk dimensions (2..5)

		std::size_t full_width = 0, full_height = 0, full_depth = 1;
		std::size_t tile_width = 0, tile_height = 0, tile_depth = 1;

		std::size_t n_levels = 1;         ///< Pyramid level count declared in multiscales
		std::size_t n_channels = 1;       ///< Channel (C) extent
		std::size_t n_timeframes = 1;     ///< Time (T) extent

		double phys_x = 1.0, phys_y = 1.0, phys_z = 1.0;   ///< Physical voxel spacing
		std::string phys_unit;            ///< Physical-size unit (e.g. "micrometer")

		PixelType dtype = PixelType::UInt16;
		short bits_per_sample = 0;        ///< Real bit depth (0 for an unrecognized dtype)
		bool fp_pixels = false;           ///< True for float/double arrays

		/// @brief The block of chunk (tile_row, tile_col, tile_layer) in plane (c,t). A chunk
		/// may span several Z-planes (tile_depth > 1), and all of them are read at once.
		/// Throws when the chunk or the plane lies outside the array.
		ZarrBlock block_at (std::size_t tile_row, std::size_t tile_col, std::size_t tile_layer,
			std::size_t c, std::size_t t) const
		{
			const std::size_t y0 = tile_row * tile_height,
				x0 = tile_col * tile_width,
				z0 = tile_layer * tile_depth;

			if (y0 >= full_height || x0 >= full_width || z0 >= full_depth || c >= n_channels || t >= n_timeframes)
				throw std::runtime_error ("OME-Zarr: chunk (row,col,layer)=(" + std::to_string (tile_row) + "," + std::to_string (tile_col)
					+ "," + std::to_string (tile_layer) + ") of plane (c,t)=(" + std::to_string (c) + "," + std::to_string (t)
					+ ") is outside the array");

			ZarrBlock b;
			b.height = (std::min) (tile_height, full_height - y0);
			b.width = (std::min) (tile_width, full_width - x0);
			b.depth = (iz >= 0) ? (std::min) (tile_depth, full_depth - z0) : 1;

			b.shape.assign (ndim, 1);
			b.offset.assign (ndim, 0);
			b.shape[iy] = b.height;   b.offset[iy] = y0;
			b.shape[ix] = b.width;    b.offset[ix] = x0;
			if (iz >= 0) { b.shape[iz] = b.depth; b.offset[iz] = z0; }
			if (ic >= 0) b.offset[ic] = c;
			if (it >= 0) b.offset[it] = t;
			return b;
		}
	};

	/// @brief Resolve the array layout from NGFF metadata + the level-0 array's shape/chunking.
	/// Prefers the 'axes' block (so the on-disk order is honored rather than assumed to be
	/// TCZYX); falls back to a rank-safe positional mapping when 'axes' is absent or unusable.
	/// Throws if the metadata is self-inconsistent (axis count vs array rank) or if X/Y cannot
	/// be resolved -- either would make the read index out of bounds.
	inline ZarrLayout resolve_zarr_layout (
		const nlohmann::json& file_attributes,
		const std::vector<std::size_t>& level0Shape,
		const std::vector<std::size_t>& chunkShape,
		z5::types::Datatype dt)
	{
		ZarrLayout L;

		const std::string dtype_str = zarr_dtype_string_of (dt);
		L.dtype = pixel_type_from_zarr_dtype (dtype_str);
		L.bits_per_sample = (short) bits_of (L.dtype);
		L.fp_pixels = is_float (L.dtype);

		OmeAxes axes = parse_ome_zarr (file_attributes, level0Shape, dtype_str);
		if (axes.valid)
		{
			// Reject self-inconsistent metadata rather than guess: if the 'axes' count
			// disagrees with the array rank, indexing the shape by axis role would read
			// out of bounds.
			if (axes.storageAxes.size() != level0Shape.size())
				throw std::runtime_error("OME-Zarr: 'axes' count " + std::to_string(axes.storageAxes.size())
					+ " does not match array rank " + std::to_string(level0Shape.size()));
			L.ndim = axes.storageAxes.size();
			L.ix = axes.storageIndexOf('X'); L.iy = axes.storageIndexOf('Y');
			L.iz = axes.storageIndexOf('Z'); L.ic = axes.storageIndexOf('C');
			L.it = axes.storageIndexOf('T');
			L.n_levels = axes.numberPyramidLevels();
			L.n_channels = axes.sizeC;
			L.n_timeframes = axes.sizeT;
			// the parsed physical voxel spacing, for opt-in calibration
			L.phys_x = axes.physX; L.phys_y = axes.physY; L.phys_z = axes.physZ;
			L.phys_unit = axes.unitXY;
		}
		else
		{
			// No usable 'axes': map by position (X,Y last; Z,C,T before) -- rank-safe.
			L.ndim = level0Shape.size();
			int n = (int) L.ndim;
			L.ix = n - 1; L.iy = n - 2;
			L.iz = (n >= 3) ? n - 3 : -1;
			L.ic = (n >= 4) ? n - 4 : -1;
			L.it = (n >= 5) ? n - 5 : -1;
			L.n_levels = 1;
			L.n_channels = (L.ic >= 0) ? level0Shape[L.ic] : 1;
			L.n_timeframes = (L.it >= 0) ? level0Shape[L.it] : 1;
		}

		// X and Y must resolve to real dimensions, else the read would index OOB.
		if (L.ix < 0 || L.iy < 0 || (std::size_t) L.ix >= level0Shape.size() || (std::size_t) L.iy >= level0Shape.size())
			throw std::runtime_error("OME-Zarr: cannot resolve X/Y axes from metadata");

		L.full_width  = level0Shape[L.ix];
		L.full_height = level0Shape[L.iy];
		L.full_depth  = (L.iz >= 0) ? level0Shape[L.iz] : 1;
		L.tile_width  = chunkShape[L.ix];
		L.tile_height = chunkShape[L.iy];
		L.tile_depth  = (L.iz >= 0) ? chunkShape[L.iz] : 1;

		return L;
	}

	/// @brief The group attributes carrying the NGFF model: 0.5 nests them under "ome", 0.4
	/// puts them at the group root. Keys outside the model (bioformats2raw.layout) appear in
	/// whichever of the two a given writer uses, so both have to be consulted.
	inline const nlohmann::json& ome_zarr_model_attributes (const nlohmann::json& groupAttrs)
	{
		auto it = groupAttrs.find ("ome");
		if (it != groupAttrs.end() && it->is_object())
			return *it;
		return groupAttrs;
	}

	/// @brief Open the level-0 (full-resolution) array of the OME-Zarr group 'file' and resolve
	/// its layout into 'layout'. z5 detects Zarr v2 (.zarray) or v3 (zarr.json) and reports
	/// shape, chunking and dtype through the same Dataset interface for both. Throws when the
	/// group declares no multiscales dataset to open, and when the array it names cannot be
	/// opened -- in both cases naming the path, since a failure here is nearly always a store
	/// whose root is one level above the image rather than a damaged file.
	inline std::unique_ptr<z5::Dataset> open_zarr_level0 (const z5::filesystem::handle::File& file, ZarrLayout& layout)
	{
		nlohmann::json file_attributes;
		z5::readAttributes (file, file_attributes);

		const nlohmann::json* multiscale = ome_zarr_multiscale (file_attributes);
		if (multiscale == nullptr)
		{
			// A bioformats2raw store's root is not an image group: it carries only
			// {"bioformats2raw.layout": <n>} and each image sits in a child group named by its
			// series index. That is the shape bioformats2raw writes by default, so it is the
			// most likely first thing anyone hands Nyxus; "declares no multiscales" alone
			// reads as a damaged file when the path is simply one level too high.
			if (ome_zarr_model_attributes (file_attributes).contains ("bioformats2raw.layout"))
				throw std::runtime_error ("OME-Zarr: '" + file.path().string() + "' is a bioformats2raw "
					"store rather than an image group -- its images are the numbered child groups. "
					"Use the series group, e.g. a copy of '" + (file.path() / "0").string() +
					"' under a name ending in .zarr");
			throw std::runtime_error ("OME-Zarr: '" + file.path().string() + "' declares no multiscales");
		}

		const std::string levelPath = ome_zarr_level_path (*multiscale, 0);
		std::unique_ptr<z5::Dataset> ds;
		try
		{
			ds = z5::openDataset (file, levelPath);
		}
		catch (const std::exception& e)
		{
			// z5 maps only the little-endian ('<') and single-byte ('|') dtype spellings, so a
			// big-endian array fails here on its dtype code alone. Worth naming, because
			// bioformats2raw through 0.9.x writes big-endian and has no switch to change it:
			// every 16-bit store that converter produced lands on this path.
			const std::string what = e.what();
			const std::string hint = what.find ("dtype: >") != std::string::npos
				? " -- big-endian arrays are unsupported; re-convert with bioformats2raw 0.12 or later,"
				  " which writes little-endian"
				: "";
			throw std::runtime_error ("OME-Zarr: cannot open level-0 array '" + levelPath + "' of '"
				+ file.path().string() + "': " + what + hint);
		}

		std::vector<std::size_t> level0Shape (ds->shape().begin(), ds->shape().end()),
			chunkShape (ds->defaultChunkShape().begin(), ds->defaultChunkShape().end());
		layout = resolve_zarr_layout (file_attributes, level0Shape, chunkShape, ds->getDtype());
		return ds;
	}
}

#endif // OMEZARR_SUPPORT
