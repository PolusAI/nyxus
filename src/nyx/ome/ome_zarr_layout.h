#pragma once

// Shared OME-Zarr array layout resolution for the two Zarr loaders (omezarr.h's
// NyxusOmeZarrLoader and raw_omezarr.h's RawOmezarrLoader). Both open the same containers and
// need the same answers -- which storage dimension carries which axis role, the X/Y/Z extents
// and chunking, the C/T extents, the physical voxel spacing and the pixel type -- so the
// resolution lives here once instead of being duplicated in both headers.
//
// Guarded by OMEZARR_SUPPORT because it depends on z5 (and, through ome_zarr_meta.h, on
// nlohmann::json vendored with z5).

#ifdef OMEZARR_SUPPORT

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
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

	// The group attributes subtree holding 'multiscales': NGFF 0.5 (Zarr v3) nests the model
	// under an "ome" key, NGFF 0.4 (Zarr v2) puts it at the attributes root.
	inline const nlohmann::json& zarr_multiscales_root (const nlohmann::json& file_attributes)
	{
		return (file_attributes.contains("ome") && file_attributes["ome"].contains("multiscales"))
			? file_attributes["ome"] : file_attributes;
	}

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
		short data_format = 2;            ///< Loader-internal code for the read-template dispatch
		bool fp_pixels = false;           ///< True for float/double arrays
	};

	// Loader-internal read-template dispatch code. Kept as a small switch over PixelType so the
	// dtype string is parsed once (pixel_type_from_zarr_dtype) rather than re-matched by a chain
	// of string compares in each loader.
	inline short zarr_read_format_of (PixelType t)
	{
		switch (t)
		{
			case PixelType::UInt8:   return 1;
			case PixelType::UInt16:  return 2;
			case PixelType::UInt32:  return 3;
			case PixelType::UInt64:  return 4;
			case PixelType::Int8:    return 5;
			case PixelType::Int16:   return 6;
			case PixelType::Int32:   return 7;
			case PixelType::Int64:   return 8;
			case PixelType::Float32: return 9;
			case PixelType::Float64: return 10;
			default:                 return 2;   // unrecognized -> read as uint16
		}
	}

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
		L.data_format = zarr_read_format_of (L.dtype);
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
			// Advertise the real C/T extents so the pipeline iterates channels and timeframes;
			// without this the base class default of 1 keeps it on plane (c=0,t=0).
			L.n_channels = axes.sizeC;
			L.n_timeframes = axes.sizeT;
			// Keep the parsed physical voxel spacing for opt-in calibration.
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
			// Derive C/T extents from the positional axes so the fallback path also reports
			// multi-channel / time-series counts (1 when the axis is absent).
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
}

#endif // OMEZARR_SUPPORT
