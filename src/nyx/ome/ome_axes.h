#pragma once

// -----------------------------------------------------------------------------
// OmeAxes — one format-agnostic dimension descriptor for native 3D/4D/5D OME
// input. Both container families populate this from real metadata:
//   - OME-TIFF : parse_ome_xml()   (ome_tiff_meta.h)  — OME-XML <Pixels> block
//   - OME-Zarr : parse_ome_zarr()  (ome_zarr_meta.h)  — multiscales/axes/datasets
// Everything downstream (loaders, SlideProps, output schema) consumes OmeAxes
// instead of assuming C=T=1 and a hard-coded TCZYX axis order.
//
// This header is STL-only so it compiles in every build configuration.
// -----------------------------------------------------------------------------

#include <string>
#include <vector>
#include <cstddef>
#include <cctype>

namespace Nyxus
{
	// Role of an axis, mirroring the OME-NGFF 'type' field (space|channel|time).
	enum class AxisKind { Space, Channel, Time };

	// Resolved pixel/element type (consolidates the ad-hoc dtype codes that
	// omezarr.h / raw_omezarr.h and the TIFF sampleFormat switch use today).
	enum class PixelType { UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float32, Float64, Unknown };

	inline unsigned bits_of(PixelType t)
	{
		switch (t)
		{
			case PixelType::UInt8:  case PixelType::Int8:    return 8;
			case PixelType::UInt16: case PixelType::Int16:   return 16;
			case PixelType::UInt32: case PixelType::Int32: case PixelType::Float32: return 32;
			case PixelType::UInt64: case PixelType::Int64: case PixelType::Float64: return 64;
			default: return 0;
		}
	}
	inline bool is_float(PixelType t) { return t == PixelType::Float32 || t == PixelType::Float64; }

	// OME 'Type' attribute (OME-XML) -> PixelType.
	inline PixelType pixel_type_from_ome_type(const std::string& s)
	{
		if (s == "uint8")   return PixelType::UInt8;
		if (s == "uint16")  return PixelType::UInt16;
		if (s == "uint32")  return PixelType::UInt32;
		if (s == "uint64")  return PixelType::UInt64;
		if (s == "int8")    return PixelType::Int8;
		if (s == "int16")   return PixelType::Int16;
		if (s == "int32")   return PixelType::Int32;
		if (s == "int64")   return PixelType::Int64;
		if (s == "float")   return PixelType::Float32;
		if (s == "double")  return PixelType::Float64;
		return PixelType::Unknown;
	}

	// Zarr dtype -> PixelType. Accepts v2 numpy strings ("<u2", ">i4", "|u1", ...)
	// and v3 names ("uint16", "int32", "float32", ...).
	inline PixelType pixel_type_from_zarr_dtype(const std::string& in)
	{
		std::string s = in;
		// strip a leading byte-order char for the v2 form
		if (!s.empty() && (s[0] == '<' || s[0] == '>' || s[0] == '|' || s[0] == '='))
			s = s.substr(1);
		if (s == "u1" || s == "uint8")   return PixelType::UInt8;
		if (s == "u2" || s == "uint16")  return PixelType::UInt16;
		if (s == "u4" || s == "uint32")  return PixelType::UInt32;
		if (s == "u8" || s == "uint64")  return PixelType::UInt64;
		if (s == "i1" || s == "int8")    return PixelType::Int8;
		if (s == "i2" || s == "int16")   return PixelType::Int16;
		if (s == "i4" || s == "int32")   return PixelType::Int32;
		if (s == "i8" || s == "int64")   return PixelType::Int64;
		if (s == "f2")                   return PixelType::Float32;  // half -> promote to float32
		if (s == "f4" || s == "float32" || s == "float") return PixelType::Float32;
		if (s == "f8" || s == "float64" || s == "double") return PixelType::Float64;
		return PixelType::Unknown;
	}

	// One axis in on-disk (storage) order.
	struct OmeAxis
	{
		AxisKind kind = AxisKind::Space;
		char     label = 'X';     // one of 'X','Y','Z','C','T'
		std::size_t size = 1;     // extent along this axis
		double   physical = 1.0;  // voxel spacing / frame interval; 1.0 if unknown
		std::string unit;         // e.g. "micrometer","second"; empty if unknown
	};


	struct OmeAxes
	{
		// Sizes resolved by axis role (independent of on-disk order).
		std::size_t sizeX = 1, sizeY = 1, sizeZ = 1, sizeC = 1, sizeT = 1;

		// On-disk axis order (numpy/slowest-first), e.g. "TCZYX" or "CTZYX" or "ZYX".
		std::string storageOrder;
		// OME DimensionOrder (fastest-first, always begins "XY"), e.g. "XYZCT".
		std::string omeDimensionOrder;
		std::vector<OmeAxis> storageAxes;  // in storageOrder order

		// Physical calibration (convenience mirror of the space axes).
		double physX = 1, physY = 1, physZ = 1;
		std::string unitXY, unitZ;

		PixelType dtype = PixelType::UInt16;
		unsigned  bitsPerSample = 16;

		// Resolution levels the container declares, level 0 being full-res. Nyxus reads level 0;
		// the count is what a loader reports through numberPyramidLevels().
		std::size_t n_pyramid_levels = 1;

		// Explicit plane-ordinal -> IFD map from the OME <TiffData> elements, indexed by the
		// canonical DimensionOrder ordinal (see canonicalPlaneOrdinal). Empty when the OME-XML
		// carries no TiffData (or none that maps same-file planes), in which case the plane
		// ordinal IS the IFD (the OME default: planes stored contiguously from IFD 0 in
		// DimensionOrder). A writer that offsets the first IFD or reorders planes populates this.
		std::vector<std::size_t> planeToIfd;

		// True when a <TiffData> names planes in another file (a <UUID> other than the file's own):
		// a multi-file OME-TIFF, whose companion files this reader does not open. The TIFF loaders
		// refuse such a file (read_ome_tiff_axes).
		bool multiFileTiff = false;

		bool valid = false;   // set true by a successful parse

		// Position of a label in storageAxes; -1 if absent.
		int storageIndexOf(char lbl) const
		{
			for (std::size_t i = 0; i < storageAxes.size(); ++i)
				if (storageAxes[i].label == lbl) return static_cast<int>(i);
			return -1;
		}

		// The plane's ordinal under the OME DimensionOrder (default rasterization: the axes
		// after XY, fastest to slowest). E.g. XYZCT -> z + c*sizeZ + t*sizeZ*sizeC. This is the
		// IFD only when planes are stored contiguously from IFD 0; TiffData can say otherwise.
		std::size_t canonicalPlaneOrdinal(std::size_t z, std::size_t c, std::size_t t) const
		{
			std::size_t idx = 0, stride = 1;
			for (char ax : omeDimensionOrder)
			{
				if (ax == 'X' || ax == 'Y') continue;
				std::size_t coord = (ax == 'Z') ? z : (ax == 'C') ? c : t;
				std::size_t sz    = (ax == 'Z') ? sizeZ : (ax == 'C') ? sizeC : sizeT;
				idx += coord * stride;
				stride *= sz;
			}
			return idx;
		}

		// Map a (z,c,t) plane to the physical IFD holding its pixels. Honors an explicit
		// <TiffData> plane->IFD map when present (writers that start at a non-zero IFD or
		// reorder planes); otherwise falls back to the canonical ordinal.
		std::size_t ifdForPlane(std::size_t z, std::size_t c, std::size_t t) const
		{
			std::size_t ord = canonicalPlaneOrdinal(z, c, t);
			if (!planeToIfd.empty() && ord < planeToIfd.size())
				return planeToIfd[ord];
			return ord;
		}

		std::size_t numberPyramidLevels() const { return n_pyramid_levels < 1 ? 1 : n_pyramid_levels; }
	};

	// ---- small shared helpers ----------------------------------------------

	// Length-unit -> micrometer scale factor (OME-XML/NGFF use the full UDUNITS-2 names;
	// tolerate common short forms too, since not every writer follows the spec exactly).
	// Returns 1.0 (no-op) for an empty/unrecognized unit -- an unrecognized unit is left
	// as-is rather than silently misscaled, and the caller keeps the original unit string
	// in that case so the mismatch stays visible instead of masquerading as "micrometer".
	inline double unit_scale_to_micrometer(const std::string& unit)
	{
		if (unit == "meter" || unit == "metre" || unit == "m")   return 1e6;
		if (unit == "centimeter" || unit == "centimetre" || unit == "cm") return 1e4;
		if (unit == "millimeter" || unit == "millimetre" || unit == "mm") return 1e3;
		if (unit == "micrometer" || unit == "micrometre" || unit == "micron" || unit == "um" || unit == "\xC2\xB5m") return 1.0;
		if (unit == "nanometer" || unit == "nanometre" || unit == "nm") return 1e-3;
		if (unit == "angstrom" || unit == "\xC3\x85")             return 1e-4;
		if (unit == "picometer" || unit == "picometre" || unit == "pm") return 1e-6;
		return 0.0;   // unrecognized (includes "" == uncalibrated)
	}

	// Canonicalize a physical-size value + its declared unit to micrometer in place.
	// A recognized non-micrometer unit is converted and relabeled "micrometer"; an
	// unrecognized/empty unit is left untouched (value AND unit string both unchanged).
	inline void canonicalize_to_micrometer(double& physical, std::string& unit)
	{
		double scale = unit_scale_to_micrometer(unit);
		if (scale == 0.0)
			return;
		physical *= scale;
		unit = "micrometer";
	}

	inline AxisKind axis_kind_of(char label)
	{
		return (label == 'C') ? AxisKind::Channel : (label == 'T') ? AxisKind::Time : AxisKind::Space;
	}

	// OME DimensionOrder (fastest->slowest) from an on-disk storage order
	// (slowest-first). For a C-order array the last axis varies fastest, so the
	// OME order is the reverse of storageOrder, padded with any absent axes kept
	// in OME's canonical relative order so the result is always a legal value.
	inline std::string ome_dimension_order_from_storage(const std::string& storageOrder)
	{
		std::string rev(storageOrder.rbegin(), storageOrder.rend());
		for (char ax : std::string("XYZCT"))
			if (rev.find(ax) == std::string::npos)
				rev.push_back(ax);
		return rev;
	}
}
