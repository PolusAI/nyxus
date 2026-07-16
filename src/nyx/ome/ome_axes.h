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
	inline bool is_signed(PixelType t)
	{
		return t == PixelType::Int8 || t == PixelType::Int16 || t == PixelType::Int32 || t == PixelType::Int64 || is_float(t);
	}

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

	// One resolution level of a pyramid (level 0 == full-res).
	struct OmePyramidLevel
	{
		std::size_t index = 0;
		std::size_t sizeX = 0, sizeY = 0, sizeZ = 1;   // 0 == extent not yet resolved for this sub-level
		double scaleX = 1, scaleY = 1, scaleZ = 1;     // coordinateTransformations 'scale'
		std::string path;                              // Zarr dataset path ("0","1",...) / TIFF SubIFD ordinal
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

		// Physical calibration (convenience mirror of the space/time axes).
		double physX = 1, physY = 1, physZ = 1, timeIncrement = 1;
		std::string unitXY, unitZ, unitT;

		PixelType dtype = PixelType::UInt16;
		unsigned  bitsPerSample = 16;

		std::vector<OmePyramidLevel> levels;      // >=1; levels[0] full-res
		std::vector<std::string> channelNames;    // optional (OME 'Channel'/omero)

		bool valid = false;   // set true by a successful parse

		// Position of a label in storageAxes; -1 if absent.
		int storageIndexOf(char lbl) const
		{
			for (std::size_t i = 0; i < storageAxes.size(); ++i)
				if (storageAxes[i].label == lbl) return static_cast<int>(i);
			return -1;
		}

		std::size_t numberPyramidLevels() const { return levels.empty() ? 1 : levels.size(); }
		bool isVolumetric()  const { return sizeZ > 1; }
		bool isMultiChannel() const { return sizeC > 1; }
		bool isTimeSeries()  const { return sizeT > 1; }
	};

	// ---- small shared helpers ----------------------------------------------

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
