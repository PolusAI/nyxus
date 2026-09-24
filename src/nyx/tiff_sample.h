#pragma once

// The C++ type that stores one TIFF sample, shared by the four TIFF loaders (the strip and tile
// loaders of both the AbstractTileLoader and RawFormatLoader stacks). Each loader reads a
// sample through the type with_tiff_sample_type() hands it, so a buffer is indexed at the file's
// own sample width and every loader accepts the same (SampleFormat, BitsPerSample) pairs.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace Nyxus
{
	/// @brief An IEEE 754 half-precision sample (SampleFormat 3, 16 bits). It is 2 bytes wide, so
	/// a buffer of them is indexed like the file, and it widens to any arithmetic type on read.
	struct TiffHalf
	{
		std::uint16_t bits;

		double to_double() const
		{
			const int sign = (bits >> 15) & 1,
				exponent = (bits >> 10) & 0x1f,
				mantissa = bits & 0x3ff;
			double v;
			if (exponent == 0)
				v = std::ldexp ((double) mantissa, -24);								// zero, subnormal
			else
				if (exponent == 0x1f)
					v = mantissa ? std::nan("") : HUGE_VAL;							// NaN, infinity
				else
					v = std::ldexp ((double) (mantissa | 0x400), exponent - 25);	// normal
			return sign ? -v : v;
		}

		template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
		operator T() const { return static_cast<T> (to_double()); }
	};
	static_assert (sizeof(TiffHalf) == 2, "a TiffHalf buffer must be indexed at the file's sample width");

	/// @brief True for the real-valued sample types (SampleFormat 3).
	template <class T>
	constexpr bool tiff_sample_is_real = std::is_floating_point_v<T> || std::is_same_v<T, TiffHalf>;

	/// @brief Sample 'idx' of a buffer of T samples, widened to double / narrowed to uint32.
	template <class T>
	double tiff_sample_as_double (const void* buf, std::size_t idx)
	{
		return (double) static_cast<const T*> (buf)[idx];
	}
	template <class T>
	std::uint32_t tiff_sample_as_uint32 (const void* buf, std::size_t idx)
	{
		return (std::uint32_t) static_cast<const T*> (buf)[idx];
	}

	/// @brief Call f with a value of the C++ type that stores one sample of the given TIFF
	/// SampleFormat (1 unsigned, 2 signed, 3 IEEE real) and BitsPerSample.
	/// @param who Loader name for the error message
	/// Throws for a pair no loader reads.
	template <class F>
	void with_tiff_sample_type (short sample_format, short bits_per_sample, const char* who, F&& f)
	{
		switch (sample_format)
		{
		case 1:
			switch (bits_per_sample)
			{
			case 8:  f (std::uint8_t{});  return;
			case 16: f (std::uint16_t{}); return;
			case 32: f (std::uint32_t{}); return;
			case 64: f (std::uint64_t{}); return;
			}
			break;
		case 2:
			switch (bits_per_sample)
			{
			case 8:  f (std::int8_t{});  return;
			case 16: f (std::int16_t{}); return;
			case 32: f (std::int32_t{}); return;
			case 64: f (std::int64_t{}); return;
			}
			break;
		case 3:
			switch (bits_per_sample)
			{
			case 16: f (TiffHalf{});   return;
			case 32: f (float{});      return;
			case 64: f (double{});     return;
			}
			break;
		}
		throw std::runtime_error (std::string (who) + ": unsupported TIFF sample format " + std::to_string (sample_format)
			+ " with " + std::to_string (bits_per_sample) + " bits per sample");
	}
}
