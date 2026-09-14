#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

namespace Nyxus
{
	namespace detail
	{
		// The bound both narrowings saturate at. (double) of a 32-bit maximum is exact, so this is
		// the real bound rather than a nearby representable one.
		template <class T>
		inline double grey_level_top ()
		{
			static_assert (std::is_integral<T>::value && ! std::is_signed<T>::value,
				"a grey level is stored as an unsigned integer");
			static_assert (sizeof(T) <= 4,
				"grey_level_rounded() narrows through long long; a 64-bit grey level needs its own path");
			return (double) (std::numeric_limits<T>::max)();
		}
	}

	// The tail every load-time map ends in: one value, already shifted or rescaled into the
	// non-negative domain the pipeline stores, narrowed to an unsigned grey level.
	//
	// Converting a double to an unsigned integer is undefined, not a wrapping cast, whenever the
	// value is non-finite, negative, or above the destination's maximum. So every narrowing here
	// takes a non-finite value to grey level 0 (it carries no intensity), clamps a negative one to
	// 0, and saturates one above the maximum. A value above the maximum is what a slide whose range
	// exceeds the grey type's produces: a 64-bit TIFF or NIfTI, a real-valued volume carried on the
	// offset map by --preserve-hu, a rescale slope large enough to stretch one, or a --fpimgdr above
	// the grey type's maximum on the quantized map.
	//
	// The narrowings differ only in their last step, because the two kinds of map mean different
	// things by a grey level.

	// Round to nearest. The offset map under --preserve-hu: there a grey level is one intensity
	// unit and the recorded inverse carries a scale of 1, so a fraction the cast dropped could never
	// be recovered.
	template <class T>
	inline T grey_level_rounded (double y)
	{
		if (! std::isfinite (y) || y < 0.0)
			return (T) 0;
		if (y >= detail::grey_level_top<T> ())
			return (std::numeric_limits<T>::max)();
		return (T) std::llround (y);
	}

	// Truncate. The quantized map, where a grey level is a bin index and the bin a value falls in is
	// the one below it; the offset map without --preserve-hu, which is what a real-valued slide read
	// without the flag has always done; and the accessors that read integer labels.
	template <class T>
	inline T grey_level_truncated (double y)
	{
		if (! std::isfinite (y) || y < 0.0)
			return (T) 0;
		if (y >= detail::grey_level_top<T> ())
			return (std::numeric_limits<T>::max)();
		return (T) y;
	}

	// The offset map's narrowing, with the rounding its caller recorded. Every loader's offset map
	// and SlideProps::to_grey_level() go through this one call, so the forward map cannot narrow
	// differently from the loaders it mirrors.
	template <class T>
	inline T grey_level (double y, bool round_to_nearest)
	{
		return round_to_nearest ? grey_level_rounded<T> (y) : grey_level_truncated<T> (y);
	}
}
