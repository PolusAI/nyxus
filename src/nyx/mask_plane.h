#pragma once

#include <cstddef>

namespace Nyxus
{
	/// @brief The mask plane that pairs with intensity plane (channel, timeframe). Frames pair
	/// N:N when the mask has as many frames as the intensity, and 1:N (mask frame 0) when one
	/// mask serves every intensity frame. A mask is usually channel-agnostic, so it is read at
	/// the requested channel only when it has that many, and at channel 0 otherwise. Both image
	/// loaders (ImageLoader for featurization, RawImageLoader for the prescan) pair through this,
	/// so every volumetric consumer reads the same mask plane.
	inline void mask_plane_for (std::size_t channel, std::size_t timeframe,
		std::size_t inten_timeframes, std::size_t mask_channels, std::size_t mask_timeframes,
		std::size_t& mask_channel, std::size_t& mask_timeframe)
	{
		mask_timeframe = (mask_timeframes == inten_timeframes && timeframe < mask_timeframes) ? timeframe : 0;
		mask_channel = (channel < mask_channels) ? channel : 0;
	}
}
