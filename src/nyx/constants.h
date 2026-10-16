#pragma once

constexpr double DEFAULT_T_INDEX = 0.0;
constexpr double DEFAULT_C_INDEX = 0.0;		// default channel index for single-channel / non-OME input
constexpr int DEFAULT_NUM_HISTO_BINS = 24;

// The number of bins the intensity histograms take from a coarse grey depth. Its sign selects a
// texture binning scheme, which a histogram has no use for, so only the magnitude is a bin count;
// 0, which bins no texture family at all, leaves a histogram at its default.
constexpr int histogram_bin_count (int coarse_gray_depth)
{
	return coarse_gray_depth > 0 ? coarse_gray_depth
		: coarse_gray_depth < 0 ? -coarse_gray_depth
		: DEFAULT_NUM_HISTO_BINS;
}
