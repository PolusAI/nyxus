#pragma once

#include <algorithm> 
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <tuple>
#include "../helpers/helpers.h"
#include "pixel.h"
#include "image_matrix_nontriv.h"
#include "voxel_cloud_nontriv.h"

using HistoItem = unsigned int;

class TrivialHistogram
{
public:

	TrivialHistogram() {}

	// One implementation for every intensity source: the in-RAM std::vector<Pxl>, and the two
	// disk-backed clouds (OutOfRamPixelCloud, OutOfRamVoxelCloud). All a source needs is a
	// range-for yielding an item with an .inten field -- its size() is not consulted, the
	// population is counted as it is read -- so a single template serves them all and the
	// in-core / out-of-core histograms stay identical by construction.
	template <class Src>
	void initialize (int n_cust_bins, HistoItem min_value, HistoItem max_value, const Src& raw_data)
	{
		// safety
		int n_custBins = std::abs(n_cust_bins);

		// Cache min/max
		minVal_ = min_value;
		maxVal_ = max_value;
		auto valRange = maxVal_ - minVal_;

		// The ONE pass over the intensity source, into the value frequencies every statistic
		// below is derived from. A disk-backed cloud is therefore read once, and the frequencies
		// cost one entry per distinct grey level rather than one per voxel -- what lets an
		// out-of-core ROI of a billion voxels be summarized in bounded memory. The population is
		// the count of what this pass read, not the source's size(), so the ranks and the bin
		// frequencies describe the same samples.
		freq_.clear();
		pop_ = 0;
		for (auto s : raw_data)
		{
			++ freq_ [s.inten];
			++ pop_;
		}

		// Allocate the "uint8" histogram
		bins_cust_.assign (n_custBins + 1, 0);

		for (const auto& vc : freq_)
		{
			// the "uint8" histogram. A constant ROI has no range to bin over: to_grayscale()
			// would divide by it and cast the resulting NaN to a bin index, which is undefined
			// and only happens to land on 0 on the toolchains in use. Every sample of a constant
			// ROI belongs to the first bin.
			HistoItem h = valRange ? Nyxus::to_grayscale (vc.first, minVal_, valRange, n_custBins) : 0;
			bins_cust_[h] += (HistoItem) vc.second;
		}

		// -- Fix the special last bin
		bins_cust_[n_custBins - 1] += bins_cust_[n_custBins];
		bins_cust_[n_custBins] = 0;

		// percentiles
		calc_percentiles();

		// robust MAD, from the same frequencies
		mean1090val_ = 0.0;
		size_t pop1090 = 0;
		for (const auto& vc : freq_)
			if (double(vc.first) >= p10_ && double(vc.first) <= p90_)
			{
				mean1090val_ += double(vc.first) * double(vc.second);
				pop1090 += vc.second;
			}
		rmad_ = 0.0;
		if (pop1090)
		{
			mean1090val_ /= double(pop1090);
			for (const auto& vc : freq_)
				if (double(vc.first) >= p10_ && double(vc.first) <= p90_)
					rmad_ += (std::fabs) (double(vc.first) - mean1090val_) * double(vc.second);
			rmad_ /= double(pop1090);
		}
	}

	// How many distinct values the histogram currently describes. The map holds one entry per
	// value, not per item, which is what bounds its footprint on a cloud of any size.
	std::size_t n_distinct() const { return freq_.size(); }

	// The value frequencies of 'raw_data', for a caller that needs only the median and the mode of
	// a set of values (e.g. chord lengths) rather than a full intensity histogram. Each call
	// describes ITS data: a caller that summarizes two sets through one instance (chords does,
	// with the max chords and then all chords) gets the second set's statistics, not both sets'.
	void initialize_uniques (const std::vector<HistoItem>& raw_data)
	{
		freq_.clear();
		pop_ = 0;
		for (auto h : raw_data)
		{
			++ freq_ [h];
			++ pop_;
		}
	}
	// Returns
	//	[0] median
	// 	[1] mode
	//	[2-7] p1, p10, p25, p75, p90, p99
	//	[8] IQR 
	//	[9] RMAD
	//	[10] entropy
	//	[11] uniformity
	std::tuple<double, HistoItem, double, double, double, double, double, double, double, double, double, double> get_stats()
	{
		// Empty histogram?
		if (freq_.empty())
			return { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

		// Interquartile range
		double iqr = p75_ - p25_;

		// Median
		double median = get_median();

		// Mode
		HistoItem mode = get_mode();

		// entropy and uniformity
		double entropy = 0.0, uniformity = 0.0;
		for (auto cnt : bins_cust_)
		{
			double p = double(cnt) / double(pop_);
			entropy += p * log2(p + 2.2e-16);
			uniformity += p*p;
		}

		return { median, mode, p1_, p10_, p25_, p75_, p90_, p99_, iqr, rmad_, -entropy, uniformity };
	}

	// The most frequent value; the smallest of them if several share the top frequency.
	HistoItem get_mode() const
	{
		HistoItem mode = 0;
		std::size_t top = 0;
		for (const auto& vc : freq_)
			if (vc.second > top)
			{
				top = vc.second;
				mode = vc.first;
			}
		return mode;
	}

	// The median of the samples: the middle one, or the average of the two middle ones when the
	// population is even. Read off the value frequencies, so nothing is sorted or held per sample.
	double get_median() const
	{
		if (freq_.empty() || pop_ == 0)
			return 0;

		if (pop_ % 2 != 0)
			return (double) sample_at_rank (pop_ / 2);

		HistoItem left = sample_at_rank (pop_ / 2 - 1),
			right = sample_at_rank (pop_ / 2);
		return (double(left) + double(right)) / 2.0;
	}

	// --- Histogram bin exposure (per-ROI intensity histogram) -----------------
	// The "custom-resolution" histogram has 'n_cust_bins' bins spanning
	// [minVal_, maxVal_]. Bin index of intensity i is floor((i-min)/range*n)
	// (see Nyxus::to_grayscale), so bin k covers [min + k*range/n, min + (k+1)*range/n)
	// and its center is min + (k+0.5)*range/n. We expose the raw per-bin
	// frequencies plus the cached min/max so a caller can reconstruct the bin
	// edges knowing only the bin count.

	// Per-bin frequencies of the custom-resolution histogram, trimmed to exactly
	// 'n_cust_bins' bins (the internal vector carries one extra, folded, slot).
	std::vector<double> get_cust_frequencies (int n_cust_bins) const
	{
		int n = std::abs(n_cust_bins);
		std::vector<double> v;
		v.reserve(n);
		for (int k = 0; k < n; k++)
			v.push_back (k < static_cast<int>(bins_cust_.size()) ? static_cast<double>(bins_cust_[k]) : 0.0);
		return v;
	}

	HistoItem get_min_value() const { return minVal_; }
	HistoItem get_max_value() const { return maxVal_; }
	size_t get_population() const { return pop_; }

private:

	size_t pop_ = 0;
	HistoItem minVal_, maxVal_;
	double mean1090val_, rmad_;	// robust estimation (p10/p90-thresholded)
	std::vector<HistoItem> bins_cust_;
	// how many samples carry each value: one entry per distinct grey level, not per sample
	std::map<HistoItem, std::size_t> freq_;
	double p1_, p10_, p25_, p75_, p90_, p99_;

	// Each q-quantile of the samples, read off the value frequencies: the two samples whose ranks
	// straddle (pop_ - 1) * q, interpolated linearly between. That is the default ('linear') method of
	// numpy.percentile, which PyRadiomics and MIRP take their percentiles with. The six q ascend, so
	// the lower ranks ascend and so do the upper ones, and one walk of the frequencies resolves both
	// sequences. They are walked as two sequences, not interleaved: two q can share a lower rank,
	// which puts the first one's upper rank above the second one's lower rank.
	void calc_percentiles()
	{
		double* const out[] = { &p1_, &p10_, &p25_, &p75_, &p90_, &p99_ };
		const double qs[] = { 0.01, 0.1, 0.25, 0.75, 0.9, 0.99 };
		constexpr int n_q = sizeof(qs) / sizeof(qs[0]);

		if (pop_ == 0)
		{
			for (auto p : out)
				*p = 0;
			return;
		}

		std::size_t lo[n_q], hi[n_q];
		double frac[n_q];
		for (int i = 0; i < n_q; i++)
		{
			const double h = double(pop_ - 1) * qs[i];
			lo[i] = (std::size_t) std::floor (h);
			hi[i] = std::min (lo[i] + 1, pop_ - 1);
			frac[i] = h - double(lo[i]);
		}

		// the samples of those ranks; hi[i] >= lo[i], so once every upper rank is resolved every
		// lower one is too. pop_ is the sum of the counts, so every rank below it is resolved.
		HistoItem a[n_q], b[n_q];
		int i_lo = 0, i_hi = 0;
		std::size_t seen = 0;
		for (const auto& vc : freq_)
		{
			seen += vc.second;
			while (i_lo < n_q && lo[i_lo] < seen)
				a[i_lo++] = vc.first;
			while (i_hi < n_q && hi[i_hi] < seen)
				b[i_hi++] = vc.first;
			if (i_hi == n_q)
				break;
		}

		for (int i = 0; i < n_q; i++)
			*out[i] = double(a[i]) + frac[i] * (double(b[i]) - double(a[i]));
	}

	// The sample of rank 'k' (0-based) in value order, walked off the frequencies.
	HistoItem sample_at_rank (std::size_t k) const
	{
		std::size_t seen = 0;
		for (const auto& vc : freq_)
		{
			seen += vc.second;
			if (k < seen)
				return vc.first;
		}
		return freq_.empty() ? 0 : freq_.rbegin()->first;
	}
};

