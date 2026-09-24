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
	// disk-backed clouds (OutOfRamPixelCloud, OutOfRamVoxelCloud). All expose size() and a
	// range-for yielding an item with an .inten field, so a single template serves them and the
	// in-core / out-of-core histograms stay identical by construction (fixes that had to be
	// applied to each copy -- e.g. the p10/p90 robust-MAD -- now live in one place).
	template <class Src>
	void initialize (int n_cust_bins, HistoItem min_value, HistoItem max_value, const Src& raw_data)
	{
		// safety
		int n_custBins = std::abs(n_cust_bins);

		pop_ = raw_data.size();

		// Cache min/max
		minVal_ = min_value;
		maxVal_ = max_value;
		auto valRange = maxVal_ - minVal_;

		// The ONE pass over the intensity source, into the value frequencies every statistic
		// below is derived from. A disk-backed cloud is therefore read once, and the frequencies
		// cost one entry per distinct grey level rather than one per voxel -- what lets an
		// out-of-core ROI of a billion voxels be summarized in bounded memory.
		freq_.clear();
		for (auto s : raw_data)
			++ freq_ [s.inten];

		// Allocate
		// -- "percentile"
		bins100_.assign (100 + 1, 0);
		// -- "uint8"
		bins_cust_.assign (n_custBins + 1, 0);

		binW100_ = double(valRange) / 100.;
		binWcust_ = double(valRange) / double(n_custBins-1);

		meanVal_ = 0;
		for (const auto& vc : freq_)
		{
			const double v = double(vc.first);
			const double cnt = double(vc.second);

			// the "percentile" histogram
			double realIdx = (v - double(minVal_)) / binW100_;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			bins100_[idx] += (HistoItem) vc.second;

			// the "uint8" histogram. A constant ROI has no range to bin over: to_grayscale()
			// would divide by it and cast the resulting NaN to a bin index, which is undefined
			// and only happens to land on 0 on the toolchains in use. Every sample of a constant
			// ROI belongs to the first bin, which is what the percentile histogram above already
			// says through its NaN guard.
			HistoItem h = valRange ? Nyxus::to_grayscale (vc.first, minVal_, valRange, n_custBins) : 0;
			bins_cust_[h] += (HistoItem) vc.second;

			meanVal_ += v * cnt;
		}

		// -- Fix the special last bins
		bins100_[100 - 1] += bins100_[100];
		bins100_[100] = 0;
		bins_cust_[n_custBins - 1] += bins_cust_[n_custBins];
		bins_cust_[n_custBins] = 0;

		// Mean calculation
		meanVal_ /= double(pop_);

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
	double meanVal_, binW100_, binWcust_;
	double mean1090val_, rmad_;	// robust estimation (p10/p90-thresholded)
	std::vector<HistoItem> bins100_, bins_cust_;
	// how many samples carry each value: one entry per distinct grey level, not per sample
	std::map<HistoItem, std::size_t> freq_;
	double p1_, p10_, p25_, p75_, p90_, p99_;

	void calc_percentiles()
	{
		p1_ = p10_ = p25_ = p75_ = p90_ = p99_ = 0;

		double cnt_p1 = double(pop_) * 0.01,
			cnt_p10 = double(pop_) * 0.1,
			cnt_p25 = double(pop_) * 0.25,
			cnt_p75 = double(pop_) * 0.75,
			cnt_p90 = double(pop_) * 0.9,
			cnt_p99 = double(pop_) * 0.99;
		size_t runSum = 0;
		for (int i = 0; i < 100; i++)
		{
			// interpolate
			if (runSum <= cnt_p1 && cnt_p1 <= runSum + bins100_[i])
				p1_ = (cnt_p1 - runSum) * binW100_ / double(bins100_[i]) + minVal_ + binW100_ * i;
			if (runSum <= cnt_p10 && cnt_p10 <= runSum + bins100_[i])
				p10_ = (cnt_p10 - runSum) * binW100_ / double(bins100_[i]) + minVal_ + binW100_ * i;
			if (runSum <= cnt_p25 && cnt_p25 <= runSum + bins100_[i])
				p25_ = (cnt_p25 - runSum) * binW100_ / double(bins100_[i]) + minVal_ + binW100_ * i;
			if (runSum <= cnt_p75 && cnt_p75 <= runSum + bins100_[i])
				p75_ = (cnt_p75 - runSum) * binW100_ / double(bins100_[i]) + minVal_ + binW100_ * i;
			if (runSum <= cnt_p90 && cnt_p90 <= runSum + bins100_[i])
				p90_ = (cnt_p90 - runSum) * binW100_ / double(bins100_[i]) + minVal_ + binW100_ * i;
			if (runSum <= cnt_p99 && cnt_p99 <= runSum + bins100_[i])
				p99_ = (cnt_p99 - runSum) * binW100_ / double(bins100_[i]) + minVal_ + binW100_ * i;

			runSum += bins100_[i];
		}
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

