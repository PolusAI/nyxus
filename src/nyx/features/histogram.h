#pragma once

#include <algorithm> 
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include "../helpers/helpers.h"
#include "pixel.h"
#include "image_matrix_nontriv.h"

using HistoItem = unsigned int;

class TrivialHistogram
{
public:

	TrivialHistogram() {}

	template <class Pxl>
	void initialize (int n_cust_bins, HistoItem min_value, HistoItem max_value, const std::vector<Pxl>& raw_data)
	{
		// safety
		int n_custBins = std::abs(n_cust_bins);

		pop_ = raw_data.size();

		// Allocate 
		// -- "percentile"
		bins100_.reserve(100);
		for (int i = 0; i < 100 + 1; i++)
			bins100_.push_back(0);
		// -- "uint8"
		bins_cust_.reserve(n_custBins);
		for (int i = 0; i < n_custBins + 1; i++)
			bins_cust_.push_back(0);

		// Cache min/max
		minVal_ = min_value;
		maxVal_ = max_value;
		auto valRange = maxVal_ - minVal_;

		// unique values
		for (auto s : raw_data)
			U_.push_back (s.inten);	

		// Build the "percentile" histogram
		binW100_ = double(valRange) / 100.;
		for (auto s : raw_data)
		{
			HistoItem h = s.inten;
			double realIdx = double(h - minVal_) / binW100_;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins100_[idx])++;
		}

		// -- Fix the special last bin
		bins100_[100 - 1] += bins100_[100];
		bins100_[100] = 0;

		// Build the "uint8" histogram
		binWcust_ = double(valRange) / double(n_custBins-1);
		for (auto s : raw_data)
		{
			HistoItem h = Nyxus::to_grayscale(s.inten, minVal_, valRange, n_custBins);
			bins_cust_[h] = bins_cust_[h] + 1;
		}

		// -- Fix the special last bin
		bins_cust_[n_custBins - 1] += bins_cust_[n_custBins];
		bins_cust_[n_custBins] = 0;

		// Mean calculation
		meanVal_ = 0;
		for (auto s : raw_data)
			meanVal_ += double(s.inten);
		meanVal_ /= double(pop_);

		// percentiles
		calc_percentiles();

		// robust MAD
		mean1090val_ = 0.0;
		size_t pop1090 = 0;
		for (auto pxl : raw_data)
		{
			double a = double(pxl.inten);
			if (a >= p10_ && a <= p90_)
			{
				mean1090val_ += a;
				pop1090++;
			}
		}
		rmad_ = 0.0;
		if (pop1090)
		{
			mean1090val_ /= double(pop1090);
			for (auto pxl : raw_data)
			{
				double a = double(pxl.inten);
				if (a >= p10_ && a <= p90_)
					rmad_ += (std::fabs)(a - mean1090val_);
			}
			rmad_ /= double(pop1090);
		}
	}

	void initialize (int n_cust_bins, HistoItem min_value, HistoItem max_value, const OutOfRamPixelCloud& raw_data)
	{
		// safety
		int n_custBins = std::abs(n_cust_bins);

		pop_ = raw_data.size();

		// Allocate 
		// -- "percentile"
		bins100_.reserve(100);
		for (int i = 0; i < 100 + 1; i++)
			bins100_.push_back(0);
		// -- "uint8"
		bins_cust_.reserve(n_custBins);
		for (int i = 0; i < n_custBins + 1; i++)
			bins_cust_.push_back(0);

		// Cache min/max
		minVal_ = min_value;
		maxVal_ = max_value;
		auto valRange = maxVal_ - minVal_;

		// unique values
		for (auto s : raw_data)
			U_.push_back (s.inten);	

		// Build the "percentile" histogram
		binW100_ = double(valRange) / 100.;
		for (auto s : raw_data)
		{
			HistoItem h = s.inten;
			double realIdx = double(h - minVal_) / binW100_;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins100_[idx])++;
		}

		// -- Fix the special last bin
		bins100_[100 - 1] += bins100_[100];
		bins100_[100] = 0;

		// Build the "uint8" histogram
		binWcust_ = double(valRange) / double(n_custBins-1);
		for (auto s : raw_data)
		{
			HistoItem h = Nyxus::to_grayscale(s.inten, minVal_, valRange, n_custBins);
			bins_cust_[h] = bins_cust_[h] + 1;
		}

		// -- Fix the special last bin
		bins_cust_[n_custBins - 1] += bins_cust_[n_custBins];
		bins_cust_[n_custBins] = 0;

		// Mean calculation
		meanVal_ = 0;
		for (auto s : raw_data)
			meanVal_ += double(s.inten);
		meanVal_ /= double(pop_);

		// percentiles
		calc_percentiles();

		// robust mean
		bin10ctr_ = bins100_ [10];
		bin90ctr_ = bins100_ [90];
		mean1090val_ = 0.0;
		size_t pop1090 = 0;
		for (auto pxl : raw_data)
		{
			double a = double (pxl.inten);
			if (a >= bin10ctr_ && a <= bin90ctr_)
			{
				mean1090val_ += a;
				pop1090++;
			}
		}
		rmad_ = 0.0;
		if (pop1090)
		{
			for (auto pxl : raw_data)
			{
				double a = double(pxl.inten);
				if (a >= bin10ctr_ && a <= bin90ctr_)
					rmad_ += (std::fabs)(a - mean1090val_);
			}
			rmad_ /= double(pop1090);
		}
	}

	void initialize_uniques (const std::vector<HistoItem>& raw_data)
	{
		for (auto h : raw_data)
			U_.push_back(h);
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
		if (U_.size() == 0)
			return { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

		// Interquartile range
		double iqr = p75_ - p25_;

		// Median
		double median = get_median(U_);

		// Mode
		HistoItem mode = get_mode(U_);

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

	HistoItem get_mode()
	{
		// Empty histogram?
		if (U_.size() == 0)
			return 0;

		// Mode
		HistoItem mode = get_mode(U_);

		return mode;
	}

	double get_median()
	{
		// Empty histogram?
		if (U_.size() == 0)
			return 0;

		// Median
		double median = get_median(U_);

		return median;
	}

private:

	size_t pop_ = 0;
	HistoItem minVal_, maxVal_;
	double meanVal_, binW100_, binWcust_;
	double bin10ctr_, bin90ctr_, mean1090val_, rmad_;	// robust estimation
	std::vector<HistoItem> bins100_, bins_cust_;
	std::vector<HistoItem> U_;
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

	HistoItem get_median(const std::unordered_set<HistoItem>& uniqueValues)
	{
		// Sort unique intensities
		std::vector<HistoItem> A{ uniqueValues.begin(), uniqueValues.end() };
		std::sort(A.begin(), A.end());

		// Pick the median
		auto n = A.size();
		if (n % 2 != 0)
		{
			int median = A[n / 2];
			return median;
		}
		else
		{
			HistoItem right = A[n / 2],
				left = A[n / 2 - 1],	// Middle left and right values
				ave = (right + left) / 2;
			return ave;
		}
	}

	// 'raw_I' is passed as non-const and gets sorted
	double get_median(std::vector<HistoItem>& raw_I)
	{
		// Sort unique intensities
		std::sort(raw_I.begin(), raw_I.end());

		// Pick the median
		auto n = raw_I.size();
		if (n % 2 != 0)
		{
			HistoItem median = raw_I[n / 2];
			return (double)median;
		}
		else
		{
			HistoItem right = raw_I[n / 2],
				left = raw_I[n / 2 - 1];	// Middle left and right values
			double ave = double(right + left) / 2.0;
			return ave;
		}
	}

	HistoItem get_mode(std::vector<HistoItem>& raw_I)
	{
		// Populate the frequency map
		std::map<HistoItem, std::size_t> freqMap;
		for (int v : raw_I)
			++freqMap[v];

		// Iterator to the highest frequency item
		auto highestFreqIter = freqMap.begin();

		// Iterate the map updating 'highestFreqIter'
		for (auto iter = freqMap.begin(); iter != freqMap.end(); ++iter)
		{
			if (highestFreqIter->second < iter->second)
				highestFreqIter = iter;
		}

		// Return the result
		auto mo = highestFreqIter->first;
		return mo;
	}

	double bin_center(size_t bin_idx, double bin_width)
	{
		return double(bin_idx) * bin_width * 1.5;
	}
};

