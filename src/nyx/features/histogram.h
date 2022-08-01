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

using HistoItem = unsigned int;

class TrivialHistogram
{
public:
	TrivialHistogram() {}

	void initialize (HistoItem min_value, HistoItem max_value, const std::vector<Pixel2>& raw_data)
	{
		// Allocate 
		// -- "binary"
		population = raw_data.size();
		n_bins = decltype(n_bins) ((1. + log2(population)) + 0.5);
		bins.reserve(n_bins+1);
		for (int i = 0; i < n_bins+1; i++)
			bins.push_back(0);
		// -- "percentile"
		bins100.reserve(100);
		for (int i = 0; i < 100+1; i++)
			bins100.push_back(0);
		// -- "uint8"
		bins256.reserve(256);
		for (int i = 0; i < 256+1; i++)
			bins256.push_back(0);

		// Cache min/max
		minVal = min_value;
		maxVal = max_value;
		auto valRange = maxVal - minVal;

		// Build the "binary" histogram
		binW = double(valRange) / double(n_bins);
		for (auto s : raw_data)
		{
			HistoItem h = s.inten;

			double realIdx = double(h-minVal) / binW;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins [idx]) ++;

			U.push_back(h); // Initialize the set for mode and median calculation
		}

		// -- Fix the special last bin
		bins[n_bins - 1] += bins[n_bins];
		bins[n_bins] = 0;
		
		// Build the "percentile" histogram
		binW100 = double(valRange) / 100.;
		for (auto s : raw_data)
		{
			HistoItem h = s.inten;
			double realIdx = double(h - minVal) / binW100;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins100[idx])++;
		}

		// -- Fix the special last bin
		bins100[100 - 1] += bins100[100];
		bins100[100] = 0;

		// Build the "uint8" histogram
		binw256 = double(valRange) / 255.;
		for (auto s : raw_data)
		{
			HistoItem h = Nyxus::to_grayscale (s.inten, minVal, valRange, 256);
			bins256[h] = bins256[h] + 1;
		}

		// -- Fix the special last bin
		bins256[256 - 1] += bins256[256];
		bins256[256] = 0;

		// Mean calculation
		meanVal = 0; 
		for (auto s : raw_data)
			meanVal += double(s.inten); 
		meanVal /= double(population); 
	}

	void initialize_uniques (const std::vector<HistoItem> & raw_data)
	{
		for (auto h : raw_data)
			U.push_back(h); 
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
		if (U.size() == 0)
			return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

		// Median
		double median = get_median (U); 

		// Mode
		HistoItem mode = get_mode (U);

		// Percentiles
		double p1 = 0, p10 = 0, p25 = 0, p75 = 0, p90 = 0, p99 = 0;
		double cnt_p1 = double(population) * 0.01 + 0.5,
			cnt_p10 = double(population) * 0.1 + 0.5,
			cnt_p25 = double(population) * 0.25 + 0.5,
			cnt_p75 = double(population) * 0.75 + 0.5,
			cnt_p90 = double(population) * 0.9 + 0.5,
			cnt_p99 = double(population) * 0.99 + 0.5;
		size_t runSum = 0;
		for (int i = 0; i < 100; i++)
		{
			// interpolate
			if (runSum <= cnt_p1 && cnt_p1 <= runSum + bins100[i])
				p1 = (cnt_p1 - runSum) * binW100 / double(bins100[i]) + minVal + binW100 * i;
			if (runSum <= cnt_p10 && cnt_p10 <= runSum + bins100[i])
				p10 = (cnt_p10 - runSum) * binW100 / double(bins100[i]) + minVal + binW100 * i;
			if (runSum <= cnt_p25 && cnt_p25 <= runSum + bins100[i])
				p25 = (cnt_p25 - runSum) * binW100 / double(bins100[i]) + minVal + binW100 * i;
			if (runSum <= cnt_p75 && cnt_p75 <= runSum + bins100[i])
				p75 = (cnt_p75 - runSum) * binW100 / double(bins100[i]) + minVal + binW100 * i;
			if (runSum <= cnt_p90 && cnt_p90 <= runSum + bins100[i])
				p90 = (cnt_p90 - runSum) * binW100 / double(bins100[i]) + minVal + binW100 * i;
			if (runSum <= cnt_p99 && cnt_p99 <= runSum + bins100[i])
				p99 = (cnt_p99 - runSum) * binW100 / double(bins100[i]) + minVal + binW100 * i;

			runSum += bins100[i];
		}

		// Interquartile range
		double iqr = p75 - p25;
		
		// RMAD 10-90 %
		double rmad = 0, entropy = 0, uniformity = 0;
		double range = maxVal - minVal;
		double lowBound = minVal + range * 0.1,
			uprBound = minVal + range * 0.9;

		// -- Calculate the 10-90% range population and mean
		double sum1090 = 0.0;
		size_t population1090 = 0;
		for (int i = 0; i < n_bins; i++)
		{
			double binC = (minVal + binW * i + minVal + binW * i + binW) / 2.f;	// Bin center
			if (binC >= lowBound && binC <= uprBound)
			{
				sum1090 += binC;
				population1090++;
			}
		}

		// -- Correction for the case where no items fall into the 10-90% range 
		if (population1090 == 0)
			rmad = 0;
		else
		{
			double mean1090 = sum1090 / double(population1090);

			double sum1090 = 0.0;
			for (int i = 0; i < n_bins; i++)
			{
				double binC = (minVal + binW * i + minVal + binW * i + binW) / 2.f;	// Bin center
				if (binC >= lowBound && binC <= uprBound)
				{
					double absDelta = std::abs (binC - mean1090);
					sum1090 += absDelta;
				}
			}

			rmad = sum1090 / double(population1090);
		}

		// entropy
		entropy = 0.0;
		for (int i=0; i<n_bins; i++)
		{
			auto cnt = bins[i];

			// skip empty bins (zero probabilities)
			if (cnt == 0)
				continue;
			
			// calculate the probability by normalizing the bin entry so that sum(normalized bin entries)==1
			double p = double(cnt) / double(population);

			// skip near-zero probabilities
			if (fabs(p) < 1e-15)
				continue;

			// entropy
			entropy += p * log2(p);  
		}

		// uniformity
		uniformity = 0.0;
		for (int i = 0; i < 256; i++)
		{
			auto cnt = bins256[i];
			uniformity += std::pow(cnt, 2);
		}

		return { median, mode, p1, p10, p25, p75, p90, p99, iqr, rmad, -entropy, uniformity };
	}

	HistoItem get_mode()
	{
		// Empty histogram?
		if (U.size() == 0)
			return 0;

		// Mode
		HistoItem mode = get_mode(U);

		return mode;
	}

	double get_median()
	{
		// Empty histogram?
		if (U.size() == 0)
			return 0;

		// Median
		double median = get_median(U);

		return median;
	}

	private:
		size_t population = 0;
		HistoItem minVal, maxVal;
		double medianVal, meanVal;
		double binW, binW100, binw256;
		std::vector<HistoItem> bins, bins100, bins256; 
		int n_bins = 0;
		std::vector<HistoItem> U;	

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
		double get_median (std::vector<HistoItem> & raw_I)
		{
			// Sort unique intensities
			std::sort (raw_I.begin(), raw_I.end());

			// Pick the median
			auto n = raw_I.size();
			if (n % 2 != 0)
			{
				HistoItem median = raw_I[n / 2];
				return (double) median;
			}
			else
			{
				HistoItem right = raw_I[n / 2],
					left = raw_I[n / 2 - 1];	// Middle left and right values
				double ave = double(right + left) / 2.0;
				return ave;
			}
		}

		double get_max_bin_item ()
		{
			// Find the heaviest bin
			int maxIdx = 0;
			auto maxCnt = bins[0];
			for (int i = 1; i < n_bins; i++)
				if (maxCnt < bins[i])
				{
					maxCnt = bins[i];
					maxIdx = i;
				}

			// Find bin enges and their average
			double edgeL = minVal + maxIdx * binW,
				edgeR = edgeL + binW;
			double biggest = HistoItem((edgeL + edgeR) / 2.0);
			return biggest;
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

		double bin_center (size_t bin_idx, double bin_width)
		{
			return double(bin_idx) * bin_width * 1.5;
		}
};


