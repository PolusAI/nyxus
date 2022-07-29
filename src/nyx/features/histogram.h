#pragma once

#include <algorithm> 
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
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
		n_bins = (1. + log2(population)) + 0.5;
		bins.reserve(n_bins+1);
		for (int i = 0; i < n_bins+1; i++)
			bins.push_back(0);
		// -- "percentile"
		bins100.reserve(100);
		for (int i = 0; i < 100+1; i++)
			bins100.push_back(0);

		// Cache min/max
		minVal = min_value;
		maxVal = max_value;

		// Build the "binary" histogram
		binW = double(maxVal-minVal) / double(n_bins);
		for (auto s : raw_data)
		{
			HistoItem h = s.inten;

			// 1
			double realIdx = double(h-minVal) / binW;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins [idx]) ++;

			// 2
			U.push_back(h); 
		}

		// -- Fix the special last bin
		bins[n_bins - 1] += bins[n_bins];
		bins[n_bins] = 0;
		
		// Build the "percentile" histogram
		binW100 = double(maxVal-minVal) / 100.;
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

		// Mean calculation
		meanVal = 0; 
		for (auto s : raw_data)
			meanVal += double(s.inten); 
		meanVal /= double(population); 
	}

	void initialize (HistoItem min_value, HistoItem max_value, const std::vector<HistoItem>& raw_data)
	{
		// Allocate 
		population = raw_data.size();
		n_bins = (1. + log2(population)) + 0.5;
		bins.reserve(n_bins + 1);
		for (int i = 0; i < n_bins + 1; i++)
			bins.push_back(0);

		// Cache min/max
		minVal = min_value;
		maxVal = max_value;

		// Build the histogram
		binW = double(maxVal-minVal) / double(n_bins);
		meanVal = 0; // Mean calculation
		for (auto h : raw_data)
		{
			// 1
			double realIdx = double(h - minVal) / binW;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins[idx])++;

			// 2
			U.push_back(h); 
			
			meanVal += h; // Mean calculation
		}
		meanVal /= double(population); // Mean calculation
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

		double median = get_median (U); 
		HistoItem mode = get_mode (U);

		double p1 = bin_center(1, binW100),
			p10 = bin_center(10, binW100),
			p25 = bin_center(25, binW100),
			p75 = bin_center(75, binW100),
			p90 = bin_center(90, binW100),
			p99 = bin_center(99, binW100),
			iqr = 0, rmad = 0, entropy = 0, uniformity = 0;

		// RMAD 10-90 %
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

		// entropy & uniformity
		entropy = 0.0;
		uniformity = 0.0;
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

			// uniformity
			uniformity += std::pow (cnt, 2);
		}

		return { median, mode, p1, p10, p25, p75, p90, p99, iqr, rmad, -entropy, uniformity };
	}

	private:
		size_t population = 0;
		HistoItem minVal, maxVal;
		double medianVal, meanVal;
		double binW, binW100;
		std::vector<HistoItem> bins, bins100; 
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
			int maxIdx = 0,
				maxCnt = bins[0];
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


