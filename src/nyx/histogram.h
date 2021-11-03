#pragma once

#include <algorithm> 
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include "pixel.h"

#define N_HISTO_BINS 100	// Max number of histogram channels. 5% step.

using HistoItem = unsigned int;

class OnlineHistogram
{
public:
	OnlineHistogram();

	void reset();

	// Diagnostic output
	void print(bool logScale, std::string header_text);

	// Appends a value to the histogram
	void add_observation(HistoItem x);

	void build_histogram();

	HistoItem get_median();

	HistoItem get_mode();

	// Returns
	//	[0] median
	// 	[1] mode
	//	[2-5] p10, p25, 075, p90
	//	[6] IQR
	//	[7] RMAD
	//	[8] entropy
	//	[9] uniformity
	std::tuple<HistoItem, HistoItem, double, double, double, double, double, double, double, double> get_stats();

protected:
	std::unordered_set <HistoItem> uniqIntensities;
	std::unordered_map <HistoItem, int> intensityCounts;
	std::vector <int> binCounts; // Populated in build_histogram(). Always has size N_HISTO_BINS
	std::vector <double> binEdges; // Populated in build_histogram(). Always has size (N_HISTO_BINS+1)
	HistoItem min_, max_;
	double binW;
};

class TrivialHistogram
{
public:
	TrivialHistogram() {}

	void initialize (HistoItem min_value, HistoItem max_value, std::vector<Pixel2> raw_data)
	{
		// Cache min/max
		minVal = min_value;
		maxVal = max_value;

		// Build the histogram
		binW = double(maxVal - minVal) / double(N_HISTO_BINS);
		for (auto s : raw_data)
		{
			HistoItem h = s.inten;
			// 1
			double realIdx = double(h - minVal) / binW;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins [idx]) ++;

			// 2
			U.push_back(h); 
		}
	}

	void initialize(HistoItem min_value, HistoItem max_value, std::vector<HistoItem> raw_data)
	{
		// Cache min/max
		minVal = min_value;
		maxVal = max_value;

		// Build the histogram
		binW = double(maxVal - minVal) / double(N_HISTO_BINS);
		for (auto h : raw_data)
		{
			// 1
			double realIdx = double(h - minVal) / binW;
			int idx = std::isnan(realIdx) ? 0 : int(realIdx);
			(bins[idx])++;

			// 2
			U.push_back(h); 
		}
	}

	// Returns
	//	[0] median
	// 	[1] mode
	//	[2-5] p10, p25, 075, p90
	//	[6] IQR
	//	[7] RMAD
	//	[8] entropy
	//	[9] uniformity
	std::tuple<double, HistoItem, double, double, double, double, double, double, double, double, double, double> get_stats()
	{
		double median = get_median (U); //xxx	 get_median(U);
		HistoItem mode = get_mode(U);

		double p1 = 0, p10 = 0, p25 = 0, p75 = 0, p90 = 0, p99 = 0, iqr = 0, rmad = 0, entropy = 0, uniformity = 0;

		// %
		auto ppb = 100.0 / N_HISTO_BINS;
		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			if (ppb * i <= 1)
				p1 += bins[i];
			if (ppb * i <= 10)
				p10 += bins[i];
			if (ppb * i <= 25)
				p25 += bins[i];
			if (ppb * i <= 75)
				p75 += bins[i];
			if (ppb * i <= 90)
				p90 += bins[i];
			if (ppb * i <= 99)
				p99 += bins[i];
		}

		// IQR
		iqr = p75 - p25;

		// RMAD 10-90 %
		double range = maxVal - minVal;
		double lowBound = minVal + range * 0.1,
			uprBound = minVal + range * 0.9;
		double mean = 0.0;
		int n = 0;
		for (int i = 0; i < N_HISTO_BINS; i++) 
		{
			double binC = (minVal + binW * i + minVal + binW * i + binW) / 2.f;	// Bin center
			if (binC >= lowBound && binC <= uprBound)
			{
				auto cnt = bins[i];
				mean += binC * cnt;
				n += cnt;
			}
		}

		// Correction for the case where no items fall into the 10-90% range 
		if (n == 0)
			n = 1;

		mean /= n;

		double sum = 0.0;
		for (int i = 0; i < N_HISTO_BINS; i++) 
		{
			double binC = (minVal + binW * i + minVal + binW * i + binW) / 2.f;	// Bin center
			if (binC >= lowBound && binC <= uprBound)
			{
				double absDelta = std::abs(binC - mean);
				sum += absDelta;
			}
		}

		rmad = sum / n;

		// entropy & uniformity
		entropy = 0.0;
		uniformity = 0.0;
		for (int i=0; i<N_HISTO_BINS; i++)
		{
			auto cnt = bins[i];
			double binEntry = double(cnt) / double(n);
			if (fabs(binEntry) < 1e-15)
				continue;

			// entropy
			entropy -= binEntry * log2(binEntry);  //if bin is not empty

			// uniformity
			uniformity += binEntry * binEntry;
		}

		// Finally
		return { median, mode, p1, p10, p25, p75, p90, p99, iqr, rmad, entropy, uniformity };
	}

	protected:

		HistoItem minVal, maxVal;
		double medianVal;
		double binW;
		int bins[N_HISTO_BINS + 1] = { 0 };
		std::vector <HistoItem> U;	//xxx	std::unordered_set <HistoItem> U;

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
			for (int i = 1; i < N_HISTO_BINS; i++)
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
			auto n = maxVal - minVal;

			// Degenerate case?
			if (n == 0)
				return minVal;

			std::vector<int> histogram (n+1, 0);
			for (int i = 0; i < raw_I.size(); ++i)
			{
				auto k = raw_I[i] - minVal;
				++histogram [k];
			}
			HistoItem maxel = std::max_element (histogram.begin(), histogram.end()) - histogram.begin();
			return maxel + minVal;
		}
};


