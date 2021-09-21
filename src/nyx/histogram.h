#pragma once

#include <algorithm> 
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include "pixel.h"

#define N_HISTO_BINS 20	// Max number of histogram channels. 5% step.

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


