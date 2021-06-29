#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include "sensemaker.h"

std::unordered_set<int> uniqueLabels;
std::unordered_map <int, StatsInt> labelCount;
std::unordered_map <int, StatsInt> labelPrevCount;
std::unordered_map <int, StatsInt> labelPrevIntens;
std::unordered_map <int, StatsReal> labelMeans;
std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
std::unordered_map <int, StatsInt> labelMedians;
std::unordered_map <int, StatsInt> labelMins;
std::unordered_map <int, StatsInt> labelMaxs;
std::unordered_map <int, StatsInt> labelMassEnergy;
std::unordered_map <int, StatsReal> labelVariance;
std::unordered_map <int, StatsReal> labelCentroid_x;
std::unordered_map <int, StatsReal> labelCentroid_y;
std::unordered_map <int, StatsReal> labelM2;
std::unordered_map <int, StatsReal> labelM3;
std::unordered_map <int, StatsReal> labelM4;
std::unordered_map <int, StatsReal> labelSkewness;
std::unordered_map <int, StatsReal> labelKurtosis;
std::unordered_map <int, StatsReal> labelMAD;


void clearLabelStats()
{
	uniqueLabels.clear();
	labelCount.clear();
	labelPrevCount.clear();
	labelPrevIntens.clear();
	labelMeans.clear();
	labelValues.clear();
	labelMedians.clear();
	labelMins.clear();
	labelMaxs.clear();
	labelMassEnergy.clear();
	labelVariance.clear();
	labelCentroid_x.clear();
	labelCentroid_y.clear();
	labelM2.clear();
	labelM3.clear();
	labelM4.clear();
	labelSkewness.clear();
	labelKurtosis.clear();
	labelMAD.clear();
}

/*
 *
 * This function should be called per each pixel.
 *
	"min",
	"max",
	"range",
	"mean",
	"median",
	"standard_deviation",
	"skewness",
	"kurtosis",
	"mean_absolute_deviation",
	"energy",
"root_mean_squared",
"entropy",
"mode",
"uniformity",
"p10",
"p25",
"p75",
"p90",
"interquartile_range",
"robust_mean_absolute_deviation",
	"weighted_centroid_y",	// wndchrm output[20]
	"weighted_centroid_x"	// wndchrm output[21]
 */

void update_label_stats (int x, int y, int label, PixIntens intensity)
{
    // Remember this label
    uniqueLabels.insert(label);

    // Calculate features updates for this "iteration"
	auto it = labelMeans.find(label);
	if (it == labelMeans.end())
	{
		// Count of pixels belonging to the label
		labelCount[label] = 1;
		labelPrevCount[label] = 0;

		// Min
		labelMins[label] = intensity;

		// Max
		labelMaxs[label] = intensity;

		// Moments
		labelMeans[label] = intensity;
		labelM2[label] = 0;
		labelM3[label] = 0;
		labelM4[label] = 0;

		// Median. Cache intensity values per label for the median calculation
		std::shared_ptr<std::unordered_set<PixIntens>> ptr = std::make_shared <std::unordered_set<PixIntens>>();
		ptr->insert(intensity);
		labelValues[label] = ptr;

        // Energy
        labelMassEnergy[label] = intensity;

        // Variance and standard deviation
        labelVariance[label] = 0.0;

		// Mean absolute deviation
		labelMAD[label] = 0;

        // Previous intensity
        labelPrevIntens[label] = intensity;

		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		labelCentroid_x [label] = StatsReal(x) + 1;
		labelCentroid_y [label] = StatsReal (y) + 1;
    }
	else
	{
		// Count of pixels belonging to the label
        auto prev_n = labelCount[label];	// Previous count
		labelPrevCount[label] = prev_n;
		auto n = prev_n + 1;	// New count
		labelCount[label] = n;

		// Cumulants for moments calculation
		auto prev_mean = labelMeans[label];
		auto delta = intensity - prev_mean;
		auto delta_n = delta / n;
		auto delta_n2 = delta_n * delta_n;
		auto term1 = delta * delta_n * prev_n;

		// Mean
		auto mean = prev_mean + delta_n;
		labelMeans[label] = mean;

		// Moments
		labelM4 [label] = labelM4[label] + term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * labelM2[label] - 4 * delta_n * labelM3[label];
		labelM3 [label] = labelM3[label] + term1 * delta_n * (n - 2) - 3 * delta_n * labelM2[label];
		labelM2 [label] = labelM2[label] + term1;

		// Median
		auto ptr = labelValues[label];
		ptr->insert(intensity); 

		// Min 
		labelMins[label] = std::min (labelMins[label], (StatsInt)intensity);

		// Max
		labelMaxs[label] = std::min(labelMins[label], (StatsInt)intensity);

        // Energy
        labelMassEnergy[label] = labelMassEnergy[label] + intensity;

        // Variance and standard deviation
        if (n >= 2)
        {
            double s_prev = labelVariance[label],
                diff = double(intensity) - prev_mean,
                diff2 = diff * diff;
            labelVariance[label] = (n - 2) * s_prev / (n - 1) + diff2 / n;
        }
        else
            labelVariance[label] = 0;

		// Mean absolute deviation
		labelM2[label] = labelM2[label] + sqrt(delta * (intensity - mean));
		labelMAD[label] = labelM2[label] / n;

		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		labelCentroid_x[label] = labelCentroid_x[label] + StatsReal(x) + 1;
		labelCentroid_y[label] = labelCentroid_y[label] + StatsReal(y) + 1;
		
		// Previous intensity for succeeding iterations
        labelPrevIntens[label] = intensity;
    }
}

/*
 * 
 * This function should be called once after a file pair processing is finished.
 *
 */

void do_partial_stats_reduction()
{
	for (auto& lv : labelValues)
	{
		auto l = lv.first;
		// Sort unique intensities
		std::vector<int> A{ lv.second->begin(), lv.second->end() };
		std::sort (A.begin(), A.end());

		// Pick the median
		if (A.size() % 2 != 0)
		{
			int median = A[A.size() / 2];
			labelMedians[l] = median;
		}
		else
		{
			int right = A[A.size() / 2],
				left = A[A.size() / 2 - 1],	// Middle left and right values
				ave = (right + left) / 2;
			labelMedians[l] = ave;
		}

		// Skewness
		labelSkewness[l] = labelM3[l] / pow(labelM2[l], 1.5);

		// Kurtosis
		labelKurtosis[l] = labelM4[l] / labelM2[l] * labelM2[l] - 3.0;
	}
}




