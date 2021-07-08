#include <cmath>
#include <memory>
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
std::unordered_map <int, StatsReal> labelStddev;	// Is calculated from 'lavelVariance' in Reduce()
std::unordered_map <int, StatsReal> labelCentroid_x;
std::unordered_map <int, StatsReal> labelCentroid_y;
std::unordered_map <int, StatsReal> labelM2;
std::unordered_map <int, StatsReal> labelM3;
std::unordered_map <int, StatsReal> labelM4;
std::unordered_map <int, StatsReal> labelSkewness;
std::unordered_map <int, StatsReal> labelKurtosis;
std::unordered_map <int, StatsReal> labelMAD;
std::unordered_map <int, StatsReal> labelRMS;
std::unordered_map <int, std::shared_ptr<Histo>> labelHistogram;
std::unordered_map <int, StatsReal> labelP10;
std::unordered_map <int, StatsReal> labelP25;
std::unordered_map <int, StatsReal> labelP75;
std::unordered_map <int, StatsReal> labelP90;
std::unordered_map <int, StatsReal> labelIQR;
std::unordered_map <int, StatsReal> labelEntropy;
std::unordered_map <int, StatsReal> labelMode;
std::unordered_map <int, StatsReal> labelUniformity;
std::unordered_map <int, StatsReal> labelRMAD;

// Research
StatsReal intensityMin = -999.0, intensityMax = -999.0;

constexpr int N2R = 50*1000;
constexpr int N2R_2 = 50*1000;

void init_feature_buffers()
{
	uniqueLabels.reserve (N2R);
	labelCount.reserve (N2R);
	labelPrevCount.reserve (N2R);
	labelPrevIntens.reserve (N2R);
	labelMeans.reserve (N2R);
	labelValues.reserve (N2R);
	labelMedians.reserve (N2R);
	labelMins.reserve (N2R);
	labelMaxs.reserve (N2R);
	labelMassEnergy.reserve (N2R);
	labelVariance.reserve (N2R);
	labelStddev.reserve (N2R);
	labelCentroid_x.reserve (N2R);
	labelCentroid_y.reserve (N2R);
	labelM2.reserve (N2R);
	labelM3.reserve (N2R);
	labelM4.reserve (N2R);
	labelSkewness.reserve (N2R);
	labelKurtosis.reserve (N2R);
	labelMAD.reserve (N2R);
	labelRMS.reserve (N2R);
	labelHistogram.reserve (N2R);
	labelP10.reserve (N2R);
	labelP25.reserve (N2R);
	labelP75.reserve (N2R);
	labelP90.reserve (N2R);
	labelIQR.reserve (N2R);
	labelEntropy.reserve (N2R);
	labelMode.reserve (N2R);
	labelUniformity.reserve (N2R);
	labelRMAD.reserve (N2R);
}

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
	labelStddev.clear();
	labelCentroid_x.clear();
	labelCentroid_y.clear();
	labelM2.clear();
	labelM3.clear();
	labelM4.clear();
	labelSkewness.clear();
	labelKurtosis.clear();
	labelMAD.clear();
	labelRMS.clear();
	labelHistogram.clear();
	labelP10.clear();
	labelP25.clear();
	labelP75.clear();
	labelP90.clear();
	labelIQR.clear();
	labelEntropy.clear();
	labelMode.clear();
	labelUniformity.clear();
	labelRMAD.clear();
}

void update_label_stats (int x, int y, int label, PixIntens intensity)
{
	// Research
	if (intensityMin == -999.0 || intensityMin>intensity)
		intensityMin = intensity;
	if (intensityMax == -999.0 || intensityMax<intensity)
		intensityMax = intensity;
	
    // Remember this label
    //-	uniqueLabels.insert(label);

    // Calculate features updates for this "iteration"
	
	auto it = uniqueLabels.find(label); //-	auto it = labelMeans.find(label);
	if (it == uniqueLabels.end()) //-	if (it == labelMeans.end())
	{
		// Remember this label
		uniqueLabels.insert(label);
			
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
		std::shared_ptr<std::unordered_set<PixIntens>> ptrUS = std::make_shared <std::unordered_set<PixIntens>>();
		ptrUS->reserve(N2R_2);
		ptrUS->insert(intensity);
		labelValues[label] = ptrUS;

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

		// Histogram
		std::shared_ptr<Histo> ptrH = std::make_shared <Histo>();
		ptrH->add_observation(intensity);
		labelHistogram[label] = ptrH;
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

		// Histogram
		auto ptrH = labelHistogram[label];
		ptrH->add_observation(intensity);
		
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
		
		auto n = labelCount [l];	// Cardinality of the label value set

		// Standard deviations
		labelStddev[l] = sqrt (labelVariance[l]);

		// Medians
		// --Sort unique intensities
		std::vector<int> A{ lv.second->begin(), lv.second->end() };
		std::sort (A.begin(), A.end());

		// --Pick the median
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
		labelSkewness[l] = labelM3[l] / std::pow(labelM2[l], 1.5);

		// Kurtosis
		labelKurtosis[l] = labelM4[l] / labelM2[l] * labelM2[l] - 3.0;
		
		// Root of mean squared
		labelRMS[l] = sqrt (labelMassEnergy[l] / n);

		// P10, 25, 75, 90
		auto ptrH = labelHistogram[l];
		float percentPerBin = 100.f / float(N_HISTO_BINS),
			p10 = 0.f,
			p25 = 0.f,
			p75 = 0.f,
			p90 = 0.f,
			entropy = 0.f,
			mode = ptrH->getBinStats (0), 
			uniformity = 0.f, 
			rmad = ptrH->getEstRMAD (10, 90);

		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			// %-tile
			auto bs = ptrH->getBinStats(i);
			if (float(i) * percentPerBin <= 10)
				p10 += bs;
			if (float(i) * percentPerBin <= 25)
				p25 += bs;
			if (float(i) * percentPerBin <= 75)
				p75 += bs;
			if (float(i) * percentPerBin <= 90)
				p90 += bs;

			// entropy
			double binEntry = bs / n;
			if (fabs(binEntry) < 1e-15) 
				continue;
			entropy -= binEntry * log2(binEntry);  //if bin is not empty

			// uniformity
			uniformity += binEntry * binEntry;

			// mode
			if (bs > mode)
				mode = bs;
		}
		labelP10[l] = p10;
		labelP25[l] = p25;
		labelP75[l] = p75;
		labelP90[l] = p90;
		labelIQR[l] = (StatsReal) p75 - p25;
		labelRMAD[l] = rmad;
		labelEntropy[l] = entropy;
		labelMode[l] = mode;
		labelUniformity[l] = uniformity;
	}
}




