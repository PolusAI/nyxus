#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include "sensemaker.h"

std::unordered_set<int> uniqueLabels;
/*
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
*/

//--- New
std::unordered_map <int, LR> labelData;
std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;

// Research
StatsReal intensityMin = -999.0, intensityMax = -999.0;

constexpr int N2R = 50*1000;
constexpr int N2R_2 = 50*1000;

void init_feature_buffers()
{
	uniqueLabels.reserve(N2R);
	/*
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
	*/
	labelData.reserve(N2R);
}

void clearLabelStats()
{
	uniqueLabels.clear();
	/*
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
	*/
	labelData.clear();
}

void update_label_stats(int x, int y, int label, PixIntens intensity)
{
	// Research
	if (intensityMin == -999.0 || intensityMin > intensity)
		intensityMin = intensity;
	if (intensityMax == -999.0 || intensityMax < intensity)
		intensityMax = intensity;

	// Calculate features updates for this "iteration"

	auto it = uniqueLabels.find(label); //-	auto it = labelMeans.find(label);
	if (it == uniqueLabels.end()) //-	if (it == labelMeans.end())
	{
		// Remember this label
		uniqueLabels.insert(label);

		//--- New
		LR lr;

		lr.labelCount = 1;
		lr.labelPrevCount = 0;
		// Min
		lr.labelMins = intensity;
		// Max
		lr.labelMaxs = intensity;
		// Moments
		lr.labelMeans = intensity;
		lr.labelM2 = 0;
		lr.labelM3 = 0;
		lr.labelM4 = 0;
		// Median. Cache intensity values per label for the median calculation
		std::shared_ptr<std::unordered_set<PixIntens>> ptrUS = std::make_shared <std::unordered_set<PixIntens>>();
		//---time consuming---	ptrUS->reserve(N2R_2);
		ptrUS->insert(intensity);
		lr.labelValues = ptrUS;
		// Energy
		lr.labelMassEnergy = intensity;
		// Variance and standard deviation
		lr.labelVariance = 0.0;
		// Mean absolute deviation
		lr.labelMAD = 0;
		// Previous intensity
		lr.labelPrevIntens = intensity;
		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		lr.labelCentroid_x = StatsReal(x) + 1;
		lr.labelCentroid_y = StatsReal(y) + 1;
		// Histogram
		std::shared_ptr<Histo> ptrH = std::make_shared <Histo>();
		ptrH->add_observation(intensity);
		lr.labelHistogram = ptrH;
		//
		labelData[label] = lr;
	}
	else
	{
#ifdef SIMULATE_WORKLOAD
		// Simulate a chunk of processing. The cost is ~300 mks
		for (long tmp = 0; tmp < 1 * 1000; tmp++)
			auto start = std::chrono::system_clock::now();
#endif

		//--- New
		LR& lr = labelData[label];
		// Count of pixels belonging to the label
		auto prev_n = lr.labelCount;	// Previous count
		lr.labelPrevCount = prev_n;
		auto n = prev_n + 1;	// New count
		lr.labelCount = n;

		// Cumulants for moments calculation
		auto prev_mean = lr.labelMeans;
		auto delta = intensity - prev_mean;
		auto delta_n = delta / n;
		auto delta_n2 = delta_n * delta_n;
		auto term1 = delta * delta_n * prev_n;

		// Mean
		auto mean = prev_mean + delta_n;
		lr.labelMeans = mean;

		// Moments
		lr.labelM4 = lr.labelM4 + term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * lr.labelM2 - 4 * delta_n * lr.labelM3;
		lr.labelM3 = lr.labelM3 + term1 * delta_n * (n - 2) - 3 * delta_n * lr.labelM2;
		lr.labelM2 = lr.labelM2 + term1;

		// Median
		auto ptr = lr.labelValues;
		ptr->insert(intensity);

		// Min 
		lr.labelMins = std::min(lr.labelMins, (StatsInt)intensity);

		// Max
		lr.labelMaxs = std::min(lr.labelMins, (StatsInt)intensity);

		// Energy
		lr.labelMassEnergy = lr.labelMassEnergy + intensity;

		// Variance and standard deviation
		if (n >= 2)
		{
			double s_prev = lr.labelVariance,
				diff = double(intensity) - prev_mean,
				diff2 = diff * diff;
			lr.labelVariance = (n - 2) * s_prev / (n - 1) + diff2 / n;
		}
		else
			lr.labelVariance = 0;

		// Mean absolute deviation
		lr.labelM2 = lr.labelM2 + sqrt(delta * (intensity - mean));
		lr.labelMAD = lr.labelM2 / n;

		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		lr.labelCentroid_x = lr.labelCentroid_x + StatsReal(x) + 1;
		lr.labelCentroid_y = lr.labelCentroid_y + StatsReal(y) + 1;

		// Histogram
		auto ptrH = lr.labelHistogram;
		ptrH->add_observation(intensity);

		// Previous intensity for succeeding iterations
		lr.labelPrevIntens = intensity;
	}
}

std::mutex glock;

void update_label_stats_parallel (int x, int y, int label, PixIntens intensity)
{
	// Create a mutex
	std::mutex* mux;
	{
		std::lock_guard<std::mutex> lg(glock);

		auto itm = labelMutexes.find(label);
		if (itm == labelMutexes.end())
		{
			//=== Create a label-specific mutex
			itm = labelMutexes.emplace(label, std::make_unique <std::mutex>()).first;

			//=== Create a label record
			auto it = uniqueLabels.find(label); 
			if (it != uniqueLabels.end())
				std::cout << "\n\tERROR\n";

			// Remember this label
			uniqueLabels.insert(label);

			//--- New
			LR lr;

			lr.labelCount = 1;
			lr.labelPrevCount = 0;
			// Min
			lr.labelMins = intensity;
			// Max
			lr.labelMaxs = intensity;
			// Moments
			lr.labelMeans = intensity;
			lr.labelM2 = 0;
			lr.labelM3 = 0;
			lr.labelM4 = 0;
			// Median. Cache intensity values per label for the median calculation
			std::shared_ptr<std::unordered_set<PixIntens>> ptrUS = std::make_shared <std::unordered_set<PixIntens>>();
			//---time consuming---	ptrUS->reserve(N2R_2);
			ptrUS->insert(intensity);
			lr.labelValues = ptrUS;
			// Energy
			lr.labelMassEnergy = intensity;
			// Variance and standard deviation
			lr.labelVariance = 0.0;
			// Mean absolute deviation
			lr.labelMAD = 0;
			// Previous intensity
			lr.labelPrevIntens = intensity;
			// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
			lr.labelCentroid_x = StatsReal(x) + 1;
			lr.labelCentroid_y = StatsReal(y) + 1;
			// Histogram
			std::shared_ptr<Histo> ptrH = std::make_shared <Histo>();
			ptrH->add_observation(intensity);
			lr.labelHistogram = ptrH;
			//
			labelData[label] = lr;
		}

		mux = itm->second.get();
	}


	// Research
	if (intensityMin == -999.0 || intensityMin > intensity)
		intensityMin = intensity;
	if (intensityMax == -999.0 || intensityMax < intensity)
		intensityMax = intensity;

	// Calculate features updates for this "iteration"

	//else
	{
		std::lock_guard<std::mutex> lock(*mux);

#ifdef SIMULATE_WORKLOAD
		// Simulate a chunk of processing. The cost is ~300 mks
		auto start = std::chrono::system_clock::now();

		for (long tmp = 0; tmp < 1 * 1000; tmp++)
			auto start = std::chrono::system_clock::now();

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double, std::milli> elap = end - start;
		std::cout << "\tidle[ms]: " << elap.count() << std::endl;
#endif
		//--- New
		LR& lr = labelData[label];
		// Count of pixels belonging to the label
		auto prev_n = lr.labelCount;	// Previous count
		lr.labelPrevCount = prev_n;
		auto n = prev_n + 1;	// New count
		lr.labelCount = n;

		// Cumulants for moments calculation
		auto prev_mean = lr.labelMeans;
		auto delta = intensity - prev_mean;
		auto delta_n = delta / n;
		auto delta_n2 = delta_n * delta_n;
		auto term1 = delta * delta_n * prev_n;

		// Mean
		auto mean = prev_mean + delta_n;
		lr.labelMeans = mean;

		// Moments
		lr.labelM4 = lr.labelM4 + term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * lr.labelM2 - 4 * delta_n * lr.labelM3;
		lr.labelM3 = lr.labelM3 + term1 * delta_n * (n - 2) - 3 * delta_n * lr.labelM2;
		lr.labelM2 = lr.labelM2 + term1;

		// Median
		auto ptr = lr.labelValues;
		ptr->insert(intensity);

		// Min 
		lr.labelMins = std::min(lr.labelMins, (StatsInt)intensity);

		// Max
		lr.labelMaxs = std::min(lr.labelMins, (StatsInt)intensity);

		// Energy
		lr.labelMassEnergy = lr.labelMassEnergy + intensity;

		// Variance and standard deviation
		if (n >= 2)
		{
			double s_prev = lr.labelVariance,
				diff = double(intensity) - prev_mean,
				diff2 = diff * diff;
			lr.labelVariance = (n - 2) * s_prev / (n - 1) + diff2 / n;
		}
		else
			lr.labelVariance = 0;

		// Mean absolute deviation
		lr.labelM2 = lr.labelM2 + sqrt(delta * (intensity - mean));
		lr.labelMAD = lr.labelM2 / n;

		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		lr.labelCentroid_x = lr.labelCentroid_x + StatsReal(x) + 1;
		lr.labelCentroid_y = lr.labelCentroid_y + StatsReal(y) + 1;

		// Histogram
		auto ptrH = lr.labelHistogram;
		ptrH->add_observation(intensity);

		// Previous intensity for succeeding iterations
		lr.labelPrevIntens = intensity;
	}
}

/*
 * 
 * This function should be called once after a file pair processing is finished.
 *
 */

void do_partial_stats_reduction()
{
	for (auto& ld : labelData) // for (auto& lv : labelValues)
	{
		auto l = ld.first;
		auto lr = ld.second;
		auto lv = lr.labelValues;
		/*
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
		*/

		//--- New
		auto n = lr.labelCount;	// Cardinality of the label value set

		// Standard deviations
		lr.labelStddev = sqrt(lr.labelVariance);

		// Medians
		// --Sort unique intensities
		std::vector<int> A{ lv->begin(), lv->end() };
		std::sort(A.begin(), A.end());

		// --Pick the median
		if (A.size() % 2 != 0)
		{
			int median = A[A.size() / 2];
			lr.labelMedians = median;
		}
		else
		{
			int right = A[A.size() / 2],
				left = A[A.size() / 2 - 1],	// Middle left and right values
				ave = (right + left) / 2;
			lr.labelMedians = ave;
		}

		// Skewness
		lr.labelSkewness = lr.labelM3 / std::pow(lr.labelM2, 1.5);

		// Kurtosis
		lr.labelKurtosis = lr.labelM4 / lr.labelM2 * lr.labelM2 - 3.0;

		// Root of mean squared
		lr.labelRMS = sqrt(lr.labelMassEnergy / n);

		// P10, 25, 75, 90
		auto ptrH = lr.labelHistogram;
		float percentPerBin = 100.f / float(N_HISTO_BINS),
			p10 = 0.f,
			p25 = 0.f,
			p75 = 0.f,
			p90 = 0.f,
			entropy = 0.f,
			mode = ptrH->getBinStats(0),
			uniformity = 0.f,
			rmad = ptrH->getEstRMAD(10, 90);

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
		lr.labelP10 = p10;
		lr.labelP25 = p25;
		lr.labelP75 = p75;
		lr.labelP90 = p90;
		lr.labelIQR = (StatsReal)p75 - p25;
		lr.labelRMAD = rmad;
		lr.labelEntropy = entropy;
		lr.labelMode = mode;
		lr.labelUniformity = uniformity;

	}
}




