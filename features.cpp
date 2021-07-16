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
std::unordered_map <int, LR> labelData;
std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;

constexpr int N2R = 50 * 1000;
constexpr int N2R_2 = 50 * 1000;

// Preallocates the intensely accessed main containers
void init_feature_buffers()
{
	uniqueLabels.reserve(N2R);
	labelData.reserve(N2R);
	labelMutexes.reserve(N2R);
}

// Resets the main containers
void clearLabelStats()
{
	uniqueLabels.clear();
	labelData.clear();
	labelMutexes.clear();
}

// Label Record (structure 'LR') is where the state of label's pixels scanning and feature calculations is maintained. This function initializes an LR instance for the 1st pixel.
void init_label_record (LR& lr, int x, int y, int label, PixIntens intensity)
{
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
	lr.labelUniqueIntensityValues = ptrUS;
	// Energy
	lr.labelMassEnergy = intensity * intensity;
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
	// Other fields
	lr.labelMedians = 0;
	lr.labelStddev = 0;
	lr.labelSkewness = 0;
	lr.labelKurtosis = 0;
	lr.labelRMS = 0;
	lr.labelP10 = lr.labelP25 = lr.labelP75 = lr.labelP90 = 0;
	lr.labelIQR = 0;
	lr.labelEntropy = 0;
	lr.labelMode = 0;
	lr.labelUniformity = 0;
	lr.labelRMAD = 0;

	#ifdef SANITY_CHECK_INTENSITIES
	// Dump intensities for testing
	if (label == 1)	// Put your label code of interest
		lr.raw_intensities.push_back(intensity);
	#endif

}

// This function 'digests' the 2nd and the following pixel of a label and updates the label's feature calculation state - the instance of structure 'LR'
void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity)
{
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
	auto ptr = lr.labelUniqueIntensityValues;
	ptr->insert(intensity);

	// Min 
	lr.labelMins = std::min(lr.labelMins, (StatsInt)intensity);

	// Max
	lr.labelMaxs = std::max(lr.labelMaxs, (StatsInt)intensity);

	// Energy
	lr.labelMassEnergy = lr.labelMassEnergy + intensity * intensity;

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
	lr.labelMAD = lr.labelMAD + std::abs(intensity - mean);

	// Weighted centroids. Do we need to make them 1-based for compatibility with Matlab and WNDCHRM?
	lr.labelCentroid_x = lr.labelCentroid_x + StatsReal(x);
	lr.labelCentroid_y = lr.labelCentroid_y + StatsReal(y);

	// Histogram
	auto ptrH = lr.labelHistogram;
	ptrH->add_observation(intensity);

	// Previous intensity for succeeding iterations
	lr.labelPrevIntens = intensity;

	#ifdef SANITY_CHECK_INTENSITIES
	// Dump intensities for testing
	if (label == 1)	// Put the label code you're tracking
		lr.raw_intensities.push_back(intensity);
	#endif
}

// The root function of handling a pixel being scanned
void update_label_stats(int x, int y, int label, PixIntens intensity)
{
	auto it = uniqueLabels.find(label);
	if (it == uniqueLabels.end())
	{
		// Remember this label
		uniqueLabels.insert(label);

		// Initialize the label record
		LR lr;
		init_label_record (lr, x, y, label, intensity);
		labelData[label] = lr;
	}
	else
	{
		#ifdef SIMULATE_WORKLOAD_FACTOR
		// Simulate a chunk of processing. 1K iterations cost ~300 mks
		for (long tmp = 0; tmp < SIMULATE_WORKLOAD_FACTOR * 1000; tmp++)
			auto start = std::chrono::system_clock::now();
		#endif

		// Update label's stats
		LR& lr = labelData[label];
		update_label_record (lr, x, y, label, intensity);
	}
}

// This function should be called once after a file pair processing is finished.
void do_partial_stats_reduction()
{
	for (auto& ld : labelData) // for (auto& lv : labelUniqueIntensityValues)
	{
		auto l = ld.first;		// Label code
		auto& lr = ld.second;	// Label record
		auto lv = lr.labelUniqueIntensityValues;	// Label's unique intensities 

		auto n = lr.labelCount;	// Cardinality of the label value set

		// Mean absolute deviation
		lr.labelMAD = lr.labelMAD / n;

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
		lr.labelSkewness = std::sqrt(double(lr.labelCount)) * lr.labelM3 / std::pow(lr.labelM2, 1.5);	// skewness = sqrt(length(data)) * m3 / (m2^1.5);

		// Kurtosis
		lr.labelKurtosis = double(lr.labelCount) * lr.labelM4 / (lr.labelM2 * lr.labelM2) - 3.0;	// kurtosis = (n * m4) / (m2 * m2) - 3;

		// Root of mean squared
		lr.labelRMS = sqrt(lr.labelMassEnergy / n);

		// P10, 25, 75, 90
		auto ptrH = lr.labelHistogram;
		float percentPerBin = 100.f / float(N_HISTO_BINS),
			entropy = 0.f,
			mode = ptrH->getBinStats(0),
			uniformity = 0.f,
			rmad = ptrH->getEstRMAD(10, 90);
		int idxP10 = -1, idxP25 = -1, idxP75 = -1, idxP90 = -1;
		int modeIdx = 0;
		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			// %-tile
			auto bs = ptrH->getBinStats(i);
			if (percentPerBin * i <= 10)
				idxP10 = i;
			if (percentPerBin * i <= 25)
				idxP25 = i;
			if (percentPerBin * i <= 75)
				idxP75 = i;
			if (percentPerBin * i <= 90)
				idxP90 = i;

			// entropy
			double binEntry = bs / n;
			if (fabs(binEntry) < 1e-15)
				continue;
			entropy -= binEntry * log2(binEntry);  //if bin is not empty

			// uniformity
			uniformity += binEntry * binEntry;

			// mode
			if (bs > mode)
			{
				mode = bs;
				modeIdx = i;
			}
		}
		lr.labelP10 = ptrH->getBinValue(idxP10);
		lr.labelP25 = ptrH->getBinValue(idxP25);
		lr.labelP75 = ptrH->getBinValue(idxP75);
		lr.labelP90 = ptrH->getBinValue(idxP90);
		lr.labelIQR = lr.labelP75 - lr.labelP25;
		lr.labelRMAD = rmad;
		lr.labelEntropy = entropy;
		lr.labelMode = ptrH->getBinValue(modeIdx); //mode;
		lr.labelUniformity = uniformity;

		// Weighted centroids
		lr.labelCentroid_x = lr.labelCentroid_x / lr.labelCount;
		lr.labelCentroid_y = lr.labelCentroid_y / lr.labelCount;
	}
}




