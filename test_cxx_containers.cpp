//
// Quick & dirty test of containers. Part of future G-test of Sensemaker
//

#include <chrono>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <tuple>
#include "sensemaker.h"

constexpr int N = 25 * 1000 * 1000;	// Mocks # of pixels in an image (the # of iterations)
constexpr int M = 5000;	// Mocks the # of unique labels

//
// Returns the time in seconds
//
double test_containers1()
{
	std::cout << "test_containers1 (" << N << ") begins..." << std::endl;

	std::unordered_map <int, StatsInt> pixelCount;
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
	std::unordered_map <int, StatsReal> centroid_x;
	std::unordered_map <int, StatsReal> centroid_y;
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

	// --Timing
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < N; i++)
	{
		int p = i % M;	// position

		pixelCount[p] = pixelCount[p] + i;
		labelPrevCount[p] = i;
		labelPrevIntens[p] = i;
		labelMeans[p] = i;
		//std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
		labelMedians[p] = i;
		labelMins[p] = i;
		labelMaxs[p] = i;
		labelMassEnergy[p] = i;
		labelVariance[p] = i;
		labelStddev[p] = i;
		centroid_x[p] = i;
		centroid_y[p] = i;
		labelM2[p] = i;
		labelM3[p] = i;
		labelM4[p] = i;
		labelSkewness[p] = i;
		labelKurtosis[p] = i;
		labelMAD[p] = i;
		labelRMS[p] = i;
		//std::unordered_map <int, std::shared_ptr<Histo>> labelHistogram;
		labelP10[p] = i;
		labelP25[p] = i;
		labelP75[p] = i;
		labelP90[p] = i;
		labelIQR[p] = i;
		labelEntropy[p] = i;
		labelMode[p] = i;
		labelUniformity[p] = i;
		labelRMAD[p] = i;
	}

	// --Timing
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed1 = end - start;

	std::cout << "test_containers(" << N << "): " << elapsed1.count() << std::endl;

	return elapsed1.count();
}

//
// Returns the time in seconds
//
double test_containers2()
{
	std::cout << "test_containers2 (" << N << ") begins..." << std::endl;

	using Tup = std::tuple <

		StatsInt, //std::unordered_map <int, StatsInt> pixelCount;
		StatsInt, //std::unordered_map <int, StatsInt> labelPrevCount;
		StatsInt, //std::unordered_map <int, StatsInt> labelPrevIntens;
		StatsReal, //std::unordered_map <int, StatsReal> labelMeans;
		//---std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
		StatsInt, //std::unordered_map <int, StatsInt> labelMedians;
		StatsInt, //std::unordered_map <int, StatsInt> labelMins;
		StatsInt, //std::unordered_map <int, StatsInt> labelMaxs;
		StatsInt, //std::unordered_map <int, StatsInt> labelMassEnergy;
		StatsReal, //std::unordered_map <int, StatsReal> labelVariance;
		StatsReal, //std::unordered_map <int, StatsReal> labelStddev;	// Is calculated from 'lavelVariance' in Reduce()
		StatsReal, //std::unordered_map <int, StatsReal> centroid_x;
		StatsReal, //std::unordered_map <int, StatsReal> centroid_y;
		StatsReal, //std::unordered_map <int, StatsReal> labelM2;
		StatsReal, //std::unordered_map <int, StatsReal> labelM3;
		StatsReal, //std::unordered_map <int, StatsReal> labelM4;
		StatsReal, //std::unordered_map <int, StatsReal> labelSkewness;
		StatsReal, //std::unordered_map <int, StatsReal> labelKurtosis;
		StatsReal, //std::unordered_map <int, StatsReal> labelMAD;
		StatsReal, //std::unordered_map <int, StatsReal> labelRMS;
	//---std::unordered_map <int, std::shared_ptr<Histo>> labelHistogram;
		StatsReal, //std::unordered_map <int, StatsReal> labelP10;
		StatsReal, //std::unordered_map <int, StatsReal> labelP25;
		StatsReal, //std::unordered_map <int, StatsReal> labelP75;
		StatsReal, //std::unordered_map <int, StatsReal> labelP90;
		StatsReal, //std::unordered_map <int, StatsReal> labelIQR;
		StatsReal, //std::unordered_map <int, StatsReal> labelEntropy;
		StatsReal, //std::unordered_map <int, StatsReal> labelMode;
		StatsReal, //std::unordered_map <int, StatsReal> labelUniformity;
		StatsReal //std::unordered_map <int, StatsReal> labelRMAD;
	> ;

	std::unordered_map <int, Tup> label;

	// --Timing
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < N; i++)
	{
		Tup t(i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i);

		int p = i % M;	// position
		
		if (i==0)
			label[p] = t;	// Initialize
		else
		{
			Tup& _t = label[p];
			std::get<0>(_t) = std::get<0>(_t) + i;
		}
	}

	// --Timing
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed1 = end - start;

	std::cout << "test_containers(" << N << "): " << elapsed1.count() << std::endl;

	return elapsed1.count();
}

