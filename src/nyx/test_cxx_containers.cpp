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

	std::unordered_map <int, StatsInt> pixelCountRoiArea;
	std::unordered_map <int, StatsInt> aux_PrevCount;
	std::unordered_map <int, StatsInt> aux_PrevIntens;
	std::unordered_map <int, StatsReal> mean;
	std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
	std::unordered_map <int, StatsInt> median;
	std::unordered_map <int, StatsInt> min;
	std::unordered_map <int, StatsInt> max;
	std::unordered_map <int, StatsInt> massEnergy;
	std::unordered_map <int, StatsReal> variance;
	std::unordered_map <int, StatsReal> stddev;	// Is calculated from 'lavelVariance' in Reduce()
	std::unordered_map <int, StatsReal> centroid_x;
	std::unordered_map <int, StatsReal> centroid_y;
	std::unordered_map <int, StatsReal> aux_M2;
	std::unordered_map <int, StatsReal> aux_M3;
	std::unordered_map <int, StatsReal> aux_M4;
	std::unordered_map <int, StatsReal> skewness;
	std::unordered_map <int, StatsReal> kurtosis;
	std::unordered_map <int, StatsReal> MAD;
	std::unordered_map <int, StatsReal> RMS;
	std::unordered_map <int, std::shared_ptr<Histo>> aux_Histogram;
	std::unordered_map <int, StatsReal> p10;
	std::unordered_map <int, StatsReal> p25;
	std::unordered_map <int, StatsReal> p75;
	std::unordered_map <int, StatsReal> p90;
	std::unordered_map <int, StatsReal> IQR;
	std::unordered_map <int, StatsReal> entropy;
	std::unordered_map <int, StatsReal> mode;
	std::unordered_map <int, StatsReal> uniformity;
	std::unordered_map <int, StatsReal> RMAD;

	// --Timing
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < N; i++)
	{
		int p = i % M;	// position

		pixelCountRoiArea[p] = pixelCountRoiArea[p] + i;
		aux_PrevCount[p] = i;
		aux_PrevIntens[p] = i;
		mean[p] = i;
		//std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
		median[p] = i;
		min[p] = i;
		max[p] = i;
		massEnergy[p] = i;
		variance[p] = i;
		stddev[p] = i;
		centroid_x[p] = i;
		centroid_y[p] = i;
		aux_M2[p] = i;
		aux_M3[p] = i;
		aux_M4[p] = i;
		skewness[p] = i;
		kurtosis[p] = i;
		MAD[p] = i;
		RMS[p] = i;
		//std::unordered_map <int, std::shared_ptr<Histo>> aux_Histogram;
		p10[p] = i;
		p25[p] = i;
		p75[p] = i;
		p90[p] = i;
		IQR[p] = i;
		entropy[p] = i;
		mode[p] = i;
		uniformity[p] = i;
		RMAD[p] = i;
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

		StatsInt, //std::unordered_map <int, StatsInt> pixelCountRoiArea;
		StatsInt, //std::unordered_map <int, StatsInt> aux_PrevCount;
		StatsInt, //std::unordered_map <int, StatsInt> aux_PrevIntens;
		StatsReal, //std::unordered_map <int, StatsReal> mean;
		//---std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
		StatsInt, //std::unordered_map <int, StatsInt> median;
		StatsInt, //std::unordered_map <int, StatsInt> min;
		StatsInt, //std::unordered_map <int, StatsInt> max;
		StatsInt, //std::unordered_map <int, StatsInt> massEnergy;
		StatsReal, //std::unordered_map <int, StatsReal> variance;
		StatsReal, //std::unordered_map <int, StatsReal> stddev;	// Is calculated from 'lavelVariance' in Reduce()
		StatsReal, //std::unordered_map <int, StatsReal> centroid_x;
		StatsReal, //std::unordered_map <int, StatsReal> centroid_y;
		StatsReal, //std::unordered_map <int, StatsReal> aux_M2;
		StatsReal, //std::unordered_map <int, StatsReal> aux_M3;
		StatsReal, //std::unordered_map <int, StatsReal> aux_M4;
		StatsReal, //std::unordered_map <int, StatsReal> skewness;
		StatsReal, //std::unordered_map <int, StatsReal> kurtosis;
		StatsReal, //std::unordered_map <int, StatsReal> MAD;
		StatsReal, //std::unordered_map <int, StatsReal> RMS;
	//---std::unordered_map <int, std::shared_ptr<Histo>> aux_Histogram;
		StatsReal, //std::unordered_map <int, StatsReal> p10;
		StatsReal, //std::unordered_map <int, StatsReal> p25;
		StatsReal, //std::unordered_map <int, StatsReal> p75;
		StatsReal, //std::unordered_map <int, StatsReal> p90;
		StatsReal, //std::unordered_map <int, StatsReal> IQR;
		StatsReal, //std::unordered_map <int, StatsReal> entropy;
		StatsReal, //std::unordered_map <int, StatsReal> mode;
		StatsReal, //std::unordered_map <int, StatsReal> uniformity;
		StatsReal //std::unordered_map <int, StatsReal> RMAD;
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

