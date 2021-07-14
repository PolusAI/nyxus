#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <stdlib.h>
#include <stdio.h>

#include "sensemaker.h"

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns)
{
	std::stringstream ss;

	std::cout << std::endl << featureName << std::endl;
	
	int i = 1;
	for (auto& x : L)
	{
		ss << 'L' << x.first << ':' << x.second;
		std::cout << std::setw(10) << ss.str();
		ss.str (std::string()); // Clear ss

		if (i++ % numColumns == 0)
			std::cout << std::endl;
	}

}

void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns)
{
	std::stringstream ss;

	std::cout << std::endl << featureName << std::endl;

	int i = 1;
	for (auto& x : L)
	{
		ss << 'L' << x.first << ':' << x.second;
		std::cout << std::setw(30) << ss.str();
		ss.str(std::string()); // Clear ss

		if (i++ % numColumns == 0)
			std::cout << std::endl;
	}

}

void print_label_stats()
{
	std::cout << "\tFeatures by label. Number of processed labels " << uniqueLabels.size() << std::endl;

	// Print stats by label
	/*
	print_by_label("Min", labelMins);
	print_by_label("Max", labelMaxs);
	print_by_label("Mean", labelMeans);
	print_by_label("Median", labelMedians);
	print_by_label("Energy", labelMassEnergy);
	print_by_label("Variance", labelVariance);
	*/
}


bool save_features (std::string inputFpath, std::string outputDir)
{
	// Research
	std::cout << "intensityMin = " << intensityMin << " intensityMax = " << intensityMax << std::endl;
	
	// Sort the labels
	std::vector<int> L { uniqueLabels.begin(), uniqueLabels.end() };
	std::sort (L.begin(), L.end());


	// Tear off the directory part of 'inputPath', we don't need it
	std::string fullPath = outputDir + "/" + getPureFname(inputFpath) + ".csv";

	std::cout << "Creating file " << fullPath << std::endl;

	//fullPath.replace("//", "/");
	//fullPath.replace("\\\\", "\\");

	// Output label data
	// -- Create a file
	FILE* fp = nullptr;
	fopen_s(&fp, fullPath.c_str(), "w");
	if (!fp) 
	{
		std::perror("fopen failed"); 
		return false;
	}

	// -- Configure buffered write
	if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0) {
		std::perror("setvbuf failed"); 
		return false;
	}
	
	// -- Header
	fprintf (fp, "# , label , pixelCount, min , max , range , mean , median , energy , stddev , skewness , kurtosis , mad , rms , weighted_centroid_x , weighted_centroid_y , entropy , P10 , P25 , P75 , P90 , IQR , RMAD , mode , uniformity\n");

	// -- Dump numbers
	int cnt = 1; 
	for (auto l : L)
	{
		std::stringstream ss;
		
		/*
		auto _min = labelMins[l];
		auto _max = labelMaxs[l];
		auto _range = _max - _min;
		auto _mean = labelMeans[l];
		auto _median = labelMedians[l];
		auto _energy = labelMassEnergy [l];
		auto _stdev = labelStddev[l];
		auto _skew = labelSkewness[l];
		auto _kurt = labelKurtosis[l];
		auto _mad = labelMAD[l];
		auto _rms = labelRMS[l];
		auto _wcx = labelCentroid_x[l], 
			_wcy = labelCentroid_y[l];
		auto _entro = labelEntropy[l];
		auto _p10 = labelP10[l], 
			_p25 = labelP25[l], 
			_p75 = labelP75[l], 
			_p90 = labelP90[l], 
			_iqr = labelIQR[l], 
			_rmad = labelRMAD[l],
			_mode = labelMode[l], 
			_unifo = labelUniformity[l];
		*/
		//--- New
		LR& lr = labelData[l];
		auto _pixCnt = lr.labelCount;
		auto _min = lr.labelMins;
		auto _max = lr.labelMaxs;
		auto _range = _max - _min;
		auto _mean = lr.labelMeans;
		auto _median = lr.labelMedians;
		auto _energy = lr.labelMassEnergy;
		auto _stdev = lr.labelStddev;
		auto _skew = lr.labelSkewness;
		auto _kurt = lr.labelKurtosis;
		auto _mad = lr.labelMAD;
		auto _rms = lr.labelRMS;
		auto _wcx = lr.labelCentroid_x,
			_wcy = lr.labelCentroid_y;
		auto _entro = lr.labelEntropy;
		auto _p10 = lr.labelP10,
			_p25 = lr.labelP25,
			_p75 = lr.labelP75,
			_p90 = lr.labelP90,
			_iqr = lr.labelIQR,
			_rmad = lr.labelRMAD,
			_mode = lr.labelMode,
			_unifo = lr.labelUniformity;


		ss << l		<< " , " 
			<< _pixCnt	<< " , "
			<< _min		<< " , " 
			<< _max		<< " , " 
			<< _range	<< " , "
			<< _mean	<< " , " 
			<< _median	<< " , " 
			<< _energy	<< " , " 
			<< _stdev	<< " , "
			<< _skew	<< " , "
			<< _kurt	<< " , "
			<< _mad		<< " , "
			<< _rms		<< " , "
			<< _wcx		<< " , "
			<< _wcy		<< " , "
			<< _entro	<< " , "
			<< _p10		<< " , "
			<< _p25		<< " , "
			<< _p75		<< " , " 
			<< _p90		<< " , "
			<< _iqr		<< " , "
			<< _rmad	<< " , "
			<< _mode	<< " , "
			<< _unifo;
		fprintf (fp, "%s\n", ss.str().c_str());
	}
	std::fclose(fp);

	return true;
}

