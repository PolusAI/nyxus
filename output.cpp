#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "sensemaker.h"

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
	std::cout << "\tFeatures by label. Number of processed labels " << labelMeans.size() << std::endl;

	// Print stats by label
	print_by_label("Min", labelMins);
	print_by_label("Max", labelMaxs);
	print_by_label("Mean", labelMeans);
	print_by_label("Median", labelMedians);
	print_by_label("Energy", labelMassEnergy);
	print_by_label("Variance", labelVariance);
}


bool save_features (std::string inputFpath, std::string outputDir)
{
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
	fprintf (fp, "# , label , min , max , range , mean , median , energy , stddev , skewness , kurtosis , mad , weighted_centroid_x , weighted_centroid_y\n");

	// -- Dump numbers
	int cnt = 1; 
	for (auto lab : L)
	{
		std::stringstream ss;

		auto _min = labelMins[lab];
		auto _max = labelMaxs[lab];
		auto _range = _max - _min;
		auto _mean = labelMeans[lab];
		auto _median = labelMedians[lab];
		auto _energy = labelMassEnergy [lab];
		auto _stdev = sqrt (labelVariance[lab]);
		auto _skew = labelSkewness[lab];
		auto _kurt = labelKurtosis[lab];
		auto _mad = labelMAD[lab];
		auto _wcx = labelCentroid_x[lab], 
			_wcy = labelCentroid_y[lab];

		ss << lab		<< " , " 
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
			<< _wcx		<< " , "
			<< _wcy 
			<< std::endl;
		fprintf (fp, "%s", ss.str().c_str());
	}
	std::fclose(fp);

	return true;
}

