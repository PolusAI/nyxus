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

void print_label_stats()
{
	std::cout << "\tFeatures by label. Number of processed labels " << labelMeans.size() << std::endl;

	// Print stats by label
	print_by_label("Min", labelMins);
	print_by_label("Max", labelMaxs);
	print_by_label("Mean", labelMeans);
	print_by_label("Median", labelMedians);
}

