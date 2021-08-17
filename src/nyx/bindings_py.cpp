#include <pybind11/pybind11.h>

#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "sensemaker.h"


#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif


namespace py = pybind11;

PYBIND11_MODULE(backend, m)
{
	m.def("calc_pixel_intensity_stats", [] (const std::string &label_path, const std::string &intensity_path) 
		{
				std::cout << "calc_pixel_intensity_stats (" << label_path << "," << intensity_path << std::endl;
				
				//... actual feature calculation ...
				
				return std::make_tuple(
					0, //Mean
					1, //Median
					2, //Min
					3, //Max
					4, //Range
					5, //Standard Deviation
					6, //Skewness
					7, //Kurtosis
					8, //Mean Absolute Deviation
					9, //Energy
					10, //Root Mean Squared
					11, //Entropy
					12, //Mode
					13, //Uniformity
					14, //10th Percentile
					15, //25th Percentile
					16, //75th Percentile
					17, //90th Percentile
					18, //Interquartile Range
					19, //Robust Mean Absolute Deviation
					20, //Weighted Centroid in y direction
					21 //Weighted Centroid in x direction
					);
		}
	);

}


