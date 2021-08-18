#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
			std::vector <std::string> intensFiles, labelFiles;
			readDirectoryFiles(intensity_path, intensFiles);
			readDirectoryFiles(label_path, labelFiles);

			// One-time initialization
			init_feature_buffers();

			// Process the image sdata
			int errorCode = ingestDataset(
				intensFiles, 
				labelFiles, 
				2, // of FastLoader threads 
				1, // # feature scanner threads
				100, // min_online_roi_size
				true, ""	// csv output directory
			);

			// Allocate and initialize some data; make this big so
			// we can see the impact on the process memory use:
			size_t size = uniqueLabels.size() * numFeaturesCalculated;
			double* foo = new double[size];
			for (size_t i = 0; i < size; i++) {
				foo[i] = (double)i;
			}

			// Create a Python object that will free the allocated
			// memory when destroyed:
			py::capsule free_when_done(foo, [](void* f) {
				double* foo = reinterpret_cast<double*>(f);
				std::cerr << "Element [0] = " << foo[0] << "\n";
				std::cerr << "freeing memory @ " << f << "\n";
				delete[] foo;
				});

			return py::array_t<double>(
				{ uniqueLabels.size(), numFeaturesCalculated }, // shape
				{ uniqueLabels.size() * 8 * 8, numFeaturesCalculated * 8 }, // C-style contiguous strides for double
				foo, // the data pointer
				free_when_done); // numpy array references this parent
			});
}


