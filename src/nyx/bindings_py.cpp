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

			// Process the image data
			int errorCode = ingestDataset(
				intensFiles, 
				labelFiles, 
				2, // of FastLoader threads 
				1, // # feature scanner threads
				100, // min_online_roi_size
				true, ""	// csv output directory
			);

			// Allocate and initialize the return data buffer - [a matrix n_labels X n_features]:
			// (Background knowledge - https://stackoverflow.com/questions/54876346/pybind11-and-stdvector-how-to-free-data-using-capsules)
			size_t ny = uniqueLabels.size(),
				nx = numFeaturesCalculated,
				len = ny * nx;
			double* retbuf = new double[len];
			for (size_t i = 0; i < len; i++) 
				retbuf[i] = 0.0;

			// Create a Python object that will free the allocated
			// memory when destroyed:
			py::capsule free_when_done (retbuf, [](void* f) {
				double* foo = reinterpret_cast<double*>(f);
				std::cerr << "Element [0] = " << foo[0] << "\n";
				std::cerr << "freeing memory @ " << f << "\n";
				delete[] foo;
				});

			return py::array_t<double> (
				{ ny, nx }, // shape
				{ ny * nx * 8, nx * 8 }, // C-style contiguous strides for double
				retbuf, // the data pointer
				free_when_done); // numpy array references this parent
			});
}


