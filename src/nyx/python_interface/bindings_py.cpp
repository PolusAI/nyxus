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
#include "../environment.h"
#include "../globals.h"

#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif

//
// Returns:
// [0] - error code (0 = success, 1... = specific error)
// [1] - error details string
// [2] - number of features calculated (nx)
// [3] - number of unique ROI labels (ny)
// [4] - data buffer [nx X ny] requiring further deallocation
//
std::tuple<int, std::string, size_t, size_t, double*> featureSetInvoker(std::initializer_list<AvailableFeatures>& desiredFeatures, const std::string& label_path, const std::string& intensity_path)
{
	double* retbuf = nullptr;

	//==== Calculate features
	theFeatureSet.enableAll (false);
	theFeatureSet.enableFeatures (desiredFeatures); 

	// Try to reach data files at directories 'label_path' and 'intensity_path'
	std::vector <std::string> intensFiles, labelFiles;
	int errorCode = Nyxus::read_dataset (intensity_path, label_path, "outputPath", "", "", false, intensFiles, labelFiles);

	// Check for error
	if (errorCode)
		return { 1, "Dataset structure error", 0, 0, nullptr };

	// One-time initialization
	Nyxus::init_feature_buffers();

	// Process the image sdata. Upon return, global buffer 'calcResultBuf' will be filled with result data
	errorCode = Nyxus::processDataset(
		intensFiles,
		labelFiles,
		Nyxus::theEnvironment.n_loader_threads,
		Nyxus::theEnvironment.n_pixel_scan_threads,
		Nyxus::theEnvironment.n_reduce_threads,
		100,	// min_online_roi_size
		false, 
		"unused_dirOut");

	// Check for error
	if (errorCode)
		return { 2, "Error while calculating features", 0, 0, nullptr};

	//==== Allocate the low-level return data buffer
	// 
	// Allocate and initialize the return data buffer - [a matrix n_labels X n_features]:
	// (Background knowledge - https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11 and https://stackoverflow.com/questions/54876346/pybind11-and-stdvector-how-to-free-data-using-capsules)
	size_t ny = Nyxus::uniqueLabels.size(),
		nx = theFeatureSet.numOfEnabled(),
		len = ny * nx;

	// Check for error
	if (ny == 0)
		return { 3, "No unique labels were processed", 0, 0, nullptr };
	if (nx == 0)
		return { 4, "No features were calculated", 0, 0, nullptr };

	//DEBUG diagnostic output:
	std::cout << "Result shape: ny=uniqueLabels.size()=" << ny << " X nx=" << theFeatureSet.numOfEnabled() << " = " << len << ", element[0]=" << Nyxus::calcResultBuf[0] << std::endl;

	// Check for error: calcResultBuf is expected to have exavtly 'len' elements
	if (len != Nyxus::calcResultBuf.size())
	{
		std::stringstream ss;
		ss << "ERROR: Result shape [ny=uniqueLabels.size()=" << ny << " X nx=" << theFeatureSet.numOfEnabled() << " = " << len << "] mismatches with the result buffer size " << Nyxus::calcResultBuf.size() << " in " << __FILE__ << ":" << __LINE__;
		return { 5, ss.str(), 0, 0, nullptr };
	}

	// Allocate 
	retbuf = new double[len];
	
	// Check for error
	if (retbuf == nullptr)
	{
		std::stringstream ss;
		ss << "ERROR: Cannot allocate the return data buffer in " << __FILE__ << ":" << __LINE__ << std::endl;
		return { 6, ss.str(), 0, 0, nullptr };
	}

	// calcResultBuf is expected to have exavtly 'len' elements
	for (size_t i = 0; i < len; i++)
		retbuf[i] = Nyxus::calcResultBuf[i];

	// Success, return the result
	return { 0, "", nx, ny, retbuf };
}

namespace py = pybind11;

PYBIND11_MODULE(nyx_backend, m)
{
	m.def("backend_is_alive_imp", [](const std::string& label_path, const std::string& intensity_path)
		{
			//==== Calculate features
			// 
			// Request the features that we want to calculate
			theFeatureSet.enableBoundingBox();

			// Try to reach data files at directories 'label_path' and 'intensity_path'
			std::vector <std::string> intensFiles, labelFiles;
			int errorCode = Nyxus::read_dataset (intensity_path, label_path, "outputPath", "", "", false, intensFiles, labelFiles);
			if (errorCode)
			{
				std::cout << std::endl << "Dataset structure error" << std::endl;
				//return 1;
			}

			//==== Mock returned results
			// 
			// Allocate and initialize the return data buffer - [a matrix n_labels X n_features]:
			// (Background knowledge - https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11 and https://stackoverflow.com/questions/54876346/pybind11-and-stdvector-how-to-free-data-using-capsules)
			size_t ny = 4,	// # unique ROI
				nx = 3,		// # features
				len = ny * nx;
			Nyxus::calcResultBuf.clear();
			for (int i = 0; i < len; i++)
				Nyxus::calcResultBuf.push_back (i + 1);	// +1 is a seed

			//DEBUG diagnostic output:
			std::cout << "Result shape: ny=uniqueLabels.size()=" << ny << " X nx=" << nx << " = " << len << ", element[0]=" << Nyxus::calcResultBuf[0] << std::endl;

			// calcResultBuf is expected to have exavtly 'len' elements
			if (len != Nyxus::calcResultBuf.size())
				std::cerr << "ERROR: Result shape [ny=uniqueLabels.size()=" << ny << " X nx=" << theFeatureSet.numOfEnabled() << " = " << len << "] mismatches with the result buffer size " << Nyxus::calcResultBuf.size() << " in " << __FILE__ << ":" << __LINE__ << std::endl;

			double* retbuf = new double[len];
			if (retbuf == nullptr)
				std::cerr << "ERROR: Cannot allocate the return data buffer in " << __FILE__ << ":" << __LINE__ << std::endl;

			// calcResultBuf is expected to have exavtly 'len' elements
			for (size_t i = 0; i < len; i++)
				retbuf[i] = Nyxus::calcResultBuf[i];

			// Create a Python object that will free the allocated
			// memory when destroyed:
			py::capsule free_when_done(retbuf, [](void* f) {
				double* foo = reinterpret_cast<double*>(f);
				std::cerr << "Element [0] = " << foo[0] << "\n";
				std::cerr << "freeing memory @ " << f << "\n";
				delete[] foo;
				});

			return py::array_t<double>(
				{ ny, nx }, // shape
				{ nx*sizeof(double), sizeof(double)}, // C-style contiguous strides for double
				retbuf, // the data pointer
				free_when_done); // numpy array references this parent
		});

		m.def("calc_pixel_intensity_stats", [](const std::string& label_path, const std::string& intensity_path)
			{
				// Calculate features
				auto desiredFeatures = {
					MEAN,
					MEDIAN,
					MIN,
					MAX,
					RANGE,
					STANDARD_DEVIATION,
					SKEWNESS,
					KURTOSIS,
					MEAN_ABSOLUTE_DEVIATION,
					ENERGY,
					ROOT_MEAN_SQUARED,
					ENTROPY,
					MODE,
					UNIFORMITY,
					P10, P25, P75, P90,
					INTERQUARTILE_RANGE,
					ROBUST_MEAN_ABSOLUTE_DEVIATION,
					WEIGHTED_CENTROID_Y,
					WEIGHTED_CENTROID_X };
				auto [errorCode, errorDetails, nx, ny, retbuf] = featureSetInvoker(desiredFeatures, label_path, intensity_path);

				// Check for errors
				if (errorCode)
				{
					std::cerr << "featureSetInvoker failed with error " << errorCode << "\n";
					PyErr_SetString (PyExc_RuntimeError, errorDetails.c_str());
				}

				// Create a Python object that will free the allocated memory when destroyed:
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

	m.def("calc_bounding_box", [](const std::string& label_path, const std::string& intensity_path)
		{
			// Calculate features
			auto desiredFeatures = { BBOX_YMIN, BBOX_XMIN, BBOX_HEIGHT, BBOX_WIDTH };
			auto [errorCode, errorDetails, nx, ny, retbuf] = featureSetInvoker(desiredFeatures, label_path, intensity_path);

			// Check for errors
			if (errorCode)
			{
				std::cerr << "featureSetInvoker failed with error " << errorCode << "\n";
				PyErr_SetString(PyExc_RuntimeError, errorDetails.c_str());
			}

			// Create a Python object that will free the allocated
			// memory when destroyed:
			py::capsule free_when_done(retbuf, [](void* f) {
				double* foo = reinterpret_cast<double*>(f);
				std::cerr << "Element [0] = " << foo[0] << "\n";
				std::cerr << "freeing memory @ " << f << "\n";
				delete[] foo;
				});

			return py::array_t<double>(
				{ ny, nx }, // shape
				{ ny * nx * 8, nx * 8 }, // C-style contiguous strides for double
				retbuf, // the data pointer
				free_when_done); // numpy array references this parent
		});

	m.def("calc_feret", [](const std::string& label_path, const std::string& intensity_path)
		{
			// Calculate features
			auto desiredFeatures = { 
				MIN_FERET_DIAMETER, 
				MAX_FERET_DIAMETER, 
				MIN_FERET_ANGLE, 
				MAX_FERET_ANGLE, 
				STAT_FERET_DIAM_MIN, 
				STAT_FERET_DIAM_MAX, 
				STAT_FERET_DIAM_MEAN, 
				STAT_FERET_DIAM_MEDIAN, 
				STAT_FERET_DIAM_STDDEV, 
				STAT_FERET_DIAM_MODE };
			auto [errorCode, errorDetails, nx, ny, retbuf] = featureSetInvoker(desiredFeatures, label_path, intensity_path);

			// Check for errors
			if (errorCode)
			{
				std::cerr << "featureSetInvoker failed with error " << errorCode << "\n";
				PyErr_SetString(PyExc_RuntimeError, errorDetails.c_str());
			}

			// Create a Python object that will free the allocated
			// memory when destroyed:
			py::capsule free_when_done(retbuf, [](void* f) {
				double* foo = reinterpret_cast<double*>(f);
				std::cerr << "Element [0] = " << foo[0] << "\n";
				std::cerr << "freeing memory @ " << f << "\n";
				delete[] foo;
				});

			return py::array_t<double>(
				{ ny, nx }, // shape
				{ ny * nx * 8, nx * 8 }, // C-style contiguous strides for double
				retbuf, // the data pointer
				free_when_done); // numpy array references this parent
		});

	m.def("calc_glrlm", [](const std::string& label_path, const std::string& intensity_path)
		{
			// Calculate features
			auto desiredFeatures = {
				GLRLM_SRE,
				GLRLM_LRE,
				GLRLM_GLN,
				GLRLM_GLNN,
				GLRLM_RLN,
				GLRLM_RLNN,
				GLRLM_RP,
				GLRLM_GLV,
				GLRLM_RV,
				GLRLM_RE,
				GLRLM_LGLRE,
				GLRLM_HGLRE,
				GLRLM_SRLGLE,
				GLRLM_SRHGLE,
				GLRLM_LRLGLE,
				GLRLM_LRHGLE };
			auto [errorCode, errorDetails, nx, ny, retbuf] = featureSetInvoker(desiredFeatures, label_path, intensity_path);

			// Check for errors
			if (errorCode)
			{
				std::cerr << "featureSetInvoker failed with error " << errorCode << "\n";
				PyErr_SetString(PyExc_RuntimeError, errorDetails.c_str());
			}

			// Create a Python object that will free the allocated
			// memory when destroyed:
			py::capsule free_when_done(retbuf, [](void* f) {
				double* foo = reinterpret_cast<double*>(f);
				std::cerr << "Element [0] = " << foo[0] << "\n";
				std::cerr << "freeing memory @ " << f << "\n";
				delete[] foo;
				});

			return py::array_t<double>(
				{ ny, nx }, // shape
				{ ny * nx * 8, nx * 8 }, // C-style contiguous strides for double
				retbuf, // the data pointer
				free_when_done); // numpy array references this parent
		});

}


