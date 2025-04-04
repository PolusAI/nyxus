#ifdef WITH_PYTHON_H

#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "dirs_and_files.h"
#include "environment.h"
#include "globals.h"
#include "raw_image_loader.h"
#include "features/contour.h"
#include "features/erosion.h"
#include "features/gabor.h"
#include "features/2d_geomoments.h"
#include "helpers/fsystem.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"

#ifdef USE_GPU
	#include "gpucache.h"
#endif

namespace Nyxus
{

	bool processIntSegImagePairInMemory (const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label, int filepair_index, const std::string& intens_name, const std::string& seg_name, std::vector<int> unprocessed_rois)
	{
		std::vector<int> trivRoiLabels;

		// Phase 1: gather ROI metrics

		if (! gatherRoisMetricsInMemory(intens, label, filepair_index))	// Output - set of ROI labels, label-ROI cache mappings
			return false;

		// ROI metrics are gathered, let's publish them non-anisotropically into ROI's AABB
		// (Montage does not support anisotropy by design leaving it to Python user's control.)
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];
			r.make_nonanisotropic_aabb();
		}

		// Allocate each ROI's feature value buffer
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];

			r.intFname = intens_name;
			r.segFname = seg_name;

			r.initialize_fvals();
		}

		// Distribute ROIs among phases
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];
			if (size_t roiFootprint = r.get_ram_footprint_estimate(), 
				ramLim = theEnvironment.get_ram_limit(); 
				roiFootprint >= ramLim)
			{
				unprocessed_rois.push_back(lab);
			}
			else
				trivRoiLabels.push_back(lab);
		}

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			processTrivialRoisInMemory (trivRoiLabels, intens, label, filepair_index, theEnvironment.get_ram_limit());
		}

		// Phase 3: skip nontrivial ROIs

		return true;
	}
	
	int processMontage(
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intensity_images,
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images,
		int numReduceThreads,
		const std::vector<std::string>& intensity_names,
		const std::vector<std::string>& seg_names,
		std::string& error_message,
		const SaveOption saveOption,
		const std::string& outputPath)
	{	
		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		if (write_apache) 
		{
			theEnvironment.arrow_stream = ArrowOutputStream();
			auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file (saveOption, get_arrow_filename(outputPath, theEnvironment.nyxus_result_fname, saveOption), Nyxus::get_header(theFeatureSet.getEnabledFeatures()));
			if (!status) 
			{
				std::cout << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		auto intens_buffer = intensity_images.request();
		auto label_buffer = label_images.request();

		auto width = intens_buffer.shape[1];
		auto height = intens_buffer.shape[2];

		auto nf = intens_buffer.shape[0];
		
		for (int i = 0; i < nf; i++)
		{
			// Clear ROI label list, ROI data, etc.
			clear_slide_rois();

			auto image_idx = i * width * height;	// image offset in memory

			std::vector<int> unprocessed_rois;

			if (! processIntSegImagePairInMemory (intensity_images, label_images, image_idx, intensity_names[i], seg_names[i], unprocessed_rois))
			{
				error_message = "processIntSegImagePairInMemory() returned an error code while processing file pair";
				return 1;
			}

			if (write_apache) {
			
				auto [status, msg] = theEnvironment.arrow_stream.write_arrow_file(Nyxus::get_feature_values());
				if (!status) {
					std::cout << "Error writing Arrow file: " << msg.value() << std::endl;
					return 2;
				}

			} 
      else 
			{
				if (!save_features_2_buffer(theResultsCache))
				{
					error_message = "save_features_2_buffer() failed";
					return 2;
				}

			}

			if (unprocessed_rois.size() > 0) 
			{
				error_message = "The following ROIS are oversized and cannot be processed: ";
				for (const auto& roi: unprocessed_rois)
				{
					error_message += roi;
					error_message += ", ";
				}
				
				// remove trailing space and comma
				error_message.pop_back();
				error_message.pop_back();
			}

			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
                		throw pybind11::error_already_set();
		}

		if (write_apache) 
		{
			// close arrow file after use
			auto [status, msg] = theEnvironment.arrow_stream.close_arrow_file();
			if (!status) 
			{
				std::cout << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}
		return 0; // success
	}

}

#endif
