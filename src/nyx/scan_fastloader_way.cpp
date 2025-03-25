//
// This file is a collection of drivers of tiled TIFF file scanning from the FastLoader side
//
#include "helpers/fsystem.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <map>
#include <array>
#include <regex>
#include <string>
#include <limits>

#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
	#include <pybind11/stl.h>
	#include <pybind11/numpy.h>
	namespace py = pybind11;
#endif

#include "dirs_and_files.h"
#include "environment.h"
#include "features/contour.h"
#include "features/erosion.h"
#include "features/gabor.h"
#include "features/2d_geomoments.h"
#include "globals.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"

#include "raw_image_loader.h"

#ifdef USE_GPU
	#include "gpucache.h"
#endif

namespace Nyxus
{
	std::string get_arrow_filename(const std::string& output_path, const std::string& default_filename, const SaveOption& arrow_file_type){

	/*
				output_path			condition			verdict
	Case 1: 	/foo/bar		exist in fs				is a directory, append default filename with proper ext
				/foo/bar/		or ends with / or \
				\foo\bar\			

	Case 2:		/foo/bar		does not exist in fs	assume the extension is missing, append proper ext
								but /foo exists

	Case 3: 	/foo/bar		neither /foo nor 		treat as directory, append default filename with proper ext
								/foo/bar exists in fs
	
	Case 4: 	/foo/bar.ext	exists in fs and is a 	append default filename with proper ext
								directory	
			
	Case 5: 	/foo/bar.ext	does not exist in fs  	this is a file, check if ext is correct and modify if needed

	Case 6:		empty									default filename with proper ext
								

	*/
		std::string valid_ext = [&arrow_file_type](){
			if (arrow_file_type == Nyxus::SaveOption::saveArrowIPC) {
				return ".arrow";
			} else if (arrow_file_type == Nyxus::SaveOption::saveParquet) {
				return ".parquet";
			} else {return "";}
		}();
		
		if (output_path != ""){
			auto arrow_path = fs::path(output_path);
			if (fs::is_directory(arrow_path) // case 1, 4
			    || Nyxus::ends_with_substr(output_path, "/") 
				|| Nyxus::ends_with_substr(output_path, "\\")){
				arrow_path = arrow_path/default_filename;
			} else if(!arrow_path.has_extension()) { 
				if(!fs::is_directory(arrow_path.parent_path())){ // case 3
					arrow_path = arrow_path/default_filename;
				}
				// else case 2, do nothing here	
			}
			// case 5 here, but also for 1-4, update extenstion here
			arrow_path.replace_extension(valid_ext);
			return arrow_path.string();
		} else { // case 6
			return default_filename + valid_ext;
		}  
	}

#ifdef WITH_PYTHON_H
	bool processIntSegImagePairInMemory (const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label, int filepair_index, const std::string& intens_name, const std::string& seg_name, std::vector<int> unprocessed_rois)
	{
		std::vector<int> trivRoiLabels;

		// Phase 1: gather ROI metrics
		bool okGather = gatherRoisMetricsInMemory(intens, label, filepair_index);	// Output - set of ROI labels, label-ROI cache mappings
		if (!okGather)
			return false;

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
#endif

#ifdef WITH_PYTHON_H
	
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

		if (write_apache) {

			theEnvironment.arrow_stream = ArrowOutputStream();
			auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file(saveOption, get_arrow_filename(outputPath, theEnvironment.nyxus_result_fname, saveOption), Nyxus::get_header(theFeatureSet.getEnabledFeatures()));
			if (!status) {
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
			auto ok = processIntSegImagePairInMemory (intensity_images, label_images, image_idx, intensity_names[i], seg_names[i], unprocessed_rois);		// Phased processing
			if (ok == false)
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
			} else {

				ok = save_features_2_buffer(theResultsCache);

				if (ok == false)
				{
					error_message = "save_features_2_buffer() failed";
					return 2;
				}

			}

			if (unprocessed_rois.size() > 0) {
				error_message = "The following ROIS are oversized and cannot be processed: ";
				for (const auto& roi: unprocessed_rois){
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


		if (write_apache) {
			// close arrow file after use
			auto [status, msg] = theEnvironment.arrow_stream.close_arrow_file();
			if (!status) {
				std::cout << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}
		return 0; // success
	}
#endif

	void dump_roi_metrics(const std::string & label_fpath)
	{
		// are we amidst a 3D scenario ?
		bool dim3 = theEnvironment.dim();

		// prepare the file name
		fs::path pseg (label_fpath);
		std::string fpath = theEnvironment.output_dir + "/roi_metrics_" + pseg.stem().string() + ".csv";

		// fix the special 3D file name character if needed
		if (dim3)
			for (auto& ch : fpath)
				if (ch == '*')
					ch = '~';

		std::cout << "Dumping ROI metrics to " << fpath << '\n';

		std::ofstream f (fpath);
		if (f.fail())
		{
			std::cerr << "Error: cannot create file " << fpath << '\n';
			return;
		}

		// header
		f << "label, area, minx, miny, maxx, maxy, width, height, min_intens, max_intens, size_bytes, size_class \n";
		// sort labels
		std::vector<int>  sortedLabs { uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(sortedLabs.begin(), sortedLabs.end());
		// body
		for (auto lab : sortedLabs)
		{
			LR& r = roiData[lab];
			auto szb = r.get_ram_footprint_estimate();
			std::string ovsz = szb < theEnvironment.get_ram_limit() ? "TRIVIAL" : "OVERSIZE";
			f << lab << ", "
				<< r.aux_area << ", "
				<< r.aabb.get_xmin() << ", "
				<< r.aabb.get_ymin() << ", "
				<< r.aabb.get_xmax() << ", "
				<< r.aabb.get_ymax() << ", "
				<< r.aabb.get_width() << ", "
				<< r.aabb.get_height() << ", "
				<< r.aux_min << ", "
				<< r.aux_max << ", "
				<< szb << ", "
				<< ovsz << ", ";
			f << "\n";
		}

		f.flush();
	}

	void dump_roi_pixels (const std::vector<int> & batch_labels, const std::string & label_fpath)
	{
		// no data ?
		if (batch_labels.size() == 0)
		{
			std::cerr << "Error: no ROI pixel data for file " << label_fpath << '\n';
			return;
		}

		// sort labels for reader's comfort
		std::vector<int>  srt_L{ batch_labels.begin(), batch_labels.end() };
		std::sort(srt_L.begin(), srt_L.end());

		// are we amidst a 3D scenario ?
		bool dim3 = theEnvironment.dim();

		// prepare the file name
		fs::path pseg (label_fpath);
		std::string fpath = theEnvironment.output_dir + "/roi_pixels_" + pseg.stem().string() + "_batch" + std::to_string(srt_L[0]) + '-' + std::to_string(srt_L[srt_L.size() - 1]) + ".csv";
		
		// fix the special 3D file name character if needed
		if (dim3)
			for (auto& ch : fpath)
				if (ch == '*')
					ch = '~';

		std::cout << "Dumping ROI pixels to " << fpath << '\n';
		
		std::ofstream f(fpath);
		if (f.fail())
		{
			std::cerr << "Error: cannot create file " << fpath << '\n';
			return;
		}

		// header
		f << "label,x,y,z,intensity, \n";

		// body
		for (auto lab : srt_L)
		{
			LR& r = roiData [lab];
			if (dim3)
				for (auto & plane : r.zplanes)
					for (auto idx : plane.second)
					{
						auto& pxl = r.raw_pixels_3D[idx];
						f << lab << "," << pxl.x << ',' << pxl.y << ',' << pxl.z << ',' << pxl.inten << ',' << '\n';

					}
			else
				for (auto pxl : r.raw_pixels)
					f << lab << "," << pxl.x << ',' << pxl.y << ',' << pxl.inten << ',' << '\n';
		}

		f.flush();
	}

}