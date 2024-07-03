//
// This file is a collection of drivers of tiled TIFF file scanning from the FastLoader side
//
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <map>
#include <array>
#include <regex>
#include <string>

#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif
#include "dirs_and_files.h"
#include "environment.h"
#include "globals.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"

// Sanity
#ifdef _WIN32
#include<windows.h>
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

	bool processIntSegImagePair (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

		// Timing block (image scanning)
		{
			//______	STOPWATCH("Image scan/ImgScan/Scan/lightsteelblue", "\t=");

			{ STOPWATCH("Image scan1/ImgScan1/Scan1/lightsteelblue", "\t=");

				VERBOSLVL2(
					// Report the amount of free RAM
					unsigned long long freeRamAmt = Nyxus::getAvailPhysMemory();
					static unsigned long long initial_freeRamAmt = 0;
					if (initial_freeRamAmt == 0)
						initial_freeRamAmt = freeRamAmt;
					unsigned long long memDiff = 0;
					char sgn;
					if (freeRamAmt > initial_freeRamAmt)
					{
						memDiff = freeRamAmt - initial_freeRamAmt;
						sgn = '+';
					}
					else // system memory can be freed by other processes
					{
						memDiff = initial_freeRamAmt - freeRamAmt;
						sgn = '-';
					}
					std::cout << std::setw(15) << freeRamAmt << " b free (" << sgn << memDiff << ") ";
					)
				// Display (1) dataset progress info and (2) file pair info
				int digits = 2, k = std::pow(10.f, digits);
				float perCent = float(filepair_index) * 100. * k / float(tot_num_filepairs) / float(k);
				VERBOSLVL1 (std::cout << "[ " << filepair_index << " = " << std::setw(digits + 2) << perCent << "% ]\t" << intens_fpath << "\n")
				VERBOSLVL2 (std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n")
			}

			{ STOPWATCH("Image scan2a/ImgScan2a/Scan2a/lightsteelblue", "\t=");

				// Phase 1: gather ROI metrics
				VERBOSLVL2(std::cout << "Gathering ROI metrics\n");
				bool okGather = gatherRoisMetrics(intens_fpath, label_fpath, num_FL_threads);	// Output - set of ROI labels, label-ROI cache mappings
				if (!okGather)
				{
					std::string msg = "Error in gatherRoisMetrics()\n";
					std::cerr << msg;
					throw (std::runtime_error(msg));
					return false;
				}

				// Any ROIs in this slide? (Such slides may exist, it's normal.)
				if (uniqueLabels.size() == 0)
				{
					return true;
				}
			}

			{ STOPWATCH("Image scan2b/ImgScan2b/Scan2b/lightsteelblue", "\t=");

				// Allocate each ROI's feature value buffer
				for (auto lab : uniqueLabels)
				{
					LR& r = roiData[lab];
					r.initialize_fvals();
				}

				#ifndef WITH_PYTHON_H
				// Dump ROI metrics to the output directory
				VERBOSLVL2(dump_roi_metrics(label_fpath))
				#endif		
			}

			{ STOPWATCH("Image scan3/ImgScan3/Scan3/lightsteelblue", "\t=");

				// Support of ROI blacklist
				fs::path fp(theSegFname);
				std::string shortSegFname = fp.stem().string() + fp.extension().string();

				// Distribute ROIs among phases
				for (auto lab : uniqueLabels)
				{
					LR& r = roiData[lab];					
					
					// Skip blacklisted ROI
					if (theEnvironment.roi_is_blacklisted(shortSegFname, lab))
					{
						r.blacklisted = true;
						VERBOSLVL2(std::cout << "Skipping blacklisted ROI " << lab << " for mask " << shortSegFname << "\n");
						continue;
					}

					// Examine ROI's memory footprint
					if (size_t roiFootprint = r.get_ram_footprint_estimate(), 
						ramLim = theEnvironment.get_ram_limit(); 
						roiFootprint >= ramLim)
					{
						VERBOSLVL2(
							std::cout << "oversized ROI " << lab 
								<< " (S=" << r.aux_area 
								<< " W=" << r.aabb.get_width() 
								<< " H=" << r.aabb.get_height() 
								<< " px footprint=" << roiFootprint << " b"								
								<< ")\n"
						);
						nontrivRoiLabels.push_back(lab);
					}
					else
						trivRoiLabels.push_back(lab);
				}
			}
		}

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			VERBOSLVL2(std::cout << "Processing trivial ROIs\n";)
			processTrivialRois (trivRoiLabels, intens_fpath, label_fpath, num_FL_threads, theEnvironment.get_ram_limit());
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRoiLabels.size())
		{
			VERBOSLVL2(std::cout << "Processing oversized ROIs\n";)
			processNontrivialRois (nontrivRoiLabels, intens_fpath, label_fpath, num_FL_threads);
		}

		return true;
	}

	bool processIntSegImagePair_3D (const std::string& intens_fpath, const std::string& label_fpath, int filepair_index, int tot_num_filepairs, const std::vector<std::string> & z_indices)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

			{ STOPWATCH("Image scan1/ImgScan1/Scan1/lightsteelblue", "\t=");

				// Report the amount of free RAM
				unsigned long long freeRamAmt = Nyxus::getAvailPhysMemory();
				static unsigned long long initial_freeRamAmt = 0;
				if (initial_freeRamAmt == 0)
					initial_freeRamAmt = freeRamAmt;
				double memDiff = double(freeRamAmt) - double(initial_freeRamAmt);
				VERBOSLVL1(std::cout << std::setw(15) << freeRamAmt << " bytes free (" << "consumed=" << memDiff << ") ")

				// Display (1) dataset progress info and (2) file pair info
				int digits = 2, k = std::pow(10.f, digits);
				float perCent = float(filepair_index * 100 * k / tot_num_filepairs) / float(k);
				VERBOSLVL1(std::cout << "[ " << std::setw(digits + 2) << perCent << "% ]\t" << " INT: " << intens_fpath << " SEG: " << label_fpath << "\n")
			}

			{ STOPWATCH("Image scan2a/ImgScan2a/Scan2a/lightsteelblue", "\t=");
				// Phase 1: gather ROI metrics
				VERBOSLVL2(std::cout << "Gathering ROI metrics\n");
				bool okGather = gatherRoisMetrics_3D (intens_fpath, label_fpath, z_indices);
				if (!okGather)
				{
					std::string msg = "Error in gatherRoisMetrics()\n";
					std::cerr << msg;
					throw (std::runtime_error(msg));
					return false;
				}
			}

			{ STOPWATCH("Image scan2b/ImgScan2b/Scan2b/lightsteelblue", "\t=");

				// Allocate each ROI's feature value buffer
				for (auto lab : uniqueLabels)
				{
					LR& r = roiData[lab];
					r.initialize_fvals();
				}

#ifndef WITH_PYTHON_H
				// Dump ROI metrics to the output directory
				VERBOSLVL2(dump_roi_metrics(label_fpath))
#endif		
			}

			{ STOPWATCH("Image scan3/ImgScan3/Scan3/lightsteelblue", "\t=");

			// Support of ROI blacklist
			fs::path fp (label_fpath);
			std::string shortSegFname = fp.stem().string() + fp.extension().string();

			// Distribute ROIs among phases
			for (auto lab : uniqueLabels)
			{
				LR& r = roiData[lab];

				// Skip blacklisted ROI
				if (theEnvironment.roi_is_blacklisted(shortSegFname, lab))
				{
					r.blacklisted = true;
					VERBOSLVL2(std::cout << "Skipping blacklisted ROI " << lab << " for mask " << shortSegFname << "\n");
					continue;
				}

				// Examine ROI's memory footprint
				if (size_t roiFootprint = r.get_ram_footprint_estimate_3D(),
					ramLim = theEnvironment.get_ram_limit();
					roiFootprint >= ramLim)
				{
					VERBOSLVL2(
						std::cout << "oversized ROI " << lab
						<< " (S=" << r.aux_area
						<< " W=" << r.aabb.get_width()
						<< " H=" << r.aabb.get_height()
						<< " px footprint=" << roiFootprint << " b"
						<< ")\n"
					);
					nontrivRoiLabels.push_back(lab);
				}
				else
					trivRoiLabels.push_back(lab);
			}
			}

		// Phase 2: process trivial-sized ROIs
		if (trivRoiLabels.size())
		{
			VERBOSLVL2(std::cout << "Processing trivial ROIs\n";)
			processTrivialRois_3D (trivRoiLabels, intens_fpath, label_fpath, theEnvironment.get_ram_limit(), z_indices);
		}

		// Phase 3: process nontrivial (oversized) ROIs, if any
		if (nontrivRoiLabels.size())
		{
			VERBOSLVL2(std::cout << "Processing oversized ROIs\n";)
			processNontrivialRois(nontrivRoiLabels, intens_fpath, label_fpath, 1/*num_FL_threads*/);
		}

		return true;
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

	int processDataset(
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		int numFastloaderThreads,
		int numSensemakerThreads,
		int numReduceThreads,
		int min_online_roi_size,
		const SaveOption saveOption,
		const std::string& outputPath)
	{

		#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
			Stopwatch::reset();
		#endif		

		// One-time initialization
		init_feature_buffers();

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize arrow writer if needed
		if (write_apache) {

			theEnvironment.arrow_stream = ArrowOutputStream(); 
			auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file(
				saveOption, 
				get_arrow_filename(outputPath, theEnvironment.nyxus_result_fname, saveOption), 
				Nyxus::get_header(theFeatureSet.getEnabledFeatures()));
			
			if (!status) {
				std::cout << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		bool ok = true;

		// Iterate file pattern-filtered images of the dataset
		auto nf = intensFiles.size();
		for (int i = 0; i < nf; i++)
		{
#ifdef CHECKTIMING
			if (Stopwatch::exclusive())
				Stopwatch::reset();
#endif

			// Clear ROI data cached for the previous image
			clear_feature_buffers();

			auto& ifp = intensFiles[i],
				& lfp = labelFiles[i];

			// Cache the file names to be picked up by labels to know their file origin
			fs::path p_int(ifp), p_seg(lfp);
			theSegFname = p_seg.string();
			theIntFname = p_int.string();

			// Scan one label-intensity pair 
			ok = theImLoader.open(theIntFname, theSegFname);
			if (ok == false)
			{
				std::cout << "Terminating\n";
				return 1;
			}

			// Do phased processing: prescan, trivial ROI processing, oversized ROI processing
			ok = processIntSegImagePair(ifp, lfp, numFastloaderThreads, i, nf);

			if (ok == false)
			{
				std::cout << "processIntSegImagePair() returned an error code while processing file pair " << ifp << " and " << lfp << std::endl;
				return 1;
			}

			if (write_apache) {

				auto [status, msg] = theEnvironment.arrow_stream.write_arrow_file(Nyxus::get_feature_values());
				
				if (!status) {
					std::cout << "Error writing Arrow file: " << msg.value() << std::endl;
					return 2;
				}
			} else if (saveOption == SaveOption::saveCSV) 
			{
				ok = save_features_2_csv(ifp, lfp, outputPath);

				if (ok == false)
				{
					std::cout << "save_features_2_csv() returned an error code" << std::endl;
					return 2;
				}
			} else 
			{
				ok = save_features_2_buffer(theResultsCache);

				if (ok == false)
				{
					std::cout << "save_features_2_buffer() returned an error code" << std::endl;
					return 2;
				}
			}

			theImLoader.close();

			// Save nested ROI related info of this image
			if (theEnvironment.nestedOptions.defined())
				save_nested_roi_info(nestedRoiData, uniqueLabels, roiData);

			#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
			#endif

			#ifdef CHECKTIMING
			if (Stopwatch::exclusive())
			{
				// Detailed timing - on the screen
				VERBOSLVL1(Stopwatch::print_stats());

				// Details - also to a file
				VERBOSLVL3(
					fs::path p(theSegFname);
					Stopwatch::save_stats(theEnvironment.output_dir + "/" + p.stem().string() + "_nyxustiming.csv");
				);
			}
			#endif
		}

		#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
		{
			// Detailed timing - on the screen
			VERBOSLVL1(Stopwatch::print_stats());

			// Details - also to a file
			VERBOSLVL3(
				fs::path p(theSegFname);
				Stopwatch::save_stats(theEnvironment.output_dir + "/inclusive_nyxustiming.csv");
			);
		}
		#endif

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

	int processDataset_3D (
		const std::vector <Imgfile3D_layoutA>& intensFiles,
		const std::vector <Imgfile3D_layoutA>& labelFiles,
		int numFastloaderThreads,
		int numSensemakerThreads,
		int numReduceThreads,
		int min_online_roi_size,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
			Stopwatch::reset();
		#endif		

		// One-time initialization
		init_feature_buffers();

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize arrow writer if needed
		if (write_apache) 
		{
			theEnvironment.arrow_stream = ArrowOutputStream();
			auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file(
				saveOption,
				get_arrow_filename(outputPath, theEnvironment.nyxus_result_fname, saveOption),
				Nyxus::get_header(theFeatureSet.getEnabledFeatures()));

			if (!status) 
			{
				std::cout << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		bool ok = true;

		// Iterate intensity-mask image pairs
		auto nf = intensFiles.size();
		for (int i = 0; i < nf; i++)
		{
			#ifdef CHECKTIMING
			if (Stopwatch::exclusive())
				Stopwatch::reset();
			#endif

			// Clear slide's ROI labels and cache allocated the previous image
			clear_feature_buffers();

			auto& ifile = intensFiles[i],	// intensity
				& mfile = labelFiles[i];	// mask

			/*
			* 
			* for (const auto z : intensFiles[i].z_indices) :
			* 

			// ifile and mfile contain a placeholder for the z-index. We need to turn them to physical filesystem files
			std::string phys_ifname = std::regex_replace (ifile.fname, std::regex("\*"), std::to_string(z)),
				phys_mfname = std::regex_replace (mfile.fname, std::regex("\*"), std::to_string(z));

			// Cache the file names to be picked up by labels to know their file origin
			theSegFname = mfile.fdir + phys_ifname;
			theIntFname = ifile.fdir + phys_mfname;

			// Extract features from this intensity-mask pair 
			ok = theImLoader.open (theIntFname, theSegFname);
			if (ok == false)
			{
				std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
				return 1;
			}		

			*/

			// Do phased processing: prescan, trivial ROI processing, oversized ROI processing
			ok = processIntSegImagePair_3D (ifile.fdir+ifile.fname, mfile.fdir+mfile.fname, i, nf, intensFiles[i].z_indices);
			if (ok == false)
			{
				std::cerr << "processIntSegImagePair() returned an error code while processing file pair " << ifile.fname << " - " << mfile.fname << '\n';
				return 1;
			}

			// Output features
			if (write_apache) {

				auto [status, msg] = theEnvironment.arrow_stream.write_arrow_file(Nyxus::get_feature_values());

				if (!status) {
					std::cout << "Error writing Arrow file: " << msg.value() << std::endl;
					return 2;
				}
			}
			else if (saveOption == SaveOption::saveCSV)
			{
				ok = save_features_2_csv(ifile.fname, mfile.fname, outputPath);

				if (ok == false)
				{
					std::cout << "save_features_2_csv() returned an error code" << std::endl;
					return 2;
				}
			}
			else {
				ok = save_features_2_buffer(theResultsCache);

				if (ok == false)
				{
					std::cout << "save_features_2_buffer() returned an error code" << std::endl;
					return 2;
				}
			}

			// Save nested ROI related info of this image
			if (theEnvironment.nestedOptions.defined())
				save_nested_roi_info(nestedRoiData, uniqueLabels, roiData);

			#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
			#endif

		} //- pairs

		#ifdef CHECKTIMING
		if (Stopwatch::inclusive())
		{
			// Detailed timing - on the screen
			VERBOSLVL1(Stopwatch::print_stats());

			// Details - also to a file
			VERBOSLVL3(
				fs::path p(theSegFname);
				Stopwatch::save_stats(theEnvironment.output_dir + "/inclusive_nyxustiming.csv");
			);
		}
		#endif

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
			clear_feature_buffers();

			auto image_idx = i * width * height;

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
				for (auto pxl : r.raw_pixels_3D)
					f << lab << "," << pxl.x << ',' << pxl.y << ',' << pxl.z << ',' << pxl.inten << ',' << '\n';
			else
				for (auto pxl : r.raw_pixels)
					f << lab << "," << pxl.x << ',' << pxl.y << ',' << pxl.inten << ',' << '\n';
		}

		f.flush();
	}

}