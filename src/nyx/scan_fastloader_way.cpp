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
#include "features/contour.h"
#include "features/erosion.h"
#include "features/gabor.h"
#include "features/2d_geomoments.h"
#include "globals.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"

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

	bool processIntSegImagePair (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int filepair_index, int tot_num_filepairs)
	{
		std::vector<int> trivRoiLabels, nontrivRoiLabels;

		// Timing block (image scanning)
		{
			//______	STOPWATCH("Image scan/ImgScan/Scan/lightsteelblue", "\t=");

			{ STOPWATCH("Image scan1/scan1/s1/#aabbcc", "\t=");

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

			{ STOPWATCH("Image scan2a/scan2a/s2a/#aabbcc", "\t=");

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

			{ STOPWATCH("Image scan2b/scan2b/s2b/#aabbcc", "\t=");

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

			{ STOPWATCH("Image scan3/scan3/s3/#aabbcc", "\t=");

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

			{ STOPWATCH("Image scan1/scan1/s1/#aabbcc", "\t=");

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

			{ STOPWATCH("Image scan2a/scan2a/s2a/#aabbcc", "\t=");
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

			{ STOPWATCH("Image scan2b/scan2b/s2b/#aabbcc", "\t=");

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

			{ STOPWATCH("Image scan3/scan3/s3/#aabbcc", "\t=");

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

	bool gatherRoisMetrics_2_slideprops (ImageLoader & ilo, SlideProps & p)
	{
		bool wholeslide = p.fname_int == p.fname_seg;

		std::unordered_set<int> U;	// unique ROI mask labels
		std::unordered_map <int, LR> R;	// ROI data

		// Reset per-image counters and extrema
		LR::reset_global_stats();

		int lvl = 0, // pyramid level
			lyr = 0; //	layer

		// Read the tiff. The image loader is put in the open state in processDataset()
		size_t nth = ilo.get_num_tiles_hor(),
			ntv = ilo.get_num_tiles_vert(),
			fw = ilo.get_tile_width(),
			th = ilo.get_tile_height(),
			tw = ilo.get_tile_width(),
			tileSize = ilo.get_tile_size(),
			fullwidth = ilo.get_full_width(),
			fullheight = ilo.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = ilo.load_tile(row, col);
				if (!ok)
				{
#ifdef WITH_PYTHON_H
					throw "Error fetching tile";
#endif	
					std::cerr << "Error fetching tile\n";
					return false;
				}

				// Get ahold of tile's pixel buffer
				auto tidx = row * nth + col;
				auto data_I = ilo.get_int_tile_buffer(),
					data_L = ilo.get_seg_tile_buffer();

				// Iterate pixels
				for (size_t i = 0; i < tileSize; i++)
				{
					// Skip non-mask pixels
					auto msk = data_L [i];
					if (!msk)
					{
						// Update zero-background area
						zero_background_area++;
						continue;
					}

					// Collapse all the labels to one if single-ROI mde is requested
					if (wholeslide)
						msk = 1;

					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					auto inten = data_I [i];

					// Update pixel's ROI metrics
					//		- the following block mocks feed_pixel_2_metrics (x, y, dataI[i], msk, tidx)
					if (U.find(msk) == U.end())
					{
						// Remember this label
						U.insert(msk);

						// Initialize the ROI label record
						LR r;
						//		- mocking init_label_record_2(newData, theSegFname, theIntFname, x, y, label, intensity, tile_index)
						// Initialize basic counters
						r.aux_area = 1;
						r.aux_min = r.aux_max = inten;
						r.init_aabb(x, y);
						// Cache the ROI label
						r.label = msk;

						//		- not storing file names (r.segFname = segFile, r.intFname = intFile) but will do so in the future

						// Attach
						R[msk] = r;
					}
					else
					{
						// Update basic ROI info (info that doesn't require costly calculations)
						LR & r = R[msk];

						//		- mocking update_label_record_2 (r, x, y, label, intensity, tile_index)
						
						// Per-ROI 
						r.aux_area++;

						r.aux_min = (std::min) (r.aux_min, inten);
						r.aux_max = (std::max) (r.aux_max, inten);

						// save
						r.update_aabb(x, y);
					}
				} // scan tile

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

				// Show stayalive progress info
				VERBOSLVL2(
					if (cnt++ % 4 == 0)
						std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
				);
			} // foreach tile

		//****** Analysis
		
		// slide-wide (max ROI area) x (number of ROIs)
		size_t maxArea = 0;
		size_t max_w = 0, max_h = 0;
		for (const auto & pair : R)
		{
			const LR& r = pair.second;
			maxArea = maxArea > r.aux_area ? maxArea : r.aux_area; //std::max (maxArea, r.aux_area);
			max_w = max_w > r.aabb.get_width() ? max_w : r.aabb.get_width();
			max_h = max_h > r.aabb.get_height() ? max_h : r.aabb.get_height();
		}
		p.max_roi_area = maxArea;
		p.n_rois = R.size();
		p.max_roi_w = max_w;
		p.max_roi_h = max_h;

		return true;
	}

	bool scan_intlabel_pair_props (SlideProps & p)
	{
		ImageLoader ilo;
		if (!ilo.open(p.fname_int, p.fname_seg))
		{
			std::cerr << "error opening an ImageLoader for " << p.fname_int << " | " << p.fname_seg << "\n";
			return false;
		}

		if (!gatherRoisMetrics_2_slideprops(ilo, p))
		{
			std::cerr << "error in gatherRoisMetrics_2_slideprops() \n";
			return false;
		}

		ilo.close();

		return true;
	}

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

		//********************** prescan ***********************

		#ifdef USE_GPU
		// what parts of GPU cache we need to bother about ?
		bool needContour = ContourFeature::required(theFeatureSet),
			needErosion = ErosionPixelsFeature::required(theFeatureSet),
			needGabor = GaborFeature::required(theFeatureSet),
			needImoments = Imoms2D_feature::required(theFeatureSet),
			needSmoments = Smoms2D_feature::required(theFeatureSet),
			needMoments = needImoments || needSmoments; // ImageMomentsFeature::required (theFeatureSet);
		#endif

		// scan the whole dataset for ROI properties, slide properties, and dataset global properties
		size_t nf = intensFiles.size();

		{ STOPWATCH("prescan/p0/P/#ccbbaa", "\t=");

		VERBOSLVL1 (std::cout << "phase 0 (prescanning)\n");

		LR::reset_dataset_props();
		LR::dataset_props.resize(nf);
		for (size_t i = 0; i < nf; i++)
		{
			auto& ifp = intensFiles[i],
				& mfp = labelFiles[i];

			SlideProps& p = LR::dataset_props[i];
			p.fname_int = ifp;
			p.fname_seg = mfp;

			VERBOSLVL1 (std::cout << "prescanning " << p.fname_int);
			if (!scan_intlabel_pair_props(p))
			{
				VERBOSLVL1 (std::cout << "error prescanning pair " << ifp << " and " << mfp << std::endl);
				return 1;
			}
			VERBOSLVL1 (std::cout << "\tmax ROI " << p.max_roi_w << " x " << p.max_roi_h << "\n");
		}

		// get global properties
		LR::dataset_max_combined_roicloud_len = 0;
		LR::dataset_max_n_rois = 0;
		LR::dataset_max_roi_area = 0;
		LR::dataset_max_roi_w = 0;
		LR::dataset_max_roi_h = 0;

		for (SlideProps& p : LR::dataset_props)
		{
			size_t sup_s_n = p.n_rois * p.max_roi_area;
			LR::dataset_max_combined_roicloud_len = (std::max)(LR::dataset_max_combined_roicloud_len, sup_s_n);

			LR::dataset_max_n_rois = (std::max)(LR::dataset_max_n_rois, p.n_rois);
			LR::dataset_max_roi_area = (std::max)(LR::dataset_max_roi_area, p.max_roi_area);

			LR::dataset_max_roi_w = (std::max) (LR::dataset_max_roi_w, p.max_roi_w);
			LR::dataset_max_roi_h = (std::max)(LR::dataset_max_roi_h, p.max_roi_h);
		}

		VERBOSLVL1 (std::cout << "\t finished phase 0 \n");

		//***********************************************************************************************
#ifdef USE_GPU
		if (theEnvironment.using_gpu())
		{
			// allocate
			VERBOSLVL1 (std::cout << "allocate GPU cache \n");

			if (!NyxusGpu::allocate_gpu_cache(
				// out
				NyxusGpu::gpu_roiclouds_2d,
				NyxusGpu::gpu_roicontours_2d,
				&NyxusGpu::dev_realintens,
				&NyxusGpu::dev_prereduce,
				NyxusGpu::gpu_featurestatebuf,
				NyxusGpu::devicereduce_temp_storage_szb,
				&NyxusGpu::dev_devicereduce_temp_storage,	
				NyxusGpu::gpu_batch_len,
				&NyxusGpu::dev_imat1,
				&NyxusGpu::dev_imat2,
				NyxusGpu::gabor_linear_image,
				NyxusGpu::gabor_linear_kernel,
				NyxusGpu::gabor_result,
				NyxusGpu::gabor_energy_image,
				// in
				needContour,
				needErosion,
				needGabor,
				needMoments,
				LR::dataset_max_combined_roicloud_len, // desired totCloLen,
				LR::dataset_max_combined_roicloud_len, // desired totKontLen,
				LR::dataset_max_n_rois,	// labels.size()
				LR::dataset_max_roi_area,
				LR::dataset_max_roi_w,
				LR::dataset_max_roi_h,
				GaborFeature::f0_theta_pairs.size(),
				GaborFeature::n
				))	// we need max ROI area inside the function to calculate the batch size if 'dataset_max_combined_roicloud_len' doesn't fit in RAM
			{
				std::cerr << "error in " << __FILE__ << ":" << __LINE__ << "\n";
				return 1;
			}

			VERBOSLVL1(std::cout << "\t ---done allocating GPU cache \n");
		}
#endif
		} // prescan timing

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
				VERBOSLVL1(
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
			VERBOSLVL1(
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

#ifdef USE_GPU
		if (theEnvironment.using_gpu())
		{
			if (!NyxusGpu::free_gpu_cache(
				/*??????
				needContour,
				needErosion,
				needGabor,
				needMoments, 
				*/
				NyxusGpu::gpu_roiclouds_2d,
				NyxusGpu::gpu_roicontours_2d,
				NyxusGpu::dev_realintens,
				NyxusGpu::dev_prereduce,
				NyxusGpu::gpu_featurestatebuf,
				NyxusGpu::dev_devicereduce_temp_storage,
				NyxusGpu::dev_imat1,
				NyxusGpu::dev_imat2,
				NyxusGpu::gabor_linear_image,
				NyxusGpu::gabor_result,
				NyxusGpu::gabor_linear_kernel,
				NyxusGpu::gabor_energy_image))
			{
				std::cerr << "error in free_gpu_cache()\n";
				return 1;
			}
		}
#endif
	
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