#include <fstream>
#include <future>
#include <string>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

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
#include "helpers/fsystem.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "raw_image_loader.h"
#include "save_option.h"

namespace Nyxus
{
	bool featurize_triv_wholevolume (size_t sidx, ImageLoader& imlo, size_t memory_limit, LR& vroi)
	{
		const std::string& ifpath = LR::dataset_props[sidx].fname_int;

		// can we process this slide ?
		size_t footp = vroi.get_ram_footprint_estimate();
		if (footp > memory_limit)
		{
			std::string erm = "Error: cannot process slide " + ifpath + " , reason: its memory footprint " + virguler_ulong(footp) + " exceeds available memory " + virguler_ulong(memory_limit);
			#ifdef WITH_PYTHON_H
				throw std::runtime_error(erm);
			#endif	
			std::cerr << erm << "\n";			
			return false;
		}

		// read the slide into a pixel cloud
		if (theEnvironment.anisoOptions.customized() == false)
		{
			VERBOSLVL2(std::cout << "\nscan_trivial_wholeslide()\n");
			scan_trivial_wholevolume (vroi, ifpath, imlo);
		}
		else
		{
			VERBOSLVL2(std::cout << "\nscan_trivial_wholeslide_ANISO()\n");
			double aniso_x = theEnvironment.anisoOptions.get_aniso_x(),
				aniso_y = theEnvironment.anisoOptions.get_aniso_y();
			scan_trivial_wholevolume_anisotropic (vroi, ifpath, imlo, aniso_x, aniso_y);
		}

		// allocate memory for feature helpers (image matrix, etc)
		VERBOSLVL2(std::cout << "\tallocating vROI buffers\n");
		size_t h = vroi.aabb.get_height(), w = vroi.aabb.get_width(), d = vroi.aabb.get_z_depth();
		vroi.aux_image_cube.allocate (w, h, d);

		// calculate the image matrix or cube 
		vroi.aux_image_cube.calculate_from_pixelcloud (vroi.raw_pixels_3D, vroi.aabb);

		// calculate features 
		VERBOSLVL2(std::cout << "\treducing whole slide\n");
		reduce_trivial_3d_wholevolume (vroi);

		// free memory
		VERBOSLVL2(std::cout << "\tfreeing vROI buffers\n");
		if (vroi.aux_image_matrix._pix_plane.size())
			std::vector<Pixel3>().swap (vroi.raw_pixels_3D);

		// no need to calculate neighbor features in WV/WSI, returning
		return true;
	}

	bool featurize_wholevolume (size_t sidx, ImageLoader& imlo, LR& vroi)
	{
		// phase 1: gather ROI metrics
		VERBOSLVL2(std::cout << "Gathering vROI metrics\n");

		// phase 1: copy ROI metrics from the slide properties, thanks to the WSI scenario
		// instead of gather_wholeslide_metrics (p.fname_int, imlo, vroi)
		vroi.slide_idx = (decltype(vroi.slide_idx))sidx;
		const SlideProps& p = LR::dataset_props[sidx];
		vroi.aux_area = p.max_roi_area;
		vroi.aabb.init_from_widthheight(p.max_roi_w, p.max_roi_h);
		// tell ROI the actual uint rynamic range or greybinned one depending on the slide's low-level properties
		vroi.aux_min = (PixIntens)p.fp_phys_pivoxels ? 0 : (PixIntens)p.min_preroi_inten;
		vroi.aux_max = (PixIntens)p.fp_phys_pivoxels ? (PixIntens)Nyxus::theEnvironment.fpimageOptions.target_dyn_range() : (PixIntens)p.max_preroi_inten;

		// fix the AABB with respect to anisotropy
		if (theEnvironment.anisoOptions.customized() == false)
			vroi.aabb.apply_anisotropy(
				theEnvironment.anisoOptions.get_aniso_x(),
				theEnvironment.anisoOptions.get_aniso_y());

		// prepare (zero) ROI's feature value buffer
		vroi.initialize_fvals();

		// assess ROI's memory footprint and check if we can featurize it as phase 2 (trivially) ?
		size_t roiFootprint = vroi.get_ram_footprint_estimate(),
			ramLim = theEnvironment.get_ram_limit();
		if (roiFootprint >= ramLim)
		{
			VERBOSLVL2(
				std::cout << "oversized slide "
				<< " (S=" << vroi.aux_area
				<< " W=" << vroi.aabb.get_width()
				<< " H=" << vroi.aabb.get_height()
				<< " D=" << vroi.aabb.get_z_depth()
				<< " px footprint=" << Nyxus::virguler_ulong(roiFootprint) << " b"
				<< ") while RAM limit is " << Nyxus::virguler_ulong(ramLim) << "\n"
			);

			std::cerr << p.fname_int << ": slide is non-trivial \n";
			return false;
		}

		// phase 2: extract features
		featurize_triv_wholevolume (sidx, imlo, theEnvironment.get_ram_limit(), vroi); // segmented counterpart: phase2.cpp / processTrivialRois ()

		return true;
	}

	void featurize_3d_wv_thread(
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		size_t slide_idx,
		size_t nf,
		const std::string& outputPath,
		bool write_apache,
		Nyxus::SaveOption saveOption,
		int& rv)
	{
		SlideProps& p = LR::dataset_props[slide_idx];

		// scan one slide
		ImageLoader imlo;
		if (imlo.open(p) == false)
		{
			std::cerr << "Terminating\n";
			rv = 1;
		}

		LR vroi(1); // virtual ROI representing the whole slide ROI-labelled as '1'

		if (featurize_wholevolume (slide_idx, imlo, vroi) == false)	// non-wsi counterpart: processIntSegImagePair()
		{
			std::cerr << "Error featurizing slide " << p.fname_int << " @ " << __FILE__ << ":" << __LINE__ << "\n";
			rv = 1;
		}

		// thread-safely save results of this single slide
		if (write_apache)
		{
			auto [status, msg] = save_features_2_apache_wholeslide(vroi, p.fname_int);
			if (!status)
			{
				std::cerr << "Error writing Arrow file: " << msg.value() << std::endl;
				rv = 2;
			}
		}
		else
			if (saveOption == SaveOption::saveCSV)
			{
				if (save_features_2_csv_wholeslide(vroi, p.fname_int, "", outputPath) == false)
				{
					std::cerr << "save_features_2_csv() returned an error code" << std::endl;
					rv = 2;
				}
			}
			else
			{
				// pulls feature values from 'vroi' and appends them to global object 'theResultsCache' exposed to Python API
				if (save_features_2_buffer_wholeslide(Nyxus::theResultsCache, vroi, p.fname_int, "") == false)
				{
					std::cerr << "Error saving features to the results buffer" << std::endl;
					rv = 2;
				}
			}

		imlo.close();

		//
		// Not saving nested ROI related info because this image is single-ROI (whole-slide)
		//

		rv = 0; // success
	}



	std::tuple<bool, std::optional<std::string>> processDataset_3D_wholevolume (
		const std::vector <std::string>& intensFiles,
		int n_threads,
		int min_online_roi_size,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		//**** prescan all files

		VERBOSLVL1(std::cout << "phase 0 (prescanning)\n");

		LR::reset_dataset_props();
		size_t nf = intensFiles.size();
		LR::dataset_props.resize(nf);
		for (size_t i = 0; i < nf; i++)
		{
			// slide file names
			SlideProps& p = LR::dataset_props[i];
			p.fname_int = intensFiles[i];
			p.fname_seg = "";

			// slide metrics
			VERBOSLVL1(std::cout << "prescanning " << p.fname_int);
			if (!scan_slide_props(p, 3, theEnvironment.resultOptions.need_annotation()))
			{
				std::string erm = "error prescanning " + p.fname_int;
				return { false, erm };
			}

			VERBOSLVL1(std::cout << " " << p.slide_w << " W x" << p.slide_h << " H max ROI " << p.max_roi_w << "x" << p.max_roi_h
				<< " DR " << Nyxus::virguler_real(p.min_preroi_inten)
				<< "-" << Nyxus::virguler_real(p.max_preroi_inten)
				<< " " << p.lolvl_slide_descr << "\n");

		}

		// global properties
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
			LR::dataset_max_roi_w = (std::max)(LR::dataset_max_roi_w, p.max_roi_w);
			LR::dataset_max_roi_h = (std::max)(LR::dataset_max_roi_h, p.max_roi_h);
		}

		VERBOSLVL1(std::cout << "\t finished prescanning \n");

		//
		// future: allocate GPU cache for all participating devices
		//

		//**** extract features

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
				std::string erm = "Error creating Arrow file: " + outputPath + " reason: " + msg.value();
				return {false, erm};
			}
		}

		// run batches of threads

		size_t n_jobs = (nf + n_threads - 1) / n_threads;
		for (size_t j = 0; j < n_jobs; j++)
		{
			VERBOSLVL1(std::cout << "whole-slide job " << j + 1 << "/" << n_jobs << "\n");

			std::vector<std::future<void>> T;
			for (int t = 0; t < n_threads; t++)
			{
				size_t idx = j * n_threads + t;

				// done?
				if (idx + 1 > nf)
					break;

				int rval = 0;
				if (n_threads > 1)
				{
					T.push_back(std::async(std::launch::async,
						featurize_3d_wv_thread,
						intensFiles,
						intensFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						std::ref(rval)));
				}
				else
				{
					featurize_3d_wv_thread(
						intensFiles,
						intensFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						rval);
				}
			}

			// allow keyboard interrupt
			#ifdef WITH_PYTHON_H
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
			#endif

		}

		//**** finalize Apache output

		if (write_apache)
		{
			// close arrow file after use
			auto [status, msg] = theEnvironment.arrow_stream.close_arrow_file();
			if (!status)
			{
				std::string erm = "Error closing Arrow file: " + msg.value();
				return { false, erm };
			}
		}

		//
		// future: free GPU cache for all participating devices
		//

		return {true, std::nullopt}; // success
	}


}