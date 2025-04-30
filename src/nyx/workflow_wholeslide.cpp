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
	bool featurize_triv_wholeslide (size_t sidx, ImageLoader & imlo, size_t memory_limit, LR & vroi)
	{
		const std::string & ifpath = LR::dataset_props[sidx].fname_int;

		// can we process this slide ?
		size_t footp = vroi.get_ram_footprint_estimate();
		if (footp > memory_limit)
		{
			std::cerr << "Error: cannot process slide " << ifpath << " , reason: its memory footprint " << virguler(footp) << " exceeds available memory " << virguler(memory_limit) << "\n";
			return false;
		}

		// read the slide into a pixel cloud
		if (theEnvironment.anisoOptions.customized() == false)
		{
			VERBOSLVL2(std::cout << "\nscan_trivial_wholeslide()\n");
			scan_trivial_wholeslide (vroi, ifpath, imlo); // counterpart of segmented scanTrivialRois ()
		}
		else
		{
			VERBOSLVL2(std::cout << "\nscan_trivial_wholeslide_ANISO()\n");
			double aniso_x = theEnvironment.anisoOptions.get_aniso_x(),
				aniso_y = theEnvironment.anisoOptions.get_aniso_y();
			scan_trivial_wholeslide_anisotropic (vroi, ifpath, imlo, aniso_x, aniso_y); // counterpart of segmented scanTrivialRois ()
		}

		// allocate memory for feature helpers (image matrix, etc)
		VERBOSLVL2(std::cout << "\tallocating vROI buffers\n");
		size_t h = vroi.aabb.get_height(), w = vroi.aabb.get_width();
		size_t len = w * h;
		vroi.aux_image_matrix.allocate (w, h);

		// calculate the image matrix or cube 
		vroi.aux_image_matrix.calculate_from_pixelcloud (vroi.raw_pixels, vroi.aabb);

		// calculate features 
		VERBOSLVL2(std::cout << "\treducing whole slide\n");
		reduce_trivial_wholeslide (vroi);	// counterpart of segmented reduce_trivial_rois_manual()

		// free memory
		VERBOSLVL2(std::cout << "\tfreeing vROI buffers\n");
		if (vroi.aux_image_matrix._pix_plane.size())
			std::vector<PixIntens>().swap(vroi.aux_image_matrix._pix_plane);

		// allow heyboard interrupt
		#ifdef WITH_PYTHON_H
		if (PyErr_CheckSignals() != 0)
		{
			sureprint("\nAborting per user input\n");
			throw pybind11::error_already_set();
		}
		#endif

		// no need to calculate neighbor features in WSI, returning
		return true;
	}

	bool featurize_wholeslide (size_t sidx, ImageLoader& imlo, LR& vroi)
	{
		// phase 1: gather ROI metrics
		VERBOSLVL2(std::cout << "Gathering vROI metrics\n");

		// phase 1: copy ROI metrics from the slide properties, thanks to the WSI scenario
		// instead of gather_wholeslide_metrics (p.fname_int, imlo, vroi)	// segmented counterpart: gatherRoisMetrics()
		vroi.slide_idx = (decltype(vroi.slide_idx)) sidx;
		const SlideProps & p = LR::dataset_props [sidx];
		vroi.aux_area = p.max_roi_area;
		vroi.aabb.init_from_widthheight (p.max_roi_w, p.max_roi_h);
		vroi.aux_min = (PixIntens) p.min_preroi_inten;
		vroi.aux_max = (PixIntens) p.max_preroi_inten;

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
				<< " px footprint=" << Nyxus::virguler(roiFootprint) << " b"
				<< ") while RAM limit is " << Nyxus::virguler(ramLim) << "\n"
			);

			std::cerr << p.fname_int << ": slide is non-trivial \n";
			return false;
		}

		// phase 2: extract features
		featurize_triv_wholeslide (sidx, imlo, theEnvironment.get_ram_limit(), vroi); // segmented counterpart: phase2.cpp / processTrivialRois ()

		return true;
	}

	void featurize_wsi_thread (
		const std::vector<std::string> & intensFiles, 
		const std::vector<std::string> & labelFiles, 
		size_t slide_idx, 
		size_t nf, 
		const std::string & outputPath, 
		bool write_apache, 
		Nyxus::SaveOption saveOption,
		int & rv)
	{
		SlideProps & p = LR::dataset_props [slide_idx];

		// scan one slide
		ImageLoader imlo;
		if (imlo.open(p) == false)
		{
			std::cerr << "Terminating\n";
			rv = 1;
		}

		LR vroi (1); // virtual ROI representing the whole slide ROI-labelled as '1'

		if (featurize_wholeslide (slide_idx, imlo, vroi) == false)	// non-wsi counterpart: processIntSegImagePair()
		{
			std::cerr << "Error featurizing slide " << p.fname_int << " @ " << __FILE__ << ":" << __LINE__ << "\n";
			rv = 1;
		}

		// thread-safely save results of this single slide
		if (write_apache) 
		{
			auto [status, msg] = save_features_2_apache_wholeslide (vroi, p.fname_int);
			if (! status) 
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
				if (save_features_2_buffer_wholeslide (Nyxus::theResultsCache, vroi, p.fname_int, "") == false)
				{
					std::cerr << "Error saving features to the results buffer" << std::endl;
					rv = 2;
				}
			}

		imlo.close();

		//
		// Not saving nested ROI related info because this image is single-ROI (whole-slide)
		//

		// allow heyboard interrupt
		#ifdef WITH_PYTHON_H
		if (PyErr_CheckSignals() != 0)
		{
			sureprint("\nAborting per user input\n");
			throw pybind11::error_already_set();
		}
		#endif

		rv = 0; // success
	}

	int processDataset_2D_wholeslide (
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		int n_threads,
		int min_online_roi_size,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		//**** prescan all slides

		size_t nf = intensFiles.size();

		VERBOSLVL1(std::cout << "phase 0 (prescanning)\n");

		LR::reset_dataset_props();
		LR::dataset_props.resize(nf);
		for (size_t i=0; i<nf; i++)
		{
			// slide file names
			SlideProps& p = LR::dataset_props[i];
			p.fname_int = intensFiles[i];
			p.fname_seg = labelFiles[i];

			// slide metrics
			VERBOSLVL1(std::cout << "prescanning " << p.fname_int);
			if (! scan_slide_props(p, theEnvironment.resultOptions.need_annotation()))
			{
				VERBOSLVL1(std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return 1;
			}
			VERBOSLVL1(std::cout << "\t " << p.slide_w << " W x " << p.slide_h << " H\tmax ROI " << p.max_roi_w << " x " << p.max_roi_h << "\tmin-max I " << Nyxus::virguler(p.min_preroi_inten) << "-" << Nyxus::virguler(p.max_preroi_inten) << "\t" << p.lolvl_slide_descr << "\n");

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

		// initialize buffer output
		Nyxus::theResultsCache.clear();

		// initialize Apache output
		if (write_apache) 
		{
			theEnvironment.arrow_stream = ArrowOutputStream();
			std::string afn = get_arrow_filename (outputPath, theEnvironment.nyxus_result_fname, saveOption);
			VERBOSLVL2 (std::cout << "arrow file name =" << afn << "\n");
			auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file (saveOption, afn, Nyxus::get_header(theFeatureSet.getEnabledFeatures()));

			if (!status) 
			{
				std::cerr << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		// run batches of threads
		size_t n_jobs = (nf + n_threads - 1) / n_threads;
		for (size_t j=0; j<n_jobs; j++)
		{
			VERBOSLVL1 (std::cout << "whole-slide job " << j+1 << "/" << n_jobs << "\n");

			std::vector<std::future<void>> T;
			for (int t=0; t < n_threads; t++)
			{
				size_t idx = j * n_threads + t;

				// done?
				if (idx + 1 > nf)
					break;

				int rval = 0;
				if (n_threads > 1)
				{
					T.push_back(std::async(std::launch::async,
						featurize_wsi_thread,
						intensFiles,
						labelFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						std::ref(rval)));
				}
				else
				{
					featurize_wsi_thread (
						intensFiles,
						labelFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						rval);
				}
			}
		}

		//**** finalize Apache output

		if (write_apache) 
		{
			// close arrow file after use
			auto [status, msg] = theEnvironment.arrow_stream.close_arrow_file();
			if (!status) 
			{
				std::cerr << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}

		//
		// future: free GPU cache for all participating devices
		//

		return 0; // success
	}

}

