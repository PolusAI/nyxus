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
	bool featurize_triv_wholevolume (Environment & env, size_t sidx, ImageLoader& imlo, size_t memory_limit, LR& vroi)
	{
		const std::string& ifpath = env.dataset.dataset_props[sidx].fname_int;

		// can we process this slide ?
		size_t footp = vroi.get_ram_footprint_estimate (1);	// 1 since single-ROI
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
		if (env.anisoOptions.customized() == false)
		{
			VERBOSLVL2(env.get_verbosity_level(), std::cout << "\nscan_trivial_wholeslide()\n");
			scan_trivial_wholevolume (vroi, ifpath, imlo);
		}
		else
		{
			VERBOSLVL2(env.get_verbosity_level(), std::cout << "\nscan_trivial_wholeslide_ANISO()\n");
			scan_trivial_wholevolume_anisotropic (
				vroi, 
				ifpath, 
				imlo, 
				env.anisoOptions.get_aniso_x(),
				env.anisoOptions.get_aniso_y(),
				env.anisoOptions.get_aniso_z());
		}

		// allocate memory for feature helpers (image matrix, etc)
		VERBOSLVL2(env.get_verbosity_level(), std::cout << "\tallocating vROI buffers\n");
		size_t h = vroi.aabb.get_height(), w = vroi.aabb.get_width(), d = vroi.aabb.get_z_depth();
		vroi.aux_image_cube.allocate (w, h, d);

		// calculate the image matrix or cube 
		vroi.aux_image_cube.calculate_from_pixelcloud (vroi.raw_pixels_3D, vroi.aabb);

		// calculate features 
		VERBOSLVL2(env.get_verbosity_level(), std::cout << "\treducing whole slide\n");
		reduce_trivial_3d_wholevolume (env, vroi);

		// free memory
		VERBOSLVL2(env.get_verbosity_level(), std::cout << "\tfreeing vROI buffers\n");
		if (vroi.aux_image_matrix._pix_plane.size())
			std::vector<Pixel3>().swap (vroi.raw_pixels_3D);

		// no need to calculate neighbor features in WV/WSI, returning
		return true;
	}

	bool featurize_wholevolume (Environment & env, size_t sidx, ImageLoader& imlo, LR& vroi)
	{
		//***** phase 1: copy ROI metrics from the slide properties, thanks to the WSI scenario
		const SlideProps& p = env.dataset.dataset_props[sidx];
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Gathering vROI metrics " + fs::path(p.fname_int).filename().string() + "\n");

		// instead of gather_wholeslide_metrics (p.fname_int, imlo, vroi)
		vroi.slide_idx = (decltype(vroi.slide_idx))sidx;
		vroi.aux_area = p.max_roi_area;
		vroi.aabb.init_from_whd (p.max_roi_w, p.max_roi_h, p.max_roi_d);

		// tell ROI the actual uint rynamic range or greybinned one depending on the slide's low-level properties
		// with the Hounsfield adjustment
		vroi.aux_min = (PixIntens) (p.min_preroi_inten - p.min_preroi_inten); // in CT datasets p.min_preroi_inten can be -1024.0
		vroi.aux_max = (PixIntens) (p.max_preroi_inten - p.min_preroi_inten);

		// fix the AABB with respect to anisotropy
		if (env.anisoOptions.customized() == false)
			vroi.aabb.apply_anisotropy(
				env.anisoOptions.get_aniso_x(),
				env.anisoOptions.get_aniso_y());

		// prepare (zero) ROI's feature value buffer
		vroi.initialize_fvals();

		// assess ROI's memory footprint and check if we can featurize it as phase 2 (trivially) ?
		size_t roiFootprint = vroi.get_ram_footprint_estimate (1),		// 1 since single-ROI
			ramLim = env.get_ram_limit();
		if (roiFootprint >= ramLim)
		{
			VERBOSLVL2(env.get_verbosity_level(),
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

		//***** phase 2: extract features
		featurize_triv_wholevolume (env, sidx, imlo, env.get_ram_limit(), vroi); // segmented counterpart: phase2.cpp / processTrivialRois ()

		return true;
	}

	void featurize_3d_wv_thread(
		Environment & env,
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		size_t slide_idx,
		size_t nf,
		const std::string& outputPath,
		bool write_apache,
		Nyxus::SaveOption saveOption,
		int& rv)
	{
		SlideProps& p = env.dataset.dataset_props[slide_idx];

		// scan one slide
		ImageLoader imlo;
		if (imlo.open(p, env.fpimageOptions) == false)
		{
			std::cerr << "Terminating\n";
			rv = 1;
		}

		LR vroi(1); // virtual ROI representing the whole slide ROI-labelled as '1'

		if (featurize_wholevolume (env, slide_idx, imlo, vroi) == false)	// non-wsi counterpart: processIntSegImagePair()
		{
			std::cerr << "Error featurizing slide " << p.fname_int << " @ " << __FILE__ << ":" << __LINE__ << "\n";
			rv = 1;
		}

		// thread-safely save results of this single slide
		if (write_apache)
		{
			auto [status, msg] = save_features_2_apache_wholeslide (env, vroi, p.fname_int);
			if (!status)
			{
				std::cerr << "Error writing Arrow file: " << msg.value() << std::endl;
				rv = 2;
			}
		}
		else
			if (saveOption == SaveOption::saveCSV)
			{
				if (save_features_2_csv_wholeslide (env, vroi, p.fname_int, "", outputPath,
					0 // pass 0 as t_index
					) == false)
				{
					std::cout << "error saving results to CSV file, details: " << __FILE__ << ":" << __LINE__ << std::endl;
					rv = 2;
				}
			}
			else
			{
				// pulls feature values from 'vroi' and appends them to global object 'theResultsCache' exposed to Python API
				if (save_features_2_buffer_wholeslide (env.theResultsCache, env, vroi, p.fname_int, "") == false)
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
		Environment & env,
		const std::vector <std::string>& intensFiles,
		int n_threads,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		//**** prescan all files

		size_t nf = intensFiles.size();
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\nphase 0: prescanning " << nf << " slides \n");
		env.dataset.reset_dataset_props();

		for (size_t i = 0; i < nf; i++)
		{
			// slide file names
			SlideProps& p = env.dataset.dataset_props.emplace_back(intensFiles[i], "");

			// slide metrics
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "prescanning " << fs::path(p.fname_int).filename().string());

			if (! scan_slide_props(p, 3, env.anisoOptions, env.resultOptions.need_annotation()))
				return { false, "error prescanning " + p.fname_int };

			VERBOSLVL1 (env.get_verbosity_level(), std::cout << " " 
				<< p.slide_w << " W x" << p.slide_h << " H x" << p.volume_d << " D"
				<< " DR " << Nyxus::virguler_real(p.min_preroi_inten)
				<< "-" << Nyxus::virguler_real(p.max_preroi_inten)
				<< " " << p.lolvl_slide_descr << "\n");
		}

		// global properties
		env.dataset.update_dataset_props_extrema();

		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "finished prescanning \n");

		//
		// future: allocate GPU cache for all participating devices
		//

		//**** extract features

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize arrow writer if needed
		if (write_apache)
		{
			env.arrow_stream = ArrowOutputStream();
			auto [status, msg] = env.arrow_stream.create_arrow_file(
				saveOption,
				get_arrow_filename(outputPath, env.nyxus_result_fname, saveOption),
				Nyxus::get_header (env));

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
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "whole-slide job " << j + 1 << "/" << n_jobs << "\n");

			std::vector<std::future<void>> T;
			std::vector<int> rvals(n_threads, 0);
			for (int t = 0; t < n_threads; t++)
			{
				size_t idx = j * n_threads + t;

				// done?
				if (idx + 1 > nf)
					break;

				if (n_threads > 1)
				{
					T.push_back(std::async(std::launch::async,
						featurize_3d_wv_thread,
						std::ref(env),
						intensFiles,
						intensFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						std::ref(rvals[t])));
				}
				else
				{
					featurize_3d_wv_thread(
						env,
						intensFiles,
						intensFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						rvals[t]);
				}
			}

			// wait for all threads to complete before proceeding
			for (auto& f : T)
				f.get();

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
			auto [status, msg] = env.arrow_stream.close_arrow_file();
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