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
	// Featurizes a whole slide that fits in RAM. The caller owns the oversized decision -- it
	// refuses anything at or above the RAM limit before reaching here, off the same estimate,
	// so a second test of it here could never fire.
	bool featurize_triv_wholeslide (Environment & env, size_t sidx, ImageLoader & imlo, LR & vroi)
	{
		const std::string & ifpath = env.dataset.dataset_props[sidx].fname_int;

		// read the slide into a pixel cloud
		if (env.anisoOptions.customized() == false)
		{
			VERBOSLVL2(env.get_verbosity_level(), std::cout << "\nscan_trivial_wholeslide()\n");
			if (! scan_trivial_wholeslide (vroi, ifpath, imlo))	// counterpart of segmented scanTrivialRois ()
				return false;
		}
		else
		{
			VERBOSLVL2(env.get_verbosity_level(), std::cout << "\nscan_trivial_wholeslide_ANISO()\n");
			double aniso_x = env.anisoOptions.get_aniso_x(),
				aniso_y = env.anisoOptions.get_aniso_y();
			if (! scan_trivial_wholeslide_anisotropic (vroi, ifpath, imlo, aniso_x, aniso_y))	// counterpart of segmented scanTrivialRois ()
				return false;
		}

		// allocate buffers of feature helpers (image matrix, etc)
		VERBOSLVL2(env.get_verbosity_level(), std::cout << "\tallocating vROI buffers\n");
		size_t h = vroi.aabb.get_height(), w = vroi.aabb.get_width();
		size_t len = w * h;
		vroi.aux_image_matrix.allocate (w, h);

		// calculate the image matrix or cube 
		vroi.aux_image_matrix.calculate_from_pixelcloud (vroi.raw_pixels, vroi.aabb);

		// calculate features 
		VERBOSLVL2(env.get_verbosity_level(), std::cout << "\treducing whole slide\n");
		reduce_trivial_wholeslide (env, vroi);	// counterpart of segmented reduce_trivial_rois_manual()

		// free buffers of feature helperss
		VERBOSLVL2(env.get_verbosity_level(), std::cout << "\tfreeing vROI buffers\n");
		if (vroi.aux_image_matrix._pix_plane.size())
			std::vector<PixIntens>().swap(vroi.aux_image_matrix._pix_plane);

		// no need to calculate neighbor features in WSI, returning
		return true;
	}

	bool featurize_wholeslide (Environment & env, size_t sidx, ImageLoader& imlo, LR& vroi)
	{
		// phase 1: gather ROI metrics
		VERBOSLVL2(env.get_verbosity_level(), std::cout << "Gathering vROI metrics\n");

		// phase 1: copy ROI metrics from the slide properties, thanks to the WSI scenario
		// instead of gather_wholeslide_metrics (p.fname_int, imlo, vroi)
		vroi.slide_idx = (decltype(vroi.slide_idx)) sidx;
		const SlideProps & p = env.dataset.dataset_props [sidx];
		vroi.aux_area = p.max_roi_area;
		vroi.aabb.init_from_wh (p.max_roi_w, p.max_roi_h);
		// tell ROI the grey levels the loader will store for this slide's range, through the same
		// map the loader itself uses -- so a negative intensity is offset rather than cast straight
		// to PixIntens and wrapped, and a real-valued slide lands on its quantization endpoints.
		vroi.aux_min = (PixIntens) p.to_grey_level (p.min_preroi_inten);
		vroi.aux_max = (PixIntens) p.to_grey_level (p.max_preroi_inten);

		// fix the AABB with respect to anisotropy
		if (env.anisoOptions.customized() == false)
			vroi.aabb.apply_anisotropy(
				env.anisoOptions.get_aniso_x(), 
				env.anisoOptions.get_aniso_y());

		// prepare (zero) ROI's feature value buffer
		vroi.initialize_fvals();

		// assess ROI's memory footprint and check if we can featurize it as phase 2 (trivially) ?
		size_t roiFootprint = vroi.get_ram_footprint_estimate (1),		// 1 since single ROI
			ramLim = env.get_ram_limit();
		if (roiFootprint >= ramLim)
		{
			VERBOSLVL2(env.get_verbosity_level(),
				std::cout << "oversized slide "
				<< " (S=" << vroi.aux_area
				<< " W=" << vroi.aabb.get_width()
				<< " H=" << vroi.aabb.get_height()
				<< " px footprint=" << Nyxus::virguler_ulong(roiFootprint) << " b"
				<< ") while RAM limit is " << Nyxus::virguler_ulong(ramLim) << "\n"
			);

			std::cerr << p.fname_int << ": slide is non-trivial \n";
			return false;
		}

		// phase 2: extract features
		return featurize_triv_wholeslide (env, sidx, imlo, vroi); // segmented counterpart: phase2.cpp / processTrivialRois ()
	}

	void featurize_wsi_thread (
		Environment & env,
		const std::vector<std::string> & intensFiles, 
		const std::vector<std::string> & labelFiles, 
		size_t slide_idx, 
		size_t nf, 
		const std::string & outputPath, 
		bool write_apache, 
		Nyxus::SaveOption saveOption,
		int & rv)
	{
		SlideProps & p = env.dataset.dataset_props [slide_idx];

		// scan one slide
		ImageLoader imlo;
		if (imlo.open(p, env.fpimageOptions) == false)
		{
			std::cerr << "Terminating\n";
			// nothing below can run without the loader, and open() can fail having allocated
			// one half of the pair
			imlo.close();
			rv = 1;
			return;
		}

		LR vroi (1); // virtual ROI representing the whole slide ROI-labelled as '1'

		if (featurize_wholeslide (env, slide_idx, imlo, vroi) == false)	// non-wsi counterpart: processIntSegImagePair()
		{
			std::cerr << "Error featurizing slide " << p.fname_int << " @ " << __FILE__ << ":" << __LINE__ << "\n";
			rv = 1;
			// and no row for it: vroi holds the zeros initialize_fvals() left, so saving it would
			// report measurements nobody made. The 3D twin skips the write the same way.
			imlo.close();
			return;
		}

		// thread-safely save results of this single slide
		if (write_apache) 
		{
			auto [status, msg] = save_features_2_apache_wholeslide (env, vroi, p.fname_int, 0/*t_index*/, 0/*c_index*/);
			if (! status) 
			{
				std::cerr << "Error writing Arrow file: " << msg.value() << std::endl;
				rv = 2;
			}
		}
		else 
			if (saveOption == SaveOption::saveCSV)
			{
				if (save_features_2_csv_wholeslide(env, vroi, p.fname_int, "", outputPath, 0 /*t_index unused in 2D*/, 0 /*c_index unused in 2D*/) == false)
				{
					std::cout << "error saving results to CSV file, details: " << __FILE__ << ":" << __LINE__ << std::endl;
					rv = 2;
				}
			}
			else
			{
				// pulls feature values from 'vroi' and appends them to global object 'theResultsCache' exposed to Python API
				if (save_features_2_buffer_wholeslide (env.theResultsCache, env, vroi, p.fname_int, "", 0/*t_index*/, 0/*c_index*/) == false)
				{
					std::cerr << "Error saving features to the results buffer" << std::endl;
					rv = 2;
				}
			}

		imlo.close();

		//
		// Not saving nested ROI related info because this image is single-ROI (whole-slide)
		//

		// NOT rv = 0: a blanket success here discards every failure recorded above -- the
		// slide that would not open, the one too big for RAM, the row that would not save.
		// rv is 0 already unless one of those set it.
	}

	int processDataset_2D_wholeslide (
		Environment & env,
		const std::vector<std::string>& intensFiles,
		int n_threads,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		env.reset_csv_output_state();		// this run's CSV files start fresh (see Environment::csv_paths_written)

		// create a vector of blank mask file names. Blank mask counterparts
		// of intensity files will serve as the condition of the whole-slide scenario in the prescan phase
		std::vector<std::string> labelFiles (intensFiles.size());

		//**** prescan all slides

		size_t nf = intensFiles.size();
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\nphase 0: prescanning " << nf << " slides \n");
		env.dataset.reset_dataset_props();

		for (size_t i=0; i<nf; i++)
		{
			// slide file names
			// Note: condition (p.fname_seg == "") activates the wholeslide-aware prescanning, 
			// particularly, setting slide's intensity min := 0 (i.e. off-mask background min) 
			// rather than setting min as within-mask min
			SlideProps& p = env.dataset.dataset_props.emplace_back (intensFiles[i], labelFiles[i]);

			// A global user option, recorded per slide because it selects the slide's
			// load-time map (see Nyxus::record_intensity_domain_map).
			p.preserve_hu = env.fpimageOptions.preserve_hu();

			// slide metrics
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "prescanning " << p.fname_int);
			if (! scan_slide_props(p, 2, env.anisoOptions, env.use_physical_spacing(), env.fpimageOptions, env.resultOptions.need_annotation()))
			{
				VERBOSLVL1 (env.get_verbosity_level(), std::cout << "error prescanning pair " << p.fname_int << " and " << p.fname_seg << std::endl);
				return 1;
			}
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << " " << p.slide_w << " W x" << p.slide_h << " H max ROI " << p.max_roi_w << "x" << p.max_roi_h 
				<< " DR " << Nyxus::virguler_real(p.min_preroi_inten) 
				<< "-" << Nyxus::virguler_real(p.max_preroi_inten) 
				<< " " << p.lolvl_slide_descr << "\n");
		}

		// global properties
		env.dataset.update_dataset_props_extrema();

		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\t finished prescanning \n");

		//
		// future: allocate GPU cache for all participating devices
		//

		//**** extract features

		bool write_apache = (saveOption == SaveOption::saveArrowIPC || saveOption == SaveOption::saveParquet);

		// initialize buffer output
		env.theResultsCache.clear();

		// initialize Apache output
		if (write_apache) 
		{
			env.arrow_stream = ArrowOutputStream();
			std::string afn = get_arrow_filename (outputPath, env.nyxus_result_fname, saveOption);
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "arrow file name =" << afn << "\n");
			auto [status, msg] = env.arrow_stream.create_arrow_file (
				saveOption, 
				afn, 
				Nyxus::get_header (env));

			if (!status) 
			{
				std::cerr << "Error creating Arrow file: " << msg.value() << std::endl;
				return 1;
			}
		}

		// run batches of threads
		int worst_rv = 0;	// aggregates per-thread failure so a slide that could not be
								// featurized makes the run fail loudly, as the 3D twin does
		size_t n_jobs = (nf + n_threads - 1) / n_threads;
		for (size_t j=0; j<n_jobs; j++)
		{
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "whole-slide job " << j+1 << "/" << n_jobs << "\n");

			std::vector<std::future<void>> T;
			std::vector<int> rvals(n_threads, 0);
			for (int t=0; t < n_threads; t++)
			{
				size_t idx = j * n_threads + t;

				// done?
				if (idx + 1 > nf)
					break;

				if (n_threads > 1)
				{
					T.push_back(std::async(std::launch::async,
						featurize_wsi_thread,
						std::ref(env),
						intensFiles,
						labelFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						std::ref(rvals[t])));
				}
				else
				{
					featurize_wsi_thread (
						env,
						intensFiles,
						labelFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption,
						rvals[t]);
				}
			}

			// wait for all threads to complete, then take the worst status any slide of this
			// batch reported: rvals was collected and never read, so a failed slide left no
			// trace in the exit code
			for (auto& f : T)
				f.get();
			for (int r : rvals)
				if (r > worst_rv) worst_rv = r;

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
				std::cerr << "Error closing Arrow file: " << msg.value() << std::endl;
				return 2;
			}
		}

		//
		// future: free GPU cache for all participating devices
		//

		// the worst status any slide reported, as the 3D twin returns: a run that could not
		// featurize a slide must not tell the shell it succeeded
		return worst_rv;
	}

}

