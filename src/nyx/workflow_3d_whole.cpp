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
	// Featurizes a whole volume that fits in RAM. The caller owns the oversized decision -- it sends
	// everything at or above the RAM limit out-of-core before reaching here, off the same estimate,
	// so a second test of it here could never fire.
	// (anisotropic, ax, ay, az) is the voxel spacing featurize_wholevolume resolved for this slide,
	// passed in rather than resolved again here: the out-of-core branch of the caller resamples the
	// voxel cloud onto that same grid, and the box the oversized check is made against was recorded
	// on it, so a second resolution is a second place for the slide's geometry to come from.
	bool featurize_triv_wholevolume (Environment & env, size_t sidx, ImageLoader& imlo, LR& vroi, size_t channel, size_t timeframe,
		bool anisotropic, double ax, double ay, double az)
	{
		const std::string& ifpath = env.dataset.dataset_props[sidx].fname_int;

		// read the slide into a pixel cloud
		// --aniso* (explicit) or opt-in OME physical spacing selects the anisotropic path
		if (! anisotropic)
		{
			VERBOSLVL2(env.get_verbosity_level(), std::cout << "\nscan_trivial_wholeslide()\n");
			if (! scan_trivial_wholevolume (vroi, ifpath, imlo, channel, timeframe))
				return false;
		}
		else
		{
			VERBOSLVL2(env.get_verbosity_level(), std::cout << "\nscan_trivial_wholeslide_ANISO()\n");
			if (! scan_trivial_wholevolume_anisotropic (
				vroi,
				ifpath,
				imlo,
				ax,
				ay,
				az,
				channel,
				timeframe))
				return false;

			// The vROI's extent and voxel count describe the cloud that was just cached.
			// featurize_wholevolume presets them from the PHYSICAL slide dimensions
			// (init_from_whd / p.max_roi_area) before any cloud exists, and the anisotropic scan
			// caches the resampled (virtual) cloud, which is both a different extent and a
			// different voxel count. The extent sizes aux_image_cube, which
			// calculate_from_pixelcloud fills, and aux_area divides every feature that averages.
			// The segmented path recomputes both the same way (processTrivialRois_3D).
			vroi.aabb.update_from_voxelcloud (vroi.raw_pixels_3D);
			vroi.aux_area = (unsigned int) vroi.raw_pixels_3D.size();
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

	void init_wholevolume_vroi (const SlideProps& p, size_t sidx, LR& vroi)
	{
		vroi.slide_idx = (decltype(vroi.slide_idx)) sidx;
		vroi.aux_area = p.max_roi_area;

		// The extent the prescan recorded, taken as it stands: scan_slide_props resolves the same
		// voxel spacing and already recorded these on the resampled grid, so scaling them again
		// here would square the resampling and inflate the oversized check by that factor.
		vroi.aabb.init_from_whd (p.max_roi_w, p.max_roi_h, p.max_roi_d);

		// the grey levels the loader will store for this volume's range, through the same map the
		// loader itself uses -- which offsets only when the volume's own minimum is negative, as a
		// CT's is, instead of shifting every volume to a zero base.
		vroi.aux_min = (PixIntens) p.to_grey_level (p.min_preroi_inten);
		vroi.aux_max = (PixIntens) p.to_grey_level (p.max_preroi_inten);
	}
	bool featurize_wholevolume (Environment & env, size_t sidx, ImageLoader& imlo, LR& vroi, size_t channel, size_t timeframe)
	{
		//***** phase 1: copy ROI metrics from the slide properties, thanks to the WSI scenario
		const SlideProps& p = env.dataset.dataset_props[sidx];
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "Gathering vROI metrics " + fs::path(p.fname_int).filename().string() + "\n");

		// instead of gather_wholeslide_metrics (p.fname_int, imlo, vroi)
		init_wholevolume_vroi (p, sidx, vroi);

		// the spacing this volume is scanned on, resolved once for both branches below (the box
		// above is already on that grid)
		double ax, ay, az;
		bool anisotropic = resolve_slide_anisotropy (env, sidx, ax, ay, az);

		// prepare (zero) ROI's feature value buffer
		vroi.initialize_fvals();

		// assess ROI's memory footprint and check if we can featurize it as phase 2 (trivially) ?
		// the 3D estimator (W*H*D) -- the 2D one ignores depth and under-counts the volume cube.
		// Matches the segmented path.
		size_t roiFootprint = vroi.get_ram_footprint_estimate_3D (1),		// 1 since single-ROI
			ramLim = env.get_ram_limit();
		if (roiFootprint >= ramLim)
		{
			VERBOSLVL1(env.get_verbosity_level(),
				std::cout << "oversized whole volume "
				<< " (S=" << vroi.aux_area
				<< " W=" << vroi.aabb.get_width()
				<< " H=" << vroi.aabb.get_height()
				<< " D=" << vroi.aabb.get_z_depth()
				<< " px footprint=" << Nyxus::virguler_ulong(roiFootprint) << " b"
				<< ") while RAM limit is " << Nyxus::virguler_ulong(ramLim)
				<< " -- streaming out-of-core\n"
			);

			// Out-of-core whole volume: stream every voxel (no mask -- workflow_3d_whole.cpp opens
			// the loader with an empty label path) plane-by-plane instead of holding the whole
			// cube, mirroring the segmented ROI path (processNontrivialRois_3D).
			if (! populate_3d_voxel_cloud (imlo, vroi, channel, timeframe, /*wholevolume=*/ true, ax, ay, az,
				p.fname_int, ""))
			{
				// the segmented path refuses the same way: an empty reason means the pair streams and the
				// refusal came from the shape check, which reports itself inside the streaming call
				const std::string why = ooc_unstreamable_reason (imlo);
				if (! why.empty())
				{
					std::string erm = "Error: cannot featurize whole volume " + p.fname_int
						+ " out-of-core: " + why + ".";
#ifdef WITH_PYTHON_H
					throw std::runtime_error(erm);
#endif
					std::cerr << erm << "\n";
				}
				return false;
			}

			run_3d_ooc_features (env, vroi, imlo);
			vroi.raw_voxels_NT.clear();
			return true;
		}

		//***** phase 2: extract features
		return featurize_triv_wholevolume (env, sidx, imlo, vroi, channel, timeframe, anisotropic, ax, ay, az); // segmented counterpart: phase2.cpp / processTrivialRois ()
	}

	// The per-thread status is returned by VALUE, through the future: a std::async worker's write
	// through std::ref(int) is not reliably visible to the caller after future::get() here, which
	// would lose an oversized slide's rv=1 and report success. Returning it makes future::get()
	// carry the status deterministically.
	int featurize_3d_wv_thread(
		Environment & env,
		const std::vector<std::string>& intensFiles,
		const std::vector<std::string>& labelFiles,
		size_t slide_idx,
		size_t nf,
		const std::string& outputPath,
		bool write_apache,
		Nyxus::SaveOption saveOption)
	{
		int rv = 0;
		SlideProps& p = env.dataset.dataset_props[slide_idx];

		// scan one slide
		ImageLoader imlo;
		if (imlo.open(p, env.fpimageOptions) == false)
		{
			std::cerr << "Terminating\n";
			rv = 1;
			// nothing below can run without the loader: every pass reads through it, and
			// ImageLoader::open() can fail having allocated one half of the pair
			imlo.close();
			return rv;
		}

		// featurize every (channel, timeframe) plane and emit one whole-slide row
		// per plane, tagged with c_index/t_index — mirroring the segmented path. A
		// single-channel, single-timeframe slide reports 1/1, so its output is unchanged.
		for (size_t c = 0; c < p.inten_channels; c++)
		  for (size_t t = 0; t < p.inten_time; t++)
		  {
			LR vroi(1); // virtual ROI representing the whole slide ROI-labelled as '1'

			if (featurize_wholevolume (env, slide_idx, imlo, vroi, c, t) == false)	// non-wsi counterpart: processIntSegImagePair()
			{
				// featurize_wholevolume returns false when this plane's volume could not be read or
				// featurized, and has already reported why (raised under Python, printed under the
				// CLI). Its feature buffer holds no values, so no row is written for it.
				rv = 1;
				continue;	// do NOT emit a (misleading all-zero) row for this plane
			}

			// thread-safely save results of this single (slide, channel, timeframe)
			if (write_apache)
			{
				auto [status, msg] = save_features_2_apache_wholeslide (env, vroi, p.fname_int, t, c);
				if (!status)
				{
					std::cerr << "Error writing Arrow file: " << msg.value() << std::endl;
					rv = 2;
				}
			}
			else
				if (saveOption == SaveOption::saveCSV)
				{
					if (save_features_2_csv_wholeslide (env, vroi, p.fname_int, "", outputPath, t, c) == false)
					{
						std::cout << "error saving results to CSV file, details: " << __FILE__ << ":" << __LINE__ << std::endl;
						rv = 2;
					}
				}
				else
				{
					// pulls feature values from 'vroi' and appends them to global object 'theResultsCache' exposed to Python API
					if (save_features_2_buffer_wholeslide (env.theResultsCache, env, vroi, p.fname_int, "", t, c) == false)
					{
						std::cerr << "Error saving features to the results buffer" << std::endl;
						rv = 2;
					}
				}
		  } //- channels x timeframes

		imlo.close();

		//
		// Not saving nested ROI related info because this image is single-ROI (whole-slide)
		//

		return rv;
	}



	std::tuple<bool, std::optional<std::string>> processDataset_3D_wholevolume (
		Environment & env,
		const std::vector <std::string>& intensFiles,
		int n_threads,
		const SaveOption saveOption,
		const std::string& outputPath)
	{
		env.reset_csv_output_state();		// this run's CSV files start fresh (see Environment::csv_paths_written)

		//**** prescan all files

		size_t nf = intensFiles.size();
		VERBOSLVL1 (env.get_verbosity_level(), std::cout << "\nphase 0: prescanning " << nf << " slides \n");
		env.dataset.reset_dataset_props();

		for (size_t i = 0; i < nf; i++)
		{
			// slide file names
			SlideProps& p = env.dataset.dataset_props.emplace_back(intensFiles[i], "");

			// A global user option, recorded per slide because it selects the slide's
			// load-time map (see Nyxus::record_intensity_domain_map).
			p.preserve_hu = env.fpimageOptions.preserve_hu();

			// slide metrics
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "prescanning " << fs::path(p.fname_int).filename().string());

			if (! scan_slide_props(p, 3, env.anisoOptions, env.use_physical_spacing(), env.fpimageOptions, env.resultOptions.need_annotation()))
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

		int worst_rv = 0;	// aggregates per-thread failure so an oversized/failed slide
							// makes the whole run fail loudly (nonzero exit) instead of returning
							// success -- the per-thread rvals were collected but never checked.
		size_t n_jobs = (nf + n_threads - 1) / n_threads;
		for (size_t j = 0; j < n_jobs; j++)
		{
			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "whole-slide job " << j + 1 << "/" << n_jobs << "\n");

			std::vector<std::future<int>> T;
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
						saveOption));
				}
				else
				{
					// aggregate the returned status
					int r = featurize_3d_wv_thread(
						env,
						intensFiles,
						intensFiles,
						idx,
						nf,
						outputPath,
						write_apache,
						saveOption);
					if (r > worst_rv) worst_rv = r;
				}
			}

			// wait for all threads to complete, aggregating each one's returned status so an
			// oversized/failed slide makes the whole run fail loudly (nonzero exit)
			for (auto& f : T)
			{
				int r = f.get();
				if (r > worst_rv) worst_rv = r;
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

		// a slide that could not be featurized is a hard failure, not a silent success --
		// report it so the CLI exits nonzero and the Python API raises. An oversized whole volume
		// streams out-of-core now (see featurize_wholevolume), so this only fires when that
		// streaming itself failed (input format can't deliver plane-by-plane, or an unsupported
		// feature was requested) -- the specific reason was already reported above.
		if (worst_rv != 0)
			return { false, "one or more slides could not be featurized (see errors above; "
				"an unsupported format needs a segmented mask, or a smaller/plane-deliverable input)" };

		return {true, std::nullopt}; // success
	}


}