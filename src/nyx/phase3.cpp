#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <map>
#include <array>

#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
#endif

#include "environment.h"
#include "feature_mgr.h"
#include "globals.h"
#include "helpers/timing.h"

namespace Nyxus
{
	// Whether 'f' computes a feature that 'selection' enables
	static bool provides_any (FeatureMethod* f, const FeatureSet& selection)
	{
		for (int c = (int) Feature2D::_FIRST_; c < (int) Feature2D::_COUNT_; c++)
			if (selection.isEnabled ((Feature2D) c) && f->provides (c))
				return true;
		for (int c = (int) FeatureIMQ::_FIRST_; c < (int) FeatureIMQ::_COUNT_; c++)
			if (selection.isEnabled ((FeatureIMQ) c) && f->provides (c))
				return true;
		return false;
	}

	// Streams the ROI's pixels resampled by (ax, ay) into r.raw_pixels_NT -- every virtual pixel below
	// (size_t)(extent * factor) carries the physical pixel its coordinate truncates back to, as the
	// in-RAM anisotropic scan maps them -- and takes the ROI's box and pixel count from that cloud.
	// Refuses a ROI the resampling leaves with no pixel, as the in-RAM pass does.
	static bool stream_resampled_roi_2d (ImageLoader& ldr, LR& r, double ax, double ay)
	{
		const size_t tw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			vw = (size_t) (double(ldr.get_full_width()) * ax),
			vh = (size_t) (double(ldr.get_full_height()) * ay);

		r.raw_pixels_NT.init (r.label, "raw_pixels_NT");
		size_t curt_x = SIZE_MAX, curt_y = SIZE_MAX, n = 0;
		for (size_t vr = 0; vr < vh; vr++)
			for (size_t vc = 0; vc < vw; vc++)
			{
				const size_t ph_col = (size_t) (double(vc) / ax),
					ph_row = (size_t) (double(vr) / ay),
					tidx_x = ph_col / tw,
					tidx_y = ph_row / th;
				if (tidx_y != curt_y || tidx_x != curt_x)
				{
					if (! ldr.load_tile (tidx_y, tidx_x))
					{
						std::string erm = "Error fetching tile row=" + std::to_string(tidx_y) + " col=" + std::to_string(tidx_x);
#ifdef WITH_PYTHON_H
						throw std::runtime_error (erm);
#endif
						std::cerr << erm << "\n";
						return false;
					}
					curt_y = tidx_y;
					curt_x = tidx_x;
				}

				const size_t i = ldr.get_within_tile_idx (ph_row, ph_col);
				if (ldr.get_seg_tile_buffer()[i] != (uint32_t) r.label)
					continue;

				r.raw_pixels_NT.add_pixel (Pixel2 ((StatsInt) vc, (StatsInt) vr, ldr.get_int_tile_buffer()[i]));
				if (n++ == 0)
				{
					r.aabb.init_x ((StatsInt) vc);
					r.aabb.init_y ((StatsInt) vr);
				}
				else
				{
					r.aabb.update_x ((StatsInt) vc);
					r.aabb.update_y ((StatsInt) vr);
				}
			}

		if (n == 0)
		{
			std::string erm = "Error: ROI " + std::to_string(r.label) + " maps to no pixel at anisotropy "
				+ std::to_string(ax) + "," + std::to_string(ay) + ", so its shape cannot be measured";
#ifdef WITH_PYTHON_H
			throw std::runtime_error (erm);
#endif
			std::cerr << erm << "\n";
			return false;
		}
		r.aux_area = (unsigned int) n;
		return true;
	}

	/// @brief Processes so called nontrivial i.e. oversized ROIs - those exceeding certain memory limit.
	/// An anisotropic ROI is streamed twice, as a trivial batch is scanned twice (processTrivialRois):
	/// the families defined on the image grid run over the pixels as acquired, then the geometric
	/// families over the ROI resampled by the factors, with its box and pixel count taken from that
	/// cloud.
	/// @param intens_fpath Intensity image path
	/// @param label_fpath Mask image path
	/// @param num_FL_threads Number of threads of FastLoader based TIFF tile browser
	/// @param memory_limit RAM limit in bytes
	/// @return Success status
	/// 
	bool processNontrivialRois (Environment & env, const std::vector<int>& nontrivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath)
	{
		// Sort labels for reproducibility with function's trivial counterpart. Nontrivial part of the workflow isn't time-critical anyway
		auto L = nontrivRoiLabels;
		std::sort (L.begin(), L.end());

		for (auto lab : L)
		{
			LR& r = env.roiData[lab];

			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "processing oversized ROI " << lab << "\n");

			// Scan one label-intensity pair
			SlideProps p (intens_fpath, label_fpath);
			// carry over what the prescan measured and recorded for this slide, so this pass's
			// grey levels land in the same domain the features will report them in
			const SlideProps * scanned = env.dataset.scanned_slide (r.slide_idx);
			if (scanned)
				p.inherit_intensity_domain (*scanned);
			if (! env.theImLoader.open(p, env.fpimageOptions))
			{
				std::cout << "Terminating\n";
				return false;
			}
			
			//=== Features permitting raster scan

			// Initialize ROI's pixel cache
			r.raw_pixels_NT.init (r.label, "raw_pixels_NT");

			// Recompute the ROI's intensity extrema from the pixels actually streamed here, so each
			// out-of-core feature's degenerate-ROI guard (aux_min == aux_max) keys off this ROI's real
			// pixels instead of a scalar carried over from phase 1. A segmented ROI's extrema equal its
			// pixel min/max, so ordinary ROIs are unchanged; this only corrects a stale scalar.
			r.aux_min = (std::numeric_limits<PixIntens>::max)();
			r.aux_max = 0;

			// Iterate ROI's tiles and scan pixels
			size_t nth = env.theImLoader.get_num_tiles_hor(),
				ntv = env.theImLoader.get_num_tiles_vert();
			for (unsigned int row = 0; row < nth; row++)
				for (unsigned int col = 0; col < ntv; col++)
				{
					unsigned int tileIdx = row * ntv + col;
					env.theImLoader.load_tile(tileIdx);
					auto& dataI = env.theImLoader.get_int_tile_buffer();
					auto& dataL = env.theImLoader.get_seg_tile_buffer();
					for (unsigned long i = 0; i < env.theImLoader.get_tile_size(); i++)
					{
						auto pixLabel = dataL[i];

						// Skip blanks and other ROI's pixel
						if (pixLabel == 0 || pixLabel != r.label)
							continue;

						// Pixel intensity and global position
						auto intens = dataI[i];
						size_t row = tileIdx / env.theImLoader.get_num_tiles_hor(),
							col = tileIdx / env.theImLoader.get_num_tiles_hor(),
							th = env.theImLoader.get_tile_height(),
							tw = env.theImLoader.get_tile_width();
						int y = row * th + i / tw,
							x = col * tw + i % tw;

						// Track the streamed extrema (see the aux_min/aux_max reset above)
						r.aux_min = (std::min)(r.aux_min, (PixIntens)intens);
						r.aux_max = (std::max)(r.aux_max, (PixIntens)intens);

						// Feed the pixel to online features and helper objects
						r.raw_pixels_NT.add_pixel(Pixel2(x, y, intens));
					}
				}

			//=== Features requiring non-raster access to pixels

			// which methods run over the pixels as acquired, and which over the resampled ROI
			const bool anisotropic = env.anisoOptions.customized();
			FeatureSet on_grid, geometric;
			if (anisotropic)
				split_2d_selection (env.theFeatureSet, on_grid, geometric);

			auto run_methods = [&](const FeatureSet* only)
			{
				int nrf = env.theFeatureMgr.get_num_requested_features();
				for (int i = 0; i < nrf; i++)
				{
					auto f = env.theFeatureMgr.get_feature_method (i);
					if (only && ! provides_any (f, *only))
						continue;

					try
					{
						// typeid(*f), not typeid(f): f is a FeatureMethod*, whose static type is the same
						// for every feature, so the pointer's type_info resolves no family. Dereferencing
						// gives the dynamic type, which is what the settings are registered under.
						const Fsettings& s = env.get_feature_settings (typeid(*f));
						// Pass the Dataset so intensity/histogram osized features reach their
						// Dataset-aware osized_calculate; the Dataset-less overload is a guard that throws.
						f->osized_scan_whole_image (r, s, env.dataset, env.theImLoader);
					}
					catch (std::exception const& e)
					{
						std::string erm = "Error while computing feature " + f->feature_info + " over oversized ROI " + std::to_string(r.label) + " : " + e.what();
#ifdef WITH_PYTHON_H
						throw std::runtime_error(erm);
#endif
						std::cerr << erm << "\n";
					}

					f->cleanup_instance();
				}
			};

			run_methods (anisotropic ? &on_grid : nullptr);

			if (anisotropic && geometric.numOfEnabled (2))
			{
				r.raw_pixels_NT.clear();
				if (! stream_resampled_roi_2d (env.theImLoader, r, env.anisoOptions.get_aniso_x(), env.anisoOptions.get_aniso_y()))
				{
					r.raw_pixels_NT.clear();
					env.theImLoader.close();
					return false;
				}
				run_methods (&geometric);
			}

			//=== Clean the ROI's cache
			r.raw_pixels_NT.clear();

			#ifdef WITH_PYTHON_H
			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();
			#endif
		}

		return true;
	}

}
