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
	/// @brief Processes so called nontrivial i.e. oversized ROIs - those exceeding certain memory limit
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
			if (! env.theImLoader.open(p, env.fpimageOptions))
			{
				std::cout << "Terminating\n";
				return false;
			}
			
			//=== Features permitting raster scan

			// Initialize ROI's pixel cache
			r.raw_pixels_NT.init (r.label, "raw_pixels_NT");

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

						// Feed the pixel to online features and helper objects
						r.raw_pixels_NT.add_pixel(Pixel2(x, y, intens));
					}
				}

			//=== Features requiring non-raster access to pixels
			
			// Automatic
			int nrf = env.theFeatureMgr.get_num_requested_features();
			for (int i = 0; i < nrf; i++)
			{
				auto f = env.theFeatureMgr.get_feature_method (i);

				try
				{
					const Fsettings& s = env.get_feature_settings (typeid(f));
					f->osized_scan_whole_image (r, s, env.theImLoader);
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
