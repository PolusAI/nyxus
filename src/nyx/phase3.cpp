#include <string>
#include <vector>
#include <map>
#include <array>
#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#endif
#include "environment.h"
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
	bool processNontrivialRois (const std::vector<int>& nontrivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
	{
		for (auto lab : nontrivRoiLabels)
		{
			LR& r = roiData[lab];

			VERBOSLVL1(std::cout << "processing oversized ROI " << lab << "\n");

			// Scan one label-intensity pair 
			bool ok = theImLoader.open(intens_fpath, label_fpath);
			if (ok == false)
			{
				std::cout << "Terminating\n";
				return false;
			}
			
			//=== Features permitting raster scan


			// Initialize ROI's pixel cache
			r.raw_pixels_NT.init (r.label, "raw_pixels_NT");

			// Iterate ROI's tiles and scan pixels
			for (auto tileIdx : r.host_tiles)
			{
				theImLoader.load_tile(tileIdx);
				auto& dataI = theImLoader.get_int_tile_buffer();
				auto& dataL = theImLoader.get_seg_tile_buffer();
				for (unsigned long i = 0; i < theImLoader.get_tile_size(); i++)
				{
					auto pixLabel = dataL[i];

					// Skip blanks and other ROI's pixel
					if (pixLabel == 0 || pixLabel != r.label)
						continue;

					// Pixel intensity and global position
					auto intens = dataI[i];
					size_t row = tileIdx / theImLoader.get_num_tiles_hor(),
						col = tileIdx / theImLoader.get_num_tiles_hor(),
						th = theImLoader.get_tile_height(),
						tw = theImLoader.get_tile_width();
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Feed the pixel to online features and helper objects
					r.raw_pixels_NT.add_pixel(Pixel2(x, y, intens));
				}
			}

			//=== Features requiring non-raster access to pixels
			
			// Automatic
			int nrf = theFeatureMgr.get_num_requested_features();
			for (int i = 0; i < nrf; i++)
			{
				auto feature = theFeatureMgr.get_feature_method(i);

				try
				{
					feature->osized_scan_whole_image (r, theImLoader);
				}
				catch (std::exception const& e)
				{
					std::cout << "Error while computing feature " << feature->feature_info << " over oversized ROI " << r.label << " : " << e.what() << "\n";
				}

				feature->cleanup_instance();
			}

			//=== Clean the ROI's cache
			r.raw_pixels_NT.clear();

			#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();
			#endif
		}

		return true;
	}

}
