#include <string>
#include <vector>
#include <fast_loader/specialised_tile_loader/grayscale_tiff_tile_loader.h>
#include <map>
#include <array>
#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#endif
#include "virtual_file_tile_channel_loader.h"
#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

namespace Nyxus
{

	bool processNontrivialRois(const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, size_t memory_limit)
	{
		for (auto lab : uniqueLabels)
		{
			LR& r = roiData[lab];

			// Skip trivial ROI
			//if (r.nontrivial_roi(memory_limit) == false)
				continue;

			//=== Features permitting raster scan

			auto& dataI = theImLoader.get_int_tile_buffer();
			auto& dataL = theImLoader.get_seg_tile_buffer();

			// Initialize ROI's pixel cache
			r.osized_pixel_cloud.init(r.label, "r_oor_pixel_cloud");

			// Iterate ROI's tiles and scan pixels
			for (auto tileIdx : r.host_tiles)
			{
				theImLoader.load_tile(tileIdx);
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
					r.osized_pixel_cloud.add_pixel(Pixel2(x, y, intens));

					// Manual
					//		pixelIntensityFeatures->osized_add_online_pixel(x, y, intens);
					//

					// Automatic
					int nrf = theFeatureMgr.get_num_requested_features();
					for (int i = 0; i < nrf; i++)
					{
						auto feature = theFeatureMgr.get_feature_method(i);
						feature->osized_add_online_pixel (x, y, intens);
					}
				}
			}

			//=== Features requiring non-raster access to pixels
			
			// Manual
			//		contourFeature->osized_scan_whole_image(r, theImLoader);
			//		convhullFeature->osized_scan_whole_image(r, theImLoader);
			//		ellipsefitFeature->osized_scan_whole_image(r, theImLoader);
			//		extremaFeature->osized_scan_whole_image(r, theImLoader);
			//		eulerNumberFeature->osized_scan_whole_image(r, theImLoader);
			//		chordsFeature->osized_scan_whole_image(r, theImLoader);
			//		gaborFeature->osized_scan_whole_image(r, theImLoader);

			// Automatic
			int nrf = theFeatureMgr.get_num_requested_features();
			for (int i = 0; i < nrf; i++)
			{
				auto feature = theFeatureMgr.get_feature_method(i);
				feature->osized_scan_whole_image (r, theImLoader);
				feature->cleanup_instance();
			}

			//=== Clean the ROI's cache
			r.osized_pixel_cloud.clear();

			#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();
			#endif
		}

		return true;
	}

}
