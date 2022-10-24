#include <string>
#include <sstream>
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
	bool gatherRoisMetrics_2d (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, int z)
	{
		// Clear ROI pixels potentially existing as a result of previous 3D feature extraction
		for (auto label : uniqueLabels)
		{
			LR& lr = roiData[label];
			lr.raw_pixels.clear();
		}
		
		// Read the tiff. The image loader is put in the open state in processDataset()
		size_t nth = theImLoader.get_num_tiles_hor(),
			ntv = theImLoader.get_num_tiles_vert(),
			fw = theImLoader.get_tile_width(),
			th = theImLoader.get_tile_height(),
			tw = theImLoader.get_tile_width(),
			tileSize = theImLoader.get_tile_size(),
			fullwidth = theImLoader.get_full_width(),
			fullheight = theImLoader.get_full_height();

		// Check validity of the z-index 
		int nz = theImLoader.get_num_layers();
		if (z >= nz)
		{
			std::stringstream ss;
			ss << "Error fetching image layer " << z << ": number of image layers = " << nz;
			#ifdef WITH_PYTHON_H
				throw ss.str();
			#endif	
			std::cerr << ss.str() << "\n";
			return false;
		}

		// Middle section requested?
		// (z can be ==-1 (constant -1 defined in class Environment) if the user did not specify z-index for 
		// calculating 2D features of a layer of a 3D image.)
		if (theEnvironment.user_specified_z_index() == false)
			z = nz / 2;

		// Scan all the tiles
		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = theImLoader.load_tile_3d (row, col, z);
				if (!ok)
				{
					std::stringstream ss;
					ss << "Error fetching tile row=" << row << " col=" << col;
					#ifdef WITH_PYTHON_H
						throw ss.str();
					#endif	
					std::cerr << ss.str() << "\n";
					return false;
				}

				// Get ahold of tile's pixel buffer
				auto tileIdx = row * nth + col;
				auto dataI = theImLoader.get_int_tile_buffer(),
					dataL = theImLoader.get_seg_tile_buffer();

				// Iterate pixels
				for (size_t i = 0; i < tileSize; i++)
				{
					// Skip non-mask pixels
					auto label = dataL[i];
					if (!label)
						continue;

					int y = row * th + i / tw,
						x = col * tw + i % tw;
					
					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Collapse all the labels to one if single-ROI mde is requested
					if (theEnvironment.singleROI)
						label = 1;
					
					// Update pixel's ROI metrics
					feed_pixel_2_metrics(x, y, z, dataI[i], label, tileIdx); // Updates 'uniqueLabels' and 'roiData'
				}

				#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
				#endif

				// Show stayalive progress info
				if (cnt++ % 4 == 0)
					std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
			}

		return true;
	}

	bool gatherRoisMetrics_3d (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
	{
		// Clear ROI pixels potentially existing as a result of previous 2D feature extraction
		for (auto label : uniqueLabels)
		{
			LR& lr = roiData[label];
			lr.raw_pixels.clear();
		}

		// Read the tiff. The image loader is put in the open state in processDataset()
		size_t nth = theImLoader.get_num_tiles_hor(),
			ntv = theImLoader.get_num_tiles_vert(),
			fw = theImLoader.get_tile_width(),
			th = theImLoader.get_tile_height(),
			tw = theImLoader.get_tile_width(),
			tileSize = theImLoader.get_tile_size(),
			fullwidth = theImLoader.get_full_width(),
			fullheight = theImLoader.get_full_height();

		// Check validity of the z-index 
		int nz = theImLoader.get_num_layers();

		// Scan all the tiles
		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
				for (int z=0; z<nz; z++)
				{
					// Fetch the tile 
					bool ok = theImLoader.load_tile_3d(row, col, z);
					if (!ok)
					{
						std::stringstream ss;
						ss << "Error fetching tile row=" << row << " col=" << col;
	#ifdef WITH_PYTHON_H
						throw ss.str();
	#endif	
						std::cerr << ss.str() << "\n";
						return false;
					}

					// Get ahold of tile's pixel buffer
					auto tileIdx = row * nth + col;
					auto dataI = theImLoader.get_int_tile_buffer(),
						dataL = theImLoader.get_seg_tile_buffer();

					// Iterate pixels
					for (size_t i = 0; i < tileSize; i++)
					{
						// Skip non-mask pixels
						auto label = dataL[i];
						if (!label)
							continue;

						int y = row * th + i / tw,
							x = col * tw + i % tw;

						// Skip tile buffer pixels beyond the image's bounds
						if (x >= fullwidth || y >= fullheight)
							continue;

						// Collapse all the labels to one if single-ROI mde is requested
						if (theEnvironment.singleROI)
							label = 1;

						// Update pixel's ROI metrics
						feed_pixel_2_metrics(x, y, z, dataI[i], label, tileIdx); // Updates 'uniqueLabels' and 'roiData'
					}

	#ifdef WITH_PYTHON_H
					if (PyErr_CheckSignals() != 0)
						throw pybind11::error_already_set();
	#endif

					// Show stayalive progress info
					if (cnt++ % 4 == 0)
						std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
				}

		return true;
	}


}
