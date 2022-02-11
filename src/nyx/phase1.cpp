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

	/// @brief Updates 'uniqueLabels' and 'roiData'
	/// @param x
	/// @param y
	/// @param label
	/// @param intensity
	/// @param tile_index
	void feed_pixel_2_metrics(int x, int y, int label, PixIntens intensity, unsigned int tile_index)
	{
		if (uniqueLabels.find(label) == uniqueLabels.end())
		{
			// Remember this label
			uniqueLabels.insert(label);

			// Initialize the ROI label record
			LR newData;
			init_label_record_2(newData, theSegFname, theIntFname, x, y, label, intensity, tile_index);
			roiData[label] = newData;
		}
		else
		{
			// Update basic ROI info (info that doesn't require costly calculations)
			LR &existingData = roiData[label];
			update_label_record_2(existingData, x, y, label, intensity, tile_index);
		}
	}

	bool gatherRoisMetrics (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
	{
		int lvl = 0, // Pyramid level
			lyr = 0; //	Layer

		// Open an image pair
		ImageLoader imlo;
		bool ok = imlo.open (intens_fpath, label_fpath);
		if (!ok)
		{
			std::stringstream ss;
			ss << "Error opening file pair " << intens_fpath << " : " << label_fpath;
			#ifdef WITH_PYTHON_H
				throw ss.str();
			#endif	
			std::cerr << ss.str() << "\n";
			return false;
		}

		// Read the tiff
		size_t nth = imlo.get_num_tiles_hor(),
			ntv = imlo.get_num_tiles_vert(), 
			fw = imlo.get_tile_width(), 
			th = imlo.get_tile_height(), 
			tw = imlo.get_tile_width(),
			tileSize = imlo.get_tile_size();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				ok = imlo.load_tile(row, col);
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
				auto tileIdx = row * fw + col;
				auto dataI = imlo.get_int_tile_buffer(),
					dataL = imlo.get_seg_tile_buffer();

				for (size_t i = 0; i < tileSize; i++)
				{
					// Skip non-mask pixels
					auto label = dataL[i];
					if (label != 0)
					{
						int y = row * th + i / tw,
							x = col * tw + i % tw;
					
						// Collapse all the labels to one if single-ROI mde is requested
						if (theEnvironment.singleROI)
							label = 1;
					
						// Update pixel's ROI metrics
						feed_pixel_2_metrics (x, y, label, dataI[i], tileIdx); // Updates 'uniqueLabels' and 'roiData'
					}
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