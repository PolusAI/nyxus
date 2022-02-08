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

	bool gatherRoisMetrics(const std::string &intens_fpath, const std::string &label_fpath, int num_FL_threads)
	{
		int lvl = 0, // Pyramid level
			lyr = 0; //	Layer

		// File #1 (intensity)
		GrayscaleTiffTileLoader<uint32_t> I(num_FL_threads, intens_fpath);

		auto th = I.tileHeight(lvl),
			 tw = I.tileWidth(lvl),
			 td = I.tileDepth(lvl),
			 tileSize = th * tw,

			 fh_int = I.fullHeight(lvl),
			 fw_int = I.fullWidth(lvl),
			 fd_int = I.fullDepth(lvl),

			 ntw = I.numberTileWidth(lvl),
			 nth = I.numberTileHeight(lvl),
			 ntd = I.numberTileDepth(lvl);

		// File #2 (labels)
		GrayscaleTiffTileLoader<uint32_t> L(num_FL_threads, label_fpath);
		auto fh_lab = L.fullHeight(lvl),
			fw_lab = L.fullWidth(lvl), 
			fd_lab = L.fullDepth(lvl);

		VERBOSLVL1(std::cout << "\tINT: " << intens_fpath << " [" << fw_int << "w x " << fh_int << "h] SEG: " << label_fpath << " [" << fw_int << "w x " << fh_int << "h]\n");

		// -- check whole file consistency
		if (fh_int != fh_lab || fw_int != fw_lab || fd_int != fd_lab)
		{
			#ifdef WITH_PYTHON_H
				throw "Error: mismatch in full height, width, or depth between the mask and intensity images";
			#endif
			std::cerr << "Error: mismatch in full height, width, or depth between the mask and intensity images \n";
			return false;
		}

		// -- check tile consistency
		if (th != L.tileHeight(lvl) || tw != L.tileWidth(lvl) || td != L.tileDepth(lvl))
		{
			#ifdef WITH_PYTHON_H
				throw "Error: mismatch in tile height, width, or depth between the mask and intensity images";
			#endif			
			std::cerr << "Error: mismatch in tile height, width, or depth between the mask and intensity images \n";
			return false;
		}

		// Read the TIFF tile by tile
		//
		// -- allocate the tile buffer
		std::shared_ptr<std::vector<uint32_t>> ptrI = std::make_shared<std::vector<uint32_t>>(tileSize);
		std::shared_ptr<std::vector<uint32_t>> ptrL = std::make_shared<std::vector<uint32_t>>(tileSize);

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntw; col++)
			{
				auto tileIdx = row * fw_int + col;
				I.loadTileFromFile(ptrI, row, col, lyr, lvl);
				L.loadTileFromFile(ptrL, row, col, lyr, lvl);
				auto &dataI = *ptrI;
				auto &dataL = *ptrL;

				for (unsigned long i = 0; i < tileSize; i++)
				{
					auto label = dataL[i];
					auto inten = dataI[i];
					if (label != 0)
					{
						int y = row * th + i / tw,
							x = col * tw + i % tw;

						// Collapse all the labels to one if single-ROI mde is requested
						if (theEnvironment.singleROI)
							label = 1;

						feed_pixel_2_metrics(x, y, label, dataI[i], tileIdx); // Updates 'uniqueLabels' and 'roiData'
					}
				}

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

				// Show stayalive progress info
				if (cnt++ % 10 == 0)
					std::cout << "\tgathered "
							  << int((row * nth + col) * 100 / float(nth * ntw) * 100) / 100. << "% image\t found " << uniqueLabels.size() << " ROIs"
							  << "\n";
			}

		return true;
	}

}
