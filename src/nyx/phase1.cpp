#include <string>
#include <vector>
#include <fast_loader/specialised_tile_loader/grayscale_tiff_tile_loader.h>
#include <map>
#include <array>
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
			LR& existingData = roiData[label];
			update_label_record_2(existingData, x, y, label, intensity, tile_index);
		}
	}

	bool gatherRoisMetrics(const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// File #1 (intensity)
		GrayscaleTiffTileLoader<uint32_t> I(num_FL_threads, intens_fpath);

		auto th = I.tileHeight(lvl),
			tw = I.tileWidth(lvl),
			td = I.tileDepth(lvl),
			tileSize = th * tw,

			fh = I.fullHeight(lvl),
			fw = I.fullWidth(lvl),
			fd = I.fullDepth(lvl),

			ntw = I.numberTileWidth(lvl),
			nth = I.numberTileHeight(lvl),
			ntd = I.numberTileDepth(lvl);

		// File #2 (labels)
		GrayscaleTiffTileLoader<uint32_t> L(num_FL_threads, label_fpath);

		// -- check whole file consistency
		if (fh != L.fullHeight(lvl) || fw != L.fullWidth(lvl) || fd != L.fullDepth(lvl))
		{
			std::cout << "\terror: mismatch in full height, width, or depth";
			return false;
		}

		// -- check tile consistency
		if (th != L.tileHeight(lvl) || tw != L.tileWidth(lvl) || td != L.tileDepth(lvl))
		{
			std::cout << "\terror: mismatch in tile height, width, or depth";
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
				auto tileIdx = row * fw + col;
				I.loadTileFromFile(ptrI, row, col, lyr, lvl);
				L.loadTileFromFile(ptrL, row, col, lyr, lvl);
				auto& dataI = *ptrI;
				auto& dataL = *ptrL;

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

						feed_pixel_2_metrics(x, y, label, dataI[i], tileIdx);	// Updates 'uniqueLabels' and 'roiData'
					}
				}

				// Show stayalive progress info
				if (cnt++ % 4 == 0)
					std::cout << "\t"
					<< int((row * nth + col) * 100 / float(nth * ntw) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs"
					<< "\n";
			}

		return true;
	}

}