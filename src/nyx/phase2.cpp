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

	void feed_pixel_2_cache(int x, int y, int label, PixIntens intensity, unsigned int tile_index)
	{
		// Update basic ROI info (info that doesn't require costly calculations)
		LR& r = roiData[label];
		r.raw_pixels.push_back(Pixel2(x, y, intensity));
	}

	bool scanTrivialRois(const std::vector<int>& PendingRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
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

					// Skip this ROI if it's label isn't in the pending set
					if (std::find(PendingRoiLabels.begin(), PendingRoiLabels.end(), label) == PendingRoiLabels.end())
						continue;

					auto inten = dataI[i];
					if (label != 0)
					{
						int y = row * th + i / tw,
							x = col * tw + i % tw;

						// Collapse all the labels to one if single-ROI mde is requested
						if (theEnvironment.singleROI)
							label = 1;

						feed_pixel_2_cache(x, y, label, dataI[i], tileIdx);
					}
				}

				// Show stayalive progress info
				if (cnt++ % 4 == 0)
					VERBOSLVL1(std::cout << "\t"
					<< int((row * nth + col) * 100 / float(nth * ntw) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs"
					<< "\n";)
			}

		return true;
	}

	void allocateTrivialRoisBuffers(const std::vector<int>& Pending)
	{
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			r.aux_image_matrix.use_roi(r.raw_pixels, r.aabb);
		}
	}

	void freeTrivialRoisBuffers(const std::vector<int>& Pending)
	{
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			r.raw_pixels.clear();
			r.aux_image_matrix.clear();
		}
	}

	bool processTrivialRois (const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, size_t memory_limit)
	{
		std::vector<int> Pending;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = roiData[lab];

			size_t itemFootprint = r.get_ram_footprint_estimate();

			// Sheck if we are good to accumulate this ROI in the current batch or should close the batch and reduce it
			if (batchDemand + itemFootprint < memory_limit)
			{
				Pending.push_back(lab);
				batchDemand += itemFootprint;
			}
			else
			{
				// Scan pixels of pending trivial ROIs 
				std::sort (Pending.begin(), Pending.end());
				VERBOSLVL1(std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of " << uniqueLabels.size() << " all ROIs\n";)
				VERBOSLVL1(
					if (Pending.size() ==1)					
						std::cout << ">>> (single ROI " << Pending[0] << ")\n";
					else
						std::cout << ">>> (ROIs " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
					)
				scanTrivialRois(Pending, intens_fpath, label_fpath, num_FL_threads);

				// Allocate memory
				VERBOSLVL1(std::cout << "\tallocating ROI buffers\n";)
				allocateTrivialRoisBuffers(Pending);

				// Reduce them
				VERBOSLVL1(std::cout << "\treducing ROIs\n";)
				// reduce_trivial_rois(Pending);	
				reduce_trivial_rois_manual(Pending);

				// Output results
				//outputRoisFeatures (Pending);

				// Free memory
				VERBOSLVL1(std::cout << "\tfreeing ROI buffers\n";)
				freeTrivialRoisBuffers(Pending);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()

				// Reset the RAM footprint accumulator
				batchDemand = 0;

				// Clear the freshly processed ROIs from pending list 
				Pending.clear();

				// Start a new pending set by adding the stopper ROI 
				Pending.push_back(lab);

				// Advance the batch counter
				roiBatchNo++;
			}

			// Allow heyboard interrupt.

#ifdef WITH_PYTHON_H
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();
#endif
		}

		// Process what's remaining pending
		if (Pending.size() > 0)
		{
			// Scan pixels of pending trivial ROIs 
			std::sort (Pending.begin(), Pending.end());
			VERBOSLVL1(std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of " << uniqueLabels.size() << " all ROIs\n";)
			VERBOSLVL1(
				if (Pending.size() == 1)
					std::cout << ">>> (single ROI " << Pending[0] << ")\n";
				else
					std::cout << ">>> (ROIs " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				)
			scanTrivialRois(Pending, intens_fpath, label_fpath, num_FL_threads);

			// Allocate memory
			VERBOSLVL1(std::cout << "\tallocating ROI buffers\n";)
			allocateTrivialRoisBuffers(Pending);

			// Reduce them
			VERBOSLVL1(std::cout << "\treducing ROIs\n";)
			//reduce_trivial_rois(Pending);	
			reduce_trivial_rois_manual(Pending);

			// Output results
			//outputRoisFeatures(Pending);

			// Free memory
			VERBOSLVL1(std::cout << "\tfreeing ROI buffers\n";)
			freeTrivialRoisBuffers(Pending);

			#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();
			#endif
		}

		return true;
	}

}