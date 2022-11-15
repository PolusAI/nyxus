#include <fstream>
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
#include "rle.hpp"

namespace Nyxus
{
	#define disable_DUMP_ALL_ROI
	#ifdef DUMP_ALL_ROI
	void dump_all_roi()
	{
		std::string fpath = theEnvironment.output_dir + "/all_roi.txt";
		std::cout << "Dumping all the ROIs to " << fpath << " ...\n";

		std::ofstream f(fpath);

		for (auto lab : uniqueLabels)
		{
			auto& r = roiData[lab];
			std::cout << "Dumping ROI " << lab << "\n";

			r.aux_image_matrix.print(f);

			f << "ROI " << lab << ": \n"
				<< "xmin = " << r.aabb.get_xmin() << "; \n"
				<< "width=" << r.aabb.get_width() << "; \n"
				<< "ymin=" << r.aabb.get_ymin() << "; \n"
				<< "height=" << r.aabb.get_height() << "; \n"
				<< "area=" << r.aux_area << "; \n";

			// C++ constant:
			f << "// C:\n"
				<< "struct NyxusPixel {\n"
				<< "\tsize_t x, y; \n"
				<< "\tunsigned int intensity; \n"
				<< "}; \n"
				<< "NyxusPixel testData[] = {\n";
			for (auto i=0; i<r.raw_pixels.size(); i++)
			{
				auto& px = r.raw_pixels[i];
				f << "\t{" << px.x-r.aabb.get_xmin() << ", " << px.y- r.aabb.get_ymin() << ", " << px.inten << "}, ";
				if (i > 0 && i % 4 == 0)
					f << "\n";
			}
			f << "}; \n";

			// Matlab constant:
			f << "// MATLAB:\n"
				<< "%==== begin \n";
			f << "pixelCloud = [ \n";
			for (auto i = 0; i < r.raw_pixels.size(); i++)
			{
				auto& px = r.raw_pixels[i];
				f << px.inten << "; % [" << i << "] \n";
			}
			f << "]; \n";

			f << "testData = zeros(" << r.aabb.get_height() << "," << r.aabb.get_width() << ");\n";
			for (auto i = 0; i < r.raw_pixels.size(); i++)
			{
				auto& px = r.raw_pixels[i];
				f << "testData(" << (px.y - r.aabb.get_ymin() + 1) << "," << (px.x - r.aabb.get_xmin() + 1) << ")=" << px.inten << "; ";	// +1 due to 1-based nature of Matlab
				if (i > 0 && i % 4 == 0)
					f << "\n";
			}
			f << "\n";
			f << "testVecZ = reshape(testData, 1, []); \n";
			f << "testVecNZ = nonzeros(testData); \n";
			f << "[mean(testVecNZ) mean(testVecZ) mean2(testData)] \n";
			f << "%==== end \n";
		}

		f.flush();
	}
	#endif

	bool scanTrivialRois (const std::vector<int>& batch_labels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort (whiteList.begin(), whiteList.end());

		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs
		size_t nth = theImLoader.get_num_tiles_hor(),
			ntv = theImLoader.get_num_tiles_vert(),
			fw = theImLoader.get_tile_width(),
			th = theImLoader.get_tile_height(),
			tw = theImLoader.get_tile_width(),
			tileSize = theImLoader.get_tile_size(),
			fullwidth = theImLoader.get_full_width(),
			fullheight = theImLoader.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				bool ok = theImLoader.load_tile(row, col);
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
				auto dataI = theImLoader.get_int_tile_buffer(),
					dataL = theImLoader.get_seg_tile_buffer();

				// Iterate pixels
				for (unsigned long i = 0; i < tileSize; i++)
				{
					// Skip non-mask pixels
					auto label = dataL[i];
					if (! label)
						continue;

					// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
					if (! theEnvironment.singleROI && ! std::binary_search(whiteList.begin(), whiteList.end(), label)) //--slow-- if (std::find(PendingRoiLabels.begin(), PendingRoiLabels.end(), label) == PendingRoiLabels.end())
						continue;

					auto inten = dataI[i];
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Collapse all the labels to one if single-ROI mde is requested
					if (theEnvironment.singleROI)
						label = 1;

					// Cache this pixel 
					feed_pixel_2_cache (x, y, dataI[i], label);
				}

				// Show stayalive progress info
				if (cnt++ % 4 == 0)
					VERBOSLVL1(std::cout << "\tscan trivial " << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "% of image scanned \n";)
			}

		return true;
	}

	bool scanTrivialRoisFast (const std::vector<int>& batch_labels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort (whiteList.begin(), whiteList.end());

		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs
		size_t nth = theImLoader.get_num_tiles_hor(),
			ntv = theImLoader.get_num_tiles_vert(),
			fw = theImLoader.get_tile_width(),
			th = theImLoader.get_tile_height(),
			tw = theImLoader.get_tile_width(),
			tileSize = theImLoader.get_tile_size(),
			fullwidth = theImLoader.get_full_width(),
			fullheight = theImLoader.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
		{
			// Fetch the tile 
			bool ok = theImLoader.load_tile(row, col);
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
			auto dataI = theImLoader.get_int_tile_buffer(),
				dataL = theImLoader.get_seg_tile_buffer();

			// Iterate pixels
			int y = 0;
			int y_max = (fullheight < y+th) ? fullheight - row*th : th;
			while (y < y_max)
			{

				size_t i = y*tw;
				
				int x = 0;
				int x_max = (fullwidth < x+tw) ? fullwidth - col*tw : tw;
				

				int x_stream = 32 * (x_max / 32);

				RLEStream<uint32_t, uint16_t> stream = rle_encode_long_stream_32(dataL.data()+i,x_stream);

				unsigned int ind = 0;
				while (ind+1 < stream.offsets.size()) {
					uint16_t x1 = stream.offsets[ind++];
					auto label = dataL[i+x1];

					if (label > 0) {
						uint16_t x2 = stream.offsets[ind];

						if (! theEnvironment.singleROI && ! std::binary_search(whiteList.begin(), whiteList.end(), label)) //--slow-- if (std::find(PendingRoiLabels.begin(), PendingRoiLabels.end(), label) == PendingRoiLabels.end())
						{
							continue;
						}

						for (int xi=x1; xi<x2; xi++) {

							feed_pixel_2_cache (xi, y, dataI[xi+i], label);

						}

						feed_rle_2_cache(x1, x2, dataI[x1+i], dataI[x2+i], y, label);
					}
				}

				i += x_stream;
				x = x_stream;

				stream = rle_encode_stream(dataL.data() + i, (uint16_t) (x_max - x_stream));

				ind = 0;
				while (ind+1 < stream.offsets.size()) {
					uint16_t x1 = stream.offsets[ind++];
					auto label = dataL[i+x1];

					if (label > 0) {
						uint16_t x2 = stream.offsets[ind];

						if (! theEnvironment.singleROI && ! std::binary_search(whiteList.begin(), whiteList.end(), label)) //--slow-- if (std::find(PendingRoiLabels.begin(), PendingRoiLabels.end(), label) == PendingRoiLabels.end())
						{
							continue;
						}

						for (int xi=x1; xi<x2; xi++) {

							feed_pixel_2_cache (xi+x, y, dataI[xi+i], label);

						}

						feed_rle_2_cache(x1+x, x2+x, dataI[x1+i], dataI[x2+i], y, label);
					}
				}

				++y;
			}

			// Show stayalive progress info
			if (cnt++ % 4 == 0)
				VERBOSLVL1(std::cout << "\tscan trivial " << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "% of image scanned \n";)
		}

		return true;
	}

	PixIntens* ImageMatrixBuffer = nullptr;
	size_t imageMatrixBufferLen = 0;

	void allocateTrivialRoisBuffers(const std::vector<int>& Pending)
	{
		// Calculate the total memory demand (in # of items) of all segments' image matrices
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			imageMatrixBufferLen += r.aabb.get_width() * r.aabb.get_height();
		}

		ImageMatrixBuffer = new PixIntens[imageMatrixBufferLen];

		// Allocate image matrices and remember each ROI's image matrix offset in 'ImageMatrixBuffer'
		size_t baseIdx = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];

			// matrix data offset
			r.im_buffer_offset = baseIdx;

			// matrix data
			size_t imgLen = r.aabb.get_width() * r.aabb.get_height();
			r.aux_image_matrix.bind_to_buffer(ImageMatrixBuffer + baseIdx, ImageMatrixBuffer + baseIdx + imgLen);
			baseIdx += imgLen;

			// Calculate the image matrix
			r.aux_image_matrix.calculate_from_pixelcloud(r.raw_pixels, r.aabb);
		}
	}

	void freeTrivialRoisBuffers(const std::vector<int>& Pending)
	{
		delete ImageMatrixBuffer;
	}

	bool processTrivialRois (const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads, size_t memory_limit, bool use_fastloop)
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
				if (use_fastloop) {
					scanTrivialRoisFast(Pending, intens_fpath, label_fpath, num_FL_threads);
				} else {
					scanTrivialRois(Pending, intens_fpath, label_fpath, num_FL_threads);
				}

				// Allocate memory
				VERBOSLVL1(std::cout << "\tallocating ROI buffers\n";)
				allocateTrivialRoisBuffers (Pending);

				// Reduce them
				VERBOSLVL1(std::cout << "\treducing ROIs\n";)
				// reduce_trivial_rois(Pending);	
				reduce_trivial_rois_manual(Pending);

				// Free memory
				VERBOSLVL1(std::cout << "\tfreeing ROI buffers\n";)
				freeTrivialRoisBuffers (Pending);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()

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
				if (use_fastloop) {
					scanTrivialRoisFast(Pending, intens_fpath, label_fpath, num_FL_threads);
				} else {
					scanTrivialRois(Pending, intens_fpath, label_fpath, num_FL_threads);
				}

			// Allocate memory
			VERBOSLVL1(std::cout << "\tallocating ROI buffers\n";)
			allocateTrivialRoisBuffers(Pending);

			// Dump ROIs for use in unit testing
#ifdef DUMP_ALL_ROI
			dump_all_roi();
#endif

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
