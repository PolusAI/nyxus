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
				for (unsigned long i = 0; i < tileSize; i++)
				{
					// Skip non-mask pixels
					auto label = dataL[i];
					if (! label)
						continue;

					// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
					if (! theEnvironment.singleROI && ! std::binary_search(whiteList.begin(), whiteList.end(), label))
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

				VERBOSLVL2(				
					// Show stayalive progress info
					if (cnt++ % 4 == 0)
					{
							static int prevIntPc = 0;
							float pc = int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. ;
							if (int(pc) != prevIntPc)
							{
								std::cout << "\t scan trivial " << int(pc) << " %\n";
								prevIntPc = int(pc);
							}
					} 
				)
			}

		return true;
	}

#ifdef WITH_PYTHON_H
	bool scanTrivialRoisInMemory (const std::vector<int>& batch_labels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_images, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images, int start_idx)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort (whiteList.begin(), whiteList.end());


		auto intens_buffer = intens_images.request();
		auto label_buffer = label_images.request();

		auto width = intens_buffer.shape[1];
		auto height = intens_buffer.shape[2];

		unsigned int* dataL = static_cast<unsigned int*>(label_buffer.ptr);
		unsigned int* dataI = static_cast<unsigned int*>(intens_buffer.ptr);

		int cnt = 1;
		for (unsigned int row = 0; row < width; row++)
			for (unsigned int col = 0; col < height; col++)
			{

				// Skip non-mask pixels
				auto label = dataL[start_idx + row*height + col];
				if (! label)
					continue;

				// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
				if (! theEnvironment.singleROI && ! std::binary_search(whiteList.begin(), whiteList.end(), label))
					continue;

				auto inten = dataI[start_idx + row*height + col];

				// Collapse all the labels to one if single-ROI mde is requested
				if (theEnvironment.singleROI)
					label = 1;

				// Cache this pixel 
				feed_pixel_2_cache (row, col, inten, label);
				
			}

		return true;
	}
#endif

	// Objects that are used by GPU code to transfer image matrices of all the image's ROIs
	PixIntens* ImageMatrixBuffer = nullptr;	// Solid buffer of all the image matrices in the image
	size_t imageMatrixBufferLen = 0;		// Combined size of all ROIs' image matrices in the image
	size_t largest_roi_imatr_buf_len = 0;

	void allocateTrivialRoisBuffers(const std::vector<int>& Pending)
	{
		// Calculate the total memory demand (in # of items) of all segments' image matrices
		imageMatrixBufferLen = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			size_t w = r.aabb.get_width(), 
				h = r.aabb.get_height(), 
				imatrSize = w * h;
			imageMatrixBufferLen += imatrSize;

			largest_roi_imatr_buf_len = largest_roi_imatr_buf_len == 0 ? imatrSize : std::max (largest_roi_imatr_buf_len, imatrSize);
		}

		ImageMatrixBuffer = new PixIntens[imageMatrixBufferLen];

		// Lagest ROI
		largest_roi_imatr_buf_len = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			largest_roi_imatr_buf_len = largest_roi_imatr_buf_len ? std::max(largest_roi_imatr_buf_len, r.raw_pixels.size()) : r.raw_pixels.size();
		}

		// Allocate image matrices and remember each ROI's image matrix offset in 'ImageMatrixBuffer'
		size_t baseIdx = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];

			// matrix data offset
			r.im_buffer_offset = baseIdx;

			// matrix data
			size_t imatrSize = r.aabb.get_width() * r.aabb.get_height();
			r.aux_image_matrix.bind_to_buffer(ImageMatrixBuffer + baseIdx, ImageMatrixBuffer + baseIdx + imatrSize);
			baseIdx += imatrSize;	

			// Calculate the image matrix in the external buffer
			r.aux_image_matrix.calculate_from_pixelcloud(r.raw_pixels, r.aabb);
		}
	}

	void freeTrivialRoisBuffers(const std::vector<int>& Pending)
	{
		// Dispose memory of ROIs having their feature calculation finished 
		// in order to give memory ROIs of the next ROI batch. 
		// (Vector 'Pending' is the set of ROIs of the finished batch.)
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			std::vector<Pixel2>().swap(r.raw_pixels);
			std::vector<PixIntens>().swap(r.aux_image_matrix._pix_plane);
			std::vector<Pixel2>().swap(r.convHull_CH);	// convex hull is not a large object but there's no point to keep it beyond the batch, unlike contour
		}

		// Dispose the buffer of batches' ROIs' image matrices. (We allocate the 
		// image matrix buffer externally to minimize host-GPU transfers.)
		delete ImageMatrixBuffer;
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

			// Check if we are good to accumulate this ROI in the current batch or should close the batch and reduce it
			if (batchDemand + itemFootprint < memory_limit)
			{
				Pending.push_back(lab);
				batchDemand += itemFootprint;
			}
			else
			{
				// Scan pixels of pending trivial ROIs 
				std::sort(Pending.begin(), Pending.end());
				VERBOSLVL1(std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of total " << uniqueLabels.size() << " ROIs\n";)
					VERBOSLVL1(
						if (Pending.size() == 1)
							std::cout << ">>> (single ROI label " << Pending[0] << ")\n";
						else
							std::cout << ">>> (ROI labels " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				)
					scanTrivialRois(Pending, intens_fpath, label_fpath, num_FL_threads);

				// Allocate memory
				VERBOSLVL1(std::cout << "\tallocating ROI buffers\n";)
					allocateTrivialRoisBuffers(Pending);

				// Reduce them
				VERBOSLVL1(std::cout << "\treducing ROIs\n";)
					// reduce_trivial_rois(Pending);	
					reduce_trivial_rois_manual(Pending);

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
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
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

			// Dump ROIs for use in unit testing
#ifdef DUMP_ALL_ROI
			dump_all_roi();
#endif

			// Reduce them
			VERBOSLVL1(std::cout << "\treducing ROIs\n";)
			//reduce_trivial_rois(Pending);	
			reduce_trivial_rois_manual(Pending);

			// Free memory
			VERBOSLVL1(std::cout << "\tfreeing ROI buffers\n";)
			freeTrivialRoisBuffers(Pending);

			#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
			#endif
		}

		VERBOSLVL1(std::cout << "\treducing neighbor features and their depends for all ROIs\n")
		reduce_neighbors_and_dependencies_manual();

		return true;
	}


#ifdef WITH_PYTHON_H
	bool processTrivialRoisInMemory (const std::vector<int>& trivRoiLabels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label, int start_idx, size_t memory_limit)
	{	
		std::vector<int> Pending;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = roiData[lab];

			size_t itemFootprint = r.get_ram_footprint_estimate();

			// Check if we are good to accumulate this ROI in the current batch or should close the batch and reduce it
			if (batchDemand + itemFootprint < memory_limit)
			{
				Pending.push_back(lab);
				batchDemand += itemFootprint;
			}
			else
			{
				// Scan pixels of pending trivial ROIs 
				std::sort (Pending.begin(), Pending.end());

				scanTrivialRoisInMemory(Pending, intens, label, start_idx);

				// Allocate memory
				allocateTrivialRoisBuffers (Pending);

				// reduce_trivial_rois(Pending);	
				reduce_trivial_rois_manual(Pending);

				// Free memory
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
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();

		}

		// Process what's remaining pending
		if (Pending.size() > 0)
		{
			// Scan pixels of pending trivial ROIs 
			std::sort (Pending.begin(), Pending.end());
			
			scanTrivialRoisInMemory(Pending, intens, label, start_idx);

			// Allocate memory
			allocateTrivialRoisBuffers(Pending);

			// Dump ROIs for use in unit testing
#ifdef DUMP_ALL_ROI
			dump_all_roi();
#endif

			// Reduce them
			reduce_trivial_rois_manual(Pending);

			// Free memory
			freeTrivialRoisBuffers(Pending);

			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
                throw pybind11::error_already_set();
		}

		reduce_neighbors_and_dependencies_manual();

		return true;

	}
#endif
}
