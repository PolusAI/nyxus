#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <array>
#include <regex>

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

	bool scanTrivialRois (
		const std::vector<int>& batch_labels, 
		const std::string& intens_fpath, 
		const std::string& label_fpath, 
		Environment & env,
		ImageLoader & ldr)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort (whiteList.begin(), whiteList.end());

		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs
		size_t nth = ldr.get_num_tiles_hor(),
			ntv = ldr.get_num_tiles_vert(),
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = ldr.load_tile(row, col);
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
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = ldr.get_seg_tile_sptr();
				bool wholeslide = spL == nullptr; // alternatively, theEnvironment.singleROI

				// Iterate pixels
				for (unsigned long i = 0; i < tileSize; i++)
				{
					// mask label if not in the wholeslide mode
					PixIntens label = 1;
					if (!wholeslide)
						label = (*spL)[i];

					// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
					if (! env.singleROI && ! std::binary_search(whiteList.begin(), whiteList.end(), label))
						continue;

					auto inten = dataI[i];
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Cache this pixel 
					LR& r = env.roiData [label];
					feed_pixel_2_cache_LR (x, y, dataI[i], r);
				}

				VERBOSLVL2 (env.get_verbosity_level(),
					// Show stayalive progress info
					if (cnt++ % 4 == 0)
					{
						static int prevIntPc = 0;
						float pc = int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100.;
						if (int(pc) != prevIntPc)
						{
							std::cout << "\t scan trivial " << int(pc) << " %\n";
							prevIntPc = int(pc);
						}
					}
				);
			}

		return true;
	}

	bool scanTrivialRois_anisotropic (
		const std::vector<int>& batch_labels,
		const std::string& intens_fpath,
		const std::string& label_fpath,
		Environment& env,
		ImageLoader& ldr,
		double sf_x, 
		double sf_y)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		int lvl = 0,	// pyramid level
			lyr = 0;	//	layer

		// physical slide properties
		size_t nth = ldr.get_num_tiles_hor(),
			ntv = ldr.get_num_tiles_vert(),
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		// virtual slide properties
		size_t vh = (size_t) (double(fullheight) * sf_y),
			vw = (size_t) (double(fullwidth) * sf_x),
			vth = (size_t)(double(th) * sf_y),
			vtw = (size_t)(double(tw) * sf_x);

		// current tile to skip tile reloads
		size_t curt_x = 999, curt_y = 999;

		for (size_t vr = 0; vr < vh; vr++)
		{
			for (size_t vc = 0; vc < vw; vc++)
			{
				// tile position
				size_t tidx_y = size_t(vr / vth),
					tidx_x = size_t(vc / vtw);

				// load it
				if (tidx_y != curt_y || tidx_x != curt_x)
				{
					bool ok = ldr.load_tile(tidx_y, tidx_x);
					if (!ok)
					{
						std::string s = "Error fetching tile row=" + std::to_string(tidx_y) + " col=" + std::to_string(tidx_x);
#ifdef WITH_PYTHON_H
						throw s;
#endif	
						std::cerr << s << "\n";
						return false;
					}

					// cache tile position to avoid reloading
					curt_y = tidx_y;
					curt_x = tidx_x;
				}

				// within-tile virtual pixel position
				size_t vx = vc - tidx_x * vtw,
					vy = vr - tidx_y * vth;

				// within-tile physical pixel position
				size_t ph_x = size_t (double(vx) / sf_x), 
					ph_y = size_t (double(vy) / sf_y),
					i = ph_y * tw + ph_x;

				// read buffered physical pixel 
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = ldr.get_seg_tile_sptr();
				bool wholeslide = spL == nullptr; // alternatively, theEnvironment.singleROI

				PixIntens label = 1;
				if (!wholeslide)
					label = (*spL)[i];

				// not a ROI ?
				if (!label)
					continue;

				// skip this ROI if the label isn't in the to-do list 'whiteList' that's only possible in multi-ROI mode
				if (wholeslide==false && !std::binary_search(whiteList.begin(), whiteList.end(), label))
					continue;

				auto inten = dataI[i];

				// cache this pixel 
				// (ROI 'label' is known to the cache by means of gatherRoisMetrics() called previously.)
				LR& r = env.roiData [label];
				feed_pixel_2_cache_LR (vc, vr, inten, r);
			}
		}

		return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi'
	//
	bool scan_trivial_wholeslide (
		LR & vroi,
		const std::string& intens_fpath,
		ImageLoader& ldr)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs
		size_t nth = ldr.get_num_tiles_hor(),
			ntv = ldr.get_num_tiles_vert(),
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = ldr.load_tile(row, col);
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
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();

				// Iterate pixels
				for (unsigned long i = 0; i < tileSize; i++)
				{
					auto inten = dataI[i];
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Cache this pixel 
					feed_pixel_2_cache_LR (x, y, dataI[i], vroi);
				}
			}

			return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi' 
	// performing anisotropy correction
	//
	bool scan_trivial_wholeslide_anisotropic (
		LR& vroi,
		const std::string& intens_fpath,
		ImageLoader& ldr,
		double aniso_x,
		double aniso_y)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// physical slide properties
		size_t nth = ldr.get_num_tiles_hor(),
			ntv = ldr.get_num_tiles_vert(),
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		// virtual slide properties
		size_t vh = (size_t)(double(fullheight) * aniso_y),
			vw = (size_t)(double(fullwidth) * aniso_x),
			vth = (size_t)(double(th) * aniso_y),
			vtw = (size_t)(double(tw) * aniso_x);

		// current tile to skip tile reloads
		size_t curt_x = 999, curt_y = 999;

		for (size_t vr = 0; vr < vh; vr++)
		{
			for (size_t vc = 0; vc < vw; vc++)
			{
				// tile position for virtual pixel (vc, vr)
				size_t tidx_y = size_t(vr / vth),
					tidx_x = size_t(vc / vtw);

				// load it
				if (tidx_y != curt_y || tidx_x != curt_x)
				{
					bool ok = ldr.load_tile(tidx_y, tidx_x);
					if (!ok)
					{
						std::string s = "Error fetching tile row=" + std::to_string(tidx_y) + " col=" + std::to_string(tidx_x);
#ifdef WITH_PYTHON_H
						throw s;
#endif	
						std::cerr << s << "\n";
						return false;
					}

					// cache tile position to avoid reloading
					curt_y = tidx_y;
					curt_x = tidx_x;
				}

				// within-tile virtual pixel position
				size_t vx = vc - tidx_x * vtw,
					vy = vr - tidx_y * vth;

				// within-tile physical pixel position
				size_t ph_x = size_t(double(vx) / aniso_x),
					ph_y = size_t(double(vy) / aniso_y),
					i = ph_y * tw + ph_x;

				// read buffered physical pixel 
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = ldr.get_seg_tile_sptr();
				bool wholeslide = spL == nullptr; // alternatively, theEnvironment.singleROI

				// Cache this pixel 
				feed_pixel_2_cache_LR (vc, vr, dataI[i], vroi);
			}
		}

		return true;
	}

	void allocateTrivialRoisBuffers (const std::vector<int>& Pending, Roidata& roiData, CpusideCache& cache)
	{
		// Calculate the total memory demand (in # of items) of all segments' image matrices
		cache.imageMatrixBufferLen = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];

			size_t w = r.aabb.get_width(), 
				h = r.aabb.get_height(), 
				imatrSize = w * h;
			cache.imageMatrixBufferLen += imatrSize;

			cache.largest_roi_imatr_buf_len = cache.largest_roi_imatr_buf_len == 0 ? imatrSize : std::max (cache.largest_roi_imatr_buf_len, imatrSize);
		}

		// Lagest ROI
		cache.largest_roi_imatr_buf_len = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			cache.largest_roi_imatr_buf_len = cache.largest_roi_imatr_buf_len ? std::max(cache.largest_roi_imatr_buf_len, r.raw_pixels.size()) : r.raw_pixels.size();
		}

		// Allocate image matrices and remember each ROI's image matrix offset in 'ImageMatrixBuffer'
		size_t baseIdx = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];

			// matrix data
			size_t imatrSize = r.aabb.get_width() * r.aabb.get_height();
			r.aux_image_matrix.allocate (r.aabb.get_width(), r.aabb.get_height());
			baseIdx += imatrSize;	

			// Calculate the image matrix or cube 
			r.aux_image_matrix.calculate_from_pixelcloud (r.raw_pixels, r.aabb);
		}
	}

	void freeTrivialRoisBuffers (const std::vector<int>& roi_labels, Roidata& roiData)
	{
		// Dispose memory of ROIs having their feature calculation finished 
		// in order to give memory ROIs of the next ROI batch. 
		// (Vector 'Pending' is the set of ROIs of the finished batch.)
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			std::vector<Pixel2>().swap(r.raw_pixels);
			std::vector<PixIntens>().swap(r.aux_image_matrix._pix_plane);
			std::vector<Pixel2>().swap(r.convHull_CH);	// convex hull is not a large object but there's no point to keep it beyond the batch, unlike contour
		}
	}

	void freeTrivialRoisBuffers_3D (const std::vector<int>& roi_labels, Roidata& roiData)
	{
		// Dispose memory of ROIs having their feature calculation finished 
		// in order to give memory ROIs of the next ROI batch. 
		// (Vector 'Pending' is the set of ROIs of the finished batch.)
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			std::vector<Pixel3>().swap(r.raw_pixels_3D);

			//
			// Deallocate image matrices, cubes, and convex shells here (in the future).
			//
		}

		// Dispose the buffer of batches' ROIs' image matrices. (We allocate the 
		// image matrix buffer externally to minimize host-GPU transfers.)
//		delete ImageMatrixBuffer;
	}

	bool processTrivialRois (Environment & env, const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit)
	{
		std::vector<int> Pending;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = env.roiData[lab];

			size_t itemFootprint = r.get_ram_footprint_estimate (trivRoiLabels.size());

			// Check if we are good to accumulate this ROI in the current batch or should close the batch and reduce it
			if (batchDemand + itemFootprint < memory_limit)
			{
				// There is room in the ROI batch. Insert another ROI in it
				Pending.push_back(lab);
				batchDemand += itemFootprint;
			}
			else
			{
				// The ROI batch is full. Let's process it 
				std::sort(Pending.begin(), Pending.end());
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of total " << env.uniqueLabels.size() << " ROIs\n");
				VERBOSLVL2(env.get_verbosity_level(),
					if (Pending.size() == 1)
						std::cout << ">>> (single ROI label " << Pending[0] << ")\n";
					else
						std::cout << ">>> (ROI labels " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				);

				if (env.anisoOptions.customized() == false)
					scanTrivialRois (Pending, intens_fpath, label_fpath, env, env.theImLoader);
				else
				{
					double ax = env.anisoOptions.get_aniso_x(), 
						ay = env.anisoOptions.get_aniso_y();
					scanTrivialRois_anisotropic (Pending, intens_fpath, label_fpath, env, env.theImLoader, ax, ay);
				}

				// Allocate memory
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tallocating ROI buffers\n");
				allocateTrivialRoisBuffers (Pending, env.roiData, env.hostCache);

				// Reduce them
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\treducing ROIs\n");
				// reduce_trivial_rois(Pending);	
				reduce_trivial_rois_manual (Pending, env);

				// Free memory
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tfreeing ROI buffers\n");
				freeTrivialRoisBuffers (Pending, env.roiData);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()

				// Reset the RAM footprint accumulator
				batchDemand = 0;

				// Clear the freshly processed ROIs from pending list 
				Pending.clear();

				// Start a new pending set by adding the batch-overflowing ROI 
				Pending.push_back(lab);

				// Advance the batch counter
				roiBatchNo++;
			}

			// Allow keyboard interrupt
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
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of " << env.uniqueLabels.size() << " all ROIs\n");
			VERBOSLVL2 (env.get_verbosity_level(),
				if (Pending.size() == 1)
					std::cout << ">>> (single ROI " << Pending[0] << ")\n";
				else
					std::cout << ">>> (ROIs " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				);

				if (env.anisoOptions.customized() == false)
				{
					scanTrivialRois (Pending, intens_fpath, label_fpath, env, env.theImLoader);
				}
				else
				{
					double	ax = env.anisoOptions.get_aniso_x(), 
								ay = env.anisoOptions.get_aniso_y();
					scanTrivialRois_anisotropic (Pending, intens_fpath, label_fpath, env, env.theImLoader, ax, ay);
				}

			// Allocate memory
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tallocating ROI buffers\n");
			allocateTrivialRoisBuffers (Pending, env.roiData, env.hostCache);

			// Dump ROIs for use in unit testing
#ifdef DUMP_ALL_ROI
			dump_all_roi();
#endif

			// Reduce them
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\treducing ROIs\n");
			//reduce_trivial_rois(Pending):
			reduce_trivial_rois_manual (Pending, env);

			// Free memory
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tfreeing ROI buffers\n");
			freeTrivialRoisBuffers (Pending, env.roiData);

			#ifdef WITH_PYTHON_H
			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
			#endif
		}

		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\treducing neighbor features and their depends for all ROIs\n");
		reduce_neighbors_and_dependencies_manual (env);

		return true;
	}

#ifdef WITH_PYTHON_H

	bool scanTrivialRoisInMemory (
		const std::vector<int>& batch_labels, 
		const py::array_t<unsigned int, 
		py::array::c_style | py::array::forcecast>& intens_images, 
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images, 
		int pair_idx,
		Environment & env)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		auto rI = intens_images.unchecked<3>();
		auto rL = label_images.unchecked<3>();
		size_t width = rI.shape(2);
		size_t height = rI.shape(1);

		int cnt = 1;
		for (size_t col = 0; col < width; col++)
			for (size_t row = 0; row < height; row++)
			{
				// Skip non-mask pixels
				auto label = rL (pair_idx, row, col);
				if (!label)
					continue;

				// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
				if (! env.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
					continue;

				auto inten = rI (pair_idx, row, col);

				// Collapse all the labels to one if single-ROI mde is requested
				if (env.singleROI)
					label = 1;

				// Cache this pixel 
				LR& r = env.roiData [label];
				feed_pixel_2_cache_LR (col, row, inten, r);
			}

		return true;
	}

	bool processTrivialRoisInMemory (Environment& env, const std::vector<int>& trivRoiLabels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label, int pair_idx, size_t memory_limit)
	{
		VERBOSLVL4(env.get_verbosity_level(), std::cout << "processTrivialRoisInMemory (pair_idx=" << pair_idx << ") \n");

		std::vector<int> Pending;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = env.roiData[lab];

			size_t itemFootprint = r.get_ram_footprint_estimate(env.uniqueLabels.size());

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
				scanTrivialRoisInMemory (Pending, intens, label, pair_idx, env);

				// Allocate memory
				allocateTrivialRoisBuffers (Pending, env.roiData, env.hostCache);

				// reduce_trivial_rois(Pending);	
				reduce_trivial_rois_manual (Pending, env);

				// Free memory
				freeTrivialRoisBuffers (Pending, env.roiData);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()

				// Reset the RAM footprint accumulator
				batchDemand = 0;

				// Clear the freshly processed ROIs from pending list 
				Pending.clear();

				// Start a new pending set by adding the stopper ROI 
				Pending.push_back(lab);

				// Advance the batch counter
				roiBatchNo++;
			}

			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();

		}

		// Process what's remaining pending
		if (Pending.size() > 0)
		{
			// Scan pixels of pending trivial ROIs 
			std::sort(Pending.begin(), Pending.end());
			scanTrivialRoisInMemory(Pending, intens, label, pair_idx, env);

			// Allocate memory
			allocateTrivialRoisBuffers (Pending, env.roiData, env.hostCache);

			// Dump ROIs for use in unit testing
#ifdef DUMP_ALL_ROI
			dump_all_roi();
#endif

			// Reduce them
			reduce_trivial_rois_manual (Pending, env);

			// Free memory
			freeTrivialRoisBuffers (Pending, env.roiData);

			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
		}

		reduce_neighbors_and_dependencies_manual (env);

		return true;

	}

#endif // python api

}
