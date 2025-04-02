#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <array>
#include <regex>

#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
	#include <pybind11/stl.h>
	#include <pybind11/numpy.h>
	namespace py = pybind11;
#endif

#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

namespace Nyxus
{
	bool gatherRoisMetrics (const std::string & intens_fpath, const std::string & label_fpath, ImageLoader & L)
	{
		// Reset per-image counters and extrema
		//	 -- disabling this due to new prescan functionality-->	LR::reset_global_stats();

		int lvl = 0, // Pyramid level
			lyr = 0; //	Layer

		// Read the tiff. The image loader is put in the open state in processDataset()
		size_t nth = L.get_num_tiles_hor(),
			ntv = L.get_num_tiles_vert(),
			fw = L.get_tile_width(),
			th = L.get_tile_height(),
			tw = L.get_tile_width(),
			tileSize = L.get_tile_size(),
			fullwidth = L.get_full_width(),
			fullheight = L.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = L.load_tile(row, col);
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
				const std::vector<uint32_t>& dataI = L.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = L.get_seg_tile_sptr();
				bool wholeslide = spL == nullptr; // alternatively, theEnvironment.singleROI

				// Iterate pixels
				for (size_t i = 0; i < tileSize; i++)
				{
					// mask label if not in the wholeslide mode
					PixIntens label = 1;
					if (!wholeslide)
						label = (*spL)[i];

					// Skip non-mask pixels
					if (! label)
						continue;

					int y = row * th + i / tw,
						x = col * tw + i % tw;
					
					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Update pixel's ROI metrics
					feed_pixel_2_metrics (x, y, dataI[i], label); // Updates 'uniqueLabels' and 'roiData'
				}

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

				// Show progress info
				VERBOSLVL2(
					if (cnt++ % 4 == 0)
						std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
				);
			}

		// fix ROIs' AABBs with respect to anisotropy
		if (theEnvironment.anisoOptions.empty())
		{
			for (auto& rd : roiData)
			{
				LR& r = rd.second;
				r.make_nonanisotropic_aabb ();
			}
		}
		else
		{
			double	ax = theEnvironment.anisoOptions.get_aniso_x(),
						ay = theEnvironment.anisoOptions.get_aniso_y();

			for (auto& rd : roiData)
			{
				LR& r = rd.second;
				r.make_anisotropic_aabb (ax, ay);
			}
		}

		return true;
	}

	bool gather_wholeslide_metrics (const std::string& intens_fpath, ImageLoader& L, LR & roi)
	{
		PixIntens minI = std::numeric_limits<PixIntens>::max(), 
			maxI = std::numeric_limits<PixIntens>::min();
		AABB a;

		int lvl = 0, // Pyramid level
			lyr = 0; //	Layer

		// Read the tiff. The image loader is put in the open state in processDataset()
		size_t nth = L.get_num_tiles_hor(),
			ntv = L.get_num_tiles_vert(),
			fw = L.get_tile_width(),
			th = L.get_tile_height(),
			tw = L.get_tile_width(),
			tileSize = L.get_tile_size(),
			fullwidth = L.get_full_width(),
			fullheight = L.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = L.load_tile(row, col);
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
				const std::vector<uint32_t>& dataI = L.get_int_tile_buffer();

				// Iterate pixels
				for (size_t i = 0; i < tileSize; i++)
				{
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Update whole slide's vROI metrics
					PixIntens I = dataI[i];
					minI = std::min (minI, I);
					maxI = std::max (maxI, I);
				}

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

				// Show progress info
				VERBOSLVL2(
					if (cnt++ % 4 == 0)
						std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
				);
			}

		// Per-ROI 
		roi.aux_area = fullwidth * fullheight;
		roi.aux_min = minI;
		roi.aux_max = maxI;
		roi.aabb.init_from_widthheight (fullwidth, fullheight);

		return true;
	}

	//
	// theImLoader needs to be pre-opened!
	//
	bool gatherRoisMetrics_3D (const std::string& intens_fpath, const std::string& mask_fpath, const std::vector<std::string>& z_indices)
	{
		// Cache the file names in global variables '' to be picked up 
		// by labels in feed_pixel_2_metrics_3D() to link ROIs with their image file origins
		theIntFname = intens_fpath;
		theSegFname = mask_fpath;

		int lvl = 0, // Pyramid level
			lyr = 0; //	Layer

		for (size_t z=0; z<z_indices.size(); z++)
		{ 
			// prepare the physical file 
			// 
			// ifile and mfile contain a placeholder for the z-index. We need to turn them to physical filesystem files
			auto zValue = z_indices[z];	// realistic dataset's z-values may be arbitrary (non-zer-based and non-contiguous), so use the actual value
			std::string ifpath = std::regex_replace (intens_fpath, std::regex("\\*"), zValue),
				mfpath = std::regex_replace (mask_fpath, std::regex("\\*"), zValue);

			// temp SlideProps object
			SlideProps sprp;
			sprp.fname_int = ifpath;
			sprp.fname_seg = mfpath;

			// Extract features from this intensity-mask pair 
			if (theImLoader.open(sprp) == false)
			{
				std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
				return false;
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

			int cnt = 1;
			for (unsigned int row = 0; row < nth; row++)
				for (unsigned int col = 0; col < ntv; col++)
				{
					// Fetch a tile 
					bool ok = theImLoader.load_tile (row, col);
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
						feed_pixel_2_metrics_3D  (x, y, z, dataI[i], label); // Updates 'uniqueLabels' and 'roiData'
					}

					#ifdef WITH_PYTHON_H
					if (PyErr_CheckSignals() != 0)
						throw pybind11::error_already_set();
					#endif

					// Show stayalive progress info
					VERBOSLVL2(
						if (cnt++ % 4 == 0)
							std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
					);
				}

			theImLoader.close();
		}

		// fix ROIs' AABBs with respect to anisotropy
		if (theEnvironment.anisoOptions.empty())
		{
			for (auto& rd : roiData)
			{
				LR& r = rd.second;
				r.make_nonanisotropic_aabb();
			}
		}
		else
		{
			double	ax = theEnvironment.anisoOptions.get_aniso_x(),
				ay = theEnvironment.anisoOptions.get_aniso_y(), 
				az = theEnvironment.anisoOptions.get_aniso_z();

			for (auto& rd : roiData)
			{
				LR& r = rd.second;
				r.make_anisotropic_aabb (ax, ay, az);
			}
		}

		return true;
	}


#ifdef WITH_PYTHON_H
	bool gatherRoisMetricsInMemory (const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_images, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images, int start_idx)
	{
		auto intens_buffer = intens_images.request();

		unsigned int* dataL = static_cast<unsigned int*>(label_images.request().ptr);
		unsigned int* dataI = static_cast<unsigned int*>(intens_buffer.ptr);
		
		auto width = intens_buffer.shape[1];
   		auto height = intens_buffer.shape[2];

		for (int row = 0; row < width; row++)
			for (int col = 0; col < height; col++)
			{
				// Skip non-mask pixels
				auto label = dataL[start_idx + row * height + col];
				if (!label)
					continue;


				// Collapse all the labels to one if single-ROI mde is requested
				if (theEnvironment.singleROI)
					label = 1;
				
				// Update pixel's ROI metrics
				feed_pixel_2_metrics (row, col, dataI[start_idx + row * height + col], label); // Updates 'uniqueLabels' and 'roiData'
				

				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
			}

		return true;
	}
#endif
}