#include <string>
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
	//
	// segmented 2D case
	//
	bool gatherRoisMetrics (int sidx, const std::string & intens_fpath, const std::string & label_fpath, Environment & env, ImageLoader & L)
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
					std::string erm = "Error fetching tile row:" + std::to_string(row) + " col:" + std::to_string(col) + " from I:" + intens_fpath + " M:" + label_fpath;
					#ifdef WITH_PYTHON_H
						throw erm;
					#endif	
					std::cerr << erm << "\n";
					return false;
				}

				// Get ahold of tile's pixel buffer
				const std::vector<uint32_t>& dataI = L.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = L.get_seg_tile_sptr();
				bool wholeslide = spL == nullptr; 

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
					feed_pixel_2_metrics (env.uniqueLabels, env.roiData, x, y, dataI[i], label, sidx);
				}

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

				// Show progress info
				VERBOSLVL2 (env.get_verbosity_level(),
					if (cnt++ % 4 == 0)
						std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << env.uniqueLabels.size() << " ROIs" << "\n";
				);
			}

		// fix ROIs' AABBs with respect to anisotropy
		if (env.anisoOptions.customized() == false)
		{
			for (auto& rd : env.roiData)
			{
				LR& r = rd.second;
				r.make_nonanisotropic_aabb ();
			}
		}
		else
		{
			double	ax = env.anisoOptions.get_aniso_x(),
						ay = env.anisoOptions.get_aniso_y();

			for (auto& rd : env.roiData)
			{
				LR& r = rd.second;
				r.make_anisotropic_aabb (ax, ay);
			}
		}

		return true;
	}

	//
	// segmented 2.5D case (aka layoutA, collections of 2D slice images e.g. blah_z1_blah.ome.tif, blah_z2_blah.ome.tif, ..., blah_z500_blah.ome.tif)
	// prerequisite: 'env.theImLoader' needs to be pre-opened !
	//
	bool gatherRoisMetrics_25D (Environment & env, size_t sidx, const std::string& intens_fpath, const std::string& mask_fpath, const std::vector<std::string>& z_indices)
	{
		for (size_t z=0; z<z_indices.size(); z++)
		{ 
			// prepare the physical file 
			// 
			// ifile and mfile contain a placeholder for the z-index. We need to turn them to physical filesystem files
			auto zValue = z_indices[z];	// realistic dataset's z-values may be arbitrary (non-zer-based and non-contiguous), so use the actual value
			std::string ifpath = std::regex_replace (intens_fpath, std::regex("\\*"), zValue),
				mfpath = std::regex_replace (mask_fpath, std::regex("\\*"), zValue);

			// temp SlideProps object
			SlideProps sprp (ifpath, mfpath);

			// Extract features from this intensity-mask pair 
			if (env.theImLoader.open(sprp, env.fpimageOptions) == false)
			{
				std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
				return false;
			}

			// Read the tiff. The image loader is put in the open state in processDataset()
			size_t nth = env.theImLoader.get_num_tiles_hor(),
				ntv = env.theImLoader.get_num_tiles_vert(),
				fw = env.theImLoader.get_tile_width(),
				th = env.theImLoader.get_tile_height(),
				tw = env.theImLoader.get_tile_width(),
				tileSize = env.theImLoader.get_tile_size(),
				fullwidth = env.theImLoader.get_full_width(),
				fullheight = env.theImLoader.get_full_height();

			int cnt = 1;
			for (unsigned int row = 0; row < nth; row++)
				for (unsigned int col = 0; col < ntv; col++)
				{
					// Fetch a tile 
					bool ok = env.theImLoader.load_tile (row, col);
					if (!ok)
					{
						std::string erm = "Error fetching tile row:" + std::to_string(row) + " col:" + std::to_string(col) + " from I:" + ifpath + " M:" + mfpath;
						#ifdef WITH_PYTHON_H
							throw erm;
						#endif	
						std::cerr << erm << "\n";
						return false;
					}

					// Get ahold of tile's pixel buffer
					auto dataI = env.theImLoader.get_int_tile_buffer(),
						dataL = env.theImLoader.get_seg_tile_buffer();

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
						if (env.singleROI)
							label = 1;

						// Update pixel's ROI metrics
						feed_pixel_2_metrics_3D  (env.uniqueLabels, env.roiData, x, y, z, dataI[i], label, sidx); // Updates 'uniqueLabels' and 'roiData'
					}

					#ifdef WITH_PYTHON_H
					if (PyErr_CheckSignals() != 0)
						throw pybind11::error_already_set();
					#endif

					// Show stayalive progress info
					VERBOSLVL2 (env.get_verbosity_level(),
						if (cnt++ % 4 == 0)
							std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << env.uniqueLabels.size() << " ROIs" << "\n";
					);
				}

			env.theImLoader.close();
		}

		// fix ROIs' AABBs with respect to anisotropy
		if (env.anisoOptions.customized() == false)
		{
			for (auto& rd : env.roiData)
			{
				LR& r = rd.second;
				r.make_nonanisotropic_aabb();
			}
		}
		else
		{
			double	ax = env.anisoOptions.get_aniso_x(),
				ay = env.anisoOptions.get_aniso_y(),
				az = env.anisoOptions.get_aniso_z();

			for (auto& rd : env.roiData)
			{
				LR& r = rd.second;
				r.make_anisotropic_aabb (ax, ay, az);
			}
		}

		return true;
	}

	//
	// segmented 3D case (true volumetric images e.g. .nii, .nii.gz, .dcm, etc)
	// prerequisite: 'env.theImLoader' needs to be pre-opened !
	//
	bool gatherRoisMetrics_3D (Environment& env, size_t sidx, const std::string& intens_fpath, const std::string& mask_fpath, size_t t_index)
	{
		SlideProps & sprp = env.dataset.dataset_props [sidx];
		if (! env.theImLoader.open(sprp, env.fpimageOptions))
		{
			std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
			return false;
		}

		// Read the tiff. The image loader is put in the open state in processDataset()
		size_t 
			w = env.theImLoader.get_full_width(),
			h = env.theImLoader.get_full_height(),
			d = env.theImLoader.get_full_depth(),
			sliceSize = w * h,
			timeFrameSize = sliceSize * d,
			timeI = env.theImLoader.get_inten_time(),
			timeM = env.theImLoader.get_mask_time(),
			nVoxI = timeFrameSize * timeI,
			nVoxM = timeFrameSize * timeM;

		// is this intensity-mask pair's shape supported?
		if (nVoxI < nVoxM)
		{
			std::string erm = "Error: unsupported shape - intensity file: " + std::to_string(nVoxI) + ", mask file: " + std::to_string(nVoxM);
#ifdef WITH_PYTHON_H
			throw erm;
#endif	
			std::cerr << erm << "\n";
			return false;
		}

		int cnt = 1;

		// Fetch a tile 
		bool ok = env.theImLoader.load_tile (0/*row*/ , 0/*col*/);
		if (!ok)
		{
			std::string erm = "Error fetching tile (0,0) from I:" + intens_fpath + " M:" + mask_fpath;
			#ifdef WITH_PYTHON_H
				throw erm;
			#endif	
			std::cerr << erm << "\n";
			return false;
		}

		// Get ahold of tile's pixel buffer
		auto dataI = env.theImLoader.get_int_tile_buffer(),
			dataL = env.theImLoader.get_seg_tile_buffer();

		size_t baseI, baseM;
		if (nVoxI == nVoxM)
		{
			baseM = baseI = t_index * timeFrameSize;
		}
		else // nVoxI > nVoxM
		{
			baseM = 0;
			baseI = t_index * timeFrameSize;
		}

		// Iterate voxels
		for (size_t i=0; i<nVoxM; i++)
		{
			size_t k = i + baseM;	// absolute index of mask voxel
			size_t j = i + baseI;	// absolute index of intensity voxel
									
			// Skip non-mask pixels
			auto label = dataL[k];
			if (!label)
				continue;

			int z = k / sliceSize,
				y = (k - z*sliceSize) / w,
				x = (k - z * sliceSize) % w;

			// Skip tile buffer pixels beyond the image's bounds
			if (x >= w || y >= h || z >= d)
				continue;

			// Collapse all the labels to one if single-ROI mde is requested
			if (env.singleROI)
				label = 1;

			// Update pixel's ROI metrics
			feed_pixel_2_metrics_3D (env.uniqueLabels, env.roiData, x, y, z, dataI[j], label, sidx);
		}

#ifdef WITH_PYTHON_H
		if (PyErr_CheckSignals() != 0)
			throw pybind11::error_already_set();
#endif

		env.theImLoader.close();

		// fix ROIs' AABBs with respect to anisotropy
		if (env.anisoOptions.customized() == false)
		{
			for (auto& rd : env.roiData)
			{
				LR& r = rd.second;
				r.make_nonanisotropic_aabb();
			}
		}
		else
		{
			double	ax = env.anisoOptions.get_aniso_x(),
				ay = env.anisoOptions.get_aniso_y(),
				az = env.anisoOptions.get_aniso_z();

			for (auto& rd : env.roiData)
			{
				LR& r = rd.second;
				r.make_anisotropic_aabb(ax, ay, az);
			}
		}

		return true;
	}

#ifdef WITH_PYTHON_H

	//
	// segmented 2D case
	//
	bool gatherRoisMetricsInMemory (Environment & env, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens_images, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images, int pair_index)
	{
		VERBOSLVL4 (env.get_verbosity_level(), std::cout << "gatherRoisMetricsInMemory (pair_index=" << pair_index << ") \n");

		auto rI = intens_images.unchecked<3>();
		auto rL = label_images.unchecked<3>();

		size_t w = rI.shape(2);
		size_t h = rI.shape(1);

		for (size_t col = 0; col < w; col++)
			for (size_t row = 0; row < h; row++)
			{
				// Skip non-mask pixels
				auto label = rL (pair_index, row, col);
				if (!label)
					continue;

				// Collapse all the labels to one if single-ROI mde is requested
				if (env.singleROI)
					label = 1;

				// Update pixel's ROI metrics
				auto inten = rI (pair_index, row, col);
				feed_pixel_2_metrics (env.uniqueLabels, env.roiData, col, row, inten, label, pair_index); // Updates 'uniqueLabels' and 'roiData'

				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
			}

		return true;
	}

#endif
}