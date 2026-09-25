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
#include "helpers/helpers.h"
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
		size_t ntHor = L.get_num_tiles_hor(),
			ntVert = L.get_num_tiles_vert(),
			fw = L.get_tile_width(),
			th = L.get_tile_height(),
			tw = L.get_tile_width(),
			tileSize = L.get_tile_size(),
			fullwidth = L.get_full_width(),
			fullheight = L.get_full_height();

		size_t tileCnt = 1;
		for (size_t row = 0; row < ntVert; row++)
			for (size_t col = 0; col < ntHor; col++)
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
				bool wholeslide = env.singleROI;

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

					// Merge all foreground labels into a single ROI if requested
					if (env.mergeLabels)
						label = 1;

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
					if (tileCnt++ % 4 == 0)
						std::cout << "\t" << Nyxus::round2(100. * float(row * ntHor + col) / float(ntHor * ntVert)) << " %\t " << Nyxus::virguler_ulong(env.uniqueLabels.size()) << " ROIs gathered" << "\n";
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
			// carry over what the prescan measured and recorded for this slide, so this pass's
			// grey levels land in the same domain the features will report them in
			const SlideProps * scanned = env.dataset.scanned_slide ((int)sidx);
			if (scanned)
				sprp.inherit_intensity_domain (*scanned);

			// Extract features from this intensity-mask pair 
			if (env.theImLoader.open(sprp, env.fpimageOptions) == false)
			{
				// open() allocates the intensity loader before the mask one and returns false from
				// either half, so a failed open leaves loaders behind unless they are released here
				env.theImLoader.close();
				std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
				return false;
			}

			// Read the tiff. The image loader is put in the open state in processDataset()
			size_t ntHor = env.theImLoader.get_num_tiles_hor(),	// tiles across a row
				ntVert = env.theImLoader.get_num_tiles_vert(),	// tiles down a column
				fw = env.theImLoader.get_tile_width(),
				th = env.theImLoader.get_tile_height(),
				tw = env.theImLoader.get_tile_width(),
				tileSize = env.theImLoader.get_tile_size(),
				fullwidth = env.theImLoader.get_full_width(),
				fullheight = env.theImLoader.get_full_height();

			int cnt = 1;
			for (unsigned int row = 0; row < ntVert; row++)
				for (unsigned int col = 0; col < ntHor; col++)
				{
					// Fetch a tile 
					bool ok = env.theImLoader.load_tile (row, col);
					if (!ok)
					{
						std::string erm = "Error fetching tile row:" + std::to_string(row) + " col:" + std::to_string(col) + " from I:" + ifpath + " M:" + mfpath;
						env.theImLoader.close();	// released before the throw below, which leaves under Python
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
					{
						// the interrupt leaves this pass for good, and the pair is this function's
						env.theImLoader.close();
						throw pybind11::error_already_set();
					}
					#endif

					// Show stayalive progress info
					VERBOSLVL2 (env.get_verbosity_level(),
						if (cnt++ % 4 == 0)
							std::cout << "\t" << int((row * ntHor + col) * 100 / float(ntHor * ntVert) * 100) / 100. << "%\t" << env.uniqueLabels.size() << " ROIs" << "\n";
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
	bool gatherRoisMetrics_3D (Environment& env, size_t sidx, const std::string& intens_fpath, const std::string& mask_fpath, size_t t_index, size_t channel)
	{
		SlideProps & sprp = env.dataset.dataset_props [sidx];
		if (! env.theImLoader.open(sprp, env.fpimageOptions))
		{
			// open() allocates the intensity loader before the mask one and returns false from
			// either half, so a failed open leaves loaders behind unless they are released here
			env.theImLoader.close();
			std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
			return false;
		}

		const size_t w = env.theImLoader.get_full_width(),
			h = env.theImLoader.get_full_height();

		// Stream this (channel, timeframe)'s volume plane by plane, with the mask plane that pairs
		// with it, exactly as the Phase-2 scan does, so the ROI metrics see the voxels Phase 2 caches
		bool ok = stream_volume_checked (env.theImLoader, channel, t_index, intens_fpath, mask_fpath,
			[&](size_t z, const std::vector<uint32_t>& dataI, const std::vector<uint32_t>& dataL)
			{
				for (size_t y = 0; y < h; y++)
					for (size_t x = 0; x < w; x++)
					{
						size_t i = y * w + x;

						// Skip non-mask pixels
						auto label = dataL[i];
						if (!label)
							continue;

						// Collapse all the labels to one if single-ROI mde is requested
						if (env.singleROI)
							label = 1;

						// Update pixel's ROI metrics
						feed_pixel_2_metrics_3D (env.uniqueLabels, env.roiData, (int) x, (int) y, (int) z, dataI[i], label, sidx);
					}
			});
		if (! ok)
		{
			// the pair opened above is this function's to release on the way out too
			env.theImLoader.close();
			return false;
		}
		// The scan is finished and nothing below reads the pair, so it is released here rather
		// than after the interrupt check, which leaves by a throw.
		env.theImLoader.close();

#ifdef WITH_PYTHON_H
		if (PyErr_CheckSignals() != 0)
			throw pybind11::error_already_set();
#endif

		// fix ROIs' AABBs with respect to anisotropy, on the spacing every pass over this volume
		// resolves -- explicit --aniso* or, opted in, the slide's physical voxel size. These boxes
		// size the ROI buffers and drive the oversized check, so they describe the same
		// (resampled) geometry the phase-2 scans cache.
		double ax, ay, az;
		bool anisotropic = resolve_slide_anisotropy (env, sidx, ax, ay, az);
		for (auto& rd : env.roiData)
		{
			LR& r = rd.second;
			if (anisotropic)
				r.make_anisotropic_aabb (ax, ay, az);
			else
				r.make_nonanisotropic_aabb();
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

				// Merge all foreground labels into a single ROI if requested
				if (env.mergeLabels)
					label = 1;

				// Collapse all the labels to one if single-ROI mde is requested
				if (env.singleROI)
					label = 1;

				// Update pixel's ROI metrics
				auto inten = rI (pair_index, row, col);
				feed_pixel_2_metrics (env.uniqueLabels, env.roiData, col, row, inten, label, pair_index); // Updates 'uniqueLabels' and 'roiData'
			}
		if (PyErr_CheckSignals() != 0)
			throw pybind11::error_already_set();

		return true;
	}

#endif
}