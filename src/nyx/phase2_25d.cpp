#include <atomic>
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
#include "helpers/fsystem.h"
#include "globals.h"
#include "helpers/timing.h"

namespace Nyxus
{

	bool scanTrivialRois_25D (Environment & env, const std::vector<int>& batch_labels, const std::string& intens_fpath, const std::string& label_fpath, const std::vector<std::string>& z_indices)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		for (size_t z = 0; z < z_indices.size(); z++)
		{
			// prepare the physical file 
			// 
			// ifile and mfile contain a placeholder for the z-index. We need to turn them to physical filesystem files
			auto zValue = z_indices[z];	// realistic dataset's z-values may be arbitrary (non-zer-based and non-contiguous), so use the actual value
			std::string ifpath = std::regex_replace(intens_fpath, std::regex("\\*"), zValue),
				mfpath = std::regex_replace(label_fpath, std::regex("\\*"), zValue);

			// Scan this Z intensity-mask pair 
			SlideProps p (ifpath, mfpath);
			if (! env.theImLoader.open(p, env.fpimageOptions))
			{
				std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
				return false;
			}

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
			{
				for (unsigned int col = 0; col < ntv; col++)
				{
					// Fetch the tile 
					bool ok = env.theImLoader.load_tile(row, col);
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
					auto dataI = env.theImLoader.get_int_tile_buffer(),
						dataL = env.theImLoader.get_seg_tile_buffer();

					// Iterate pixels
					for (unsigned long i = 0; i < tileSize; i++)
					{
						// Skip non-mask pixels
						auto label = dataL[i];
						if (!label)
							continue;

						// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
						if (!env.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
							continue;

						auto inten = dataI[i];
						int y = row * th + i / tw,
							x = col * tw + i % tw;

						// Skip tile buffer pixels beyond the image's bounds
						if (x >= fullwidth || y >= fullheight)
							continue;

						// Collapse all the labels to one if single-ROI mde is requested
						if (env.singleROI)
							label = 1;

						// Cache this pixel 
						LR& r = env.roiData[label];
						feed_pixel_2_cache_3D_LR (x, y, z, dataI[i], r);
					}

					VERBOSLVL2(env.get_verbosity_level(),
						// Show stayalive progress info
						if (cnt++ % 4 == 0)
						{
							static std::atomic<int> prevIntPc{0};
							float pc = int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100.;
							if (int(pc) != prevIntPc)
							{
								std::cout << "\t scan trivial " << int(pc) << " %\n";
								prevIntPc = int(pc);
							}
						}
					);
				}
			}

			// Close the image pair
			env.theImLoader.close();
		}

		// Dump ROI pixel clouds to the output directory
		VERBOSLVL5 (env.get_verbosity_level(), dump_roi_pixels(env.dim(), Nyxus::get_temp_dir_path(), batch_labels, label_fpath, env.uniqueLabels, env.roiData));

		return true;
	}

	bool scanTrivialRois_25D_anisotropic (
		Environment & env,
		const std::vector<int>& batch_labels,
		const std::string& intens_fpath,
		const std::string& label_fpath,
		const std::vector<std::string>& z_indices,
		double aniso_x,
		double aniso_y,
		double aniso_z)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		int lvl = 0,	// pyramid level
			lyr = 0;	//	layer

		size_t vD = (size_t)(double(z_indices.size()) * aniso_z);	// virtual depth

		for (size_t vz = 0; vz < vD; vz++)
		{
			size_t z = size_t(double(vz) / aniso_z);	// physical z

			// prepare the physical file 
			// 
			// ifile and mfile contain a placeholder for the z-index. We need to turn them to physical filesystem files
			auto zValue = z_indices[z];	// realistic dataset's z-values may be arbitrary (non-zer-based and non-contiguous), so use the actual value
			std::string ifpath = std::regex_replace(intens_fpath, std::regex("\\*"), zValue),
				mfpath = std::regex_replace(label_fpath, std::regex("\\*"), zValue);

			// Scan this Z intensity-mask pair 
			SlideProps p (ifpath, mfpath);
			if (! env.theImLoader.open(p, env.fpimageOptions))
			{
				std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
				return false;
			}

			size_t nth = env.theImLoader.get_num_tiles_hor(),
				ntv = env.theImLoader.get_num_tiles_vert(),
				fw = env.theImLoader.get_tile_width(),
				th = env.theImLoader.get_tile_height(),
				tw = env.theImLoader.get_tile_width(),
				tileSize = env.theImLoader.get_tile_size(),
				fullwidth = env.theImLoader.get_full_width(),
				fullheight = env.theImLoader.get_full_height();

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
					// tile position
					size_t tidx_y = size_t(vr / vth),
						tidx_x = size_t(vc / vtw);

					// load it
					if (tidx_y != curt_y || tidx_x != curt_x)
					{
						bool ok = env.theImLoader.load_tile(tidx_y, tidx_x);
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
					auto dataI = env.theImLoader.get_int_tile_buffer(),
						dataL = env.theImLoader.get_seg_tile_buffer();

					// skip non-mask pixels
					auto label = dataL[i];
					if (!label)
						continue;

					// skip this ROI if the label isn't in the pending set of a multi-ROI mode
					if (!env.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
						continue;

					// skip tile buffer pixels beyond the image's bounds
					if (vc >= fullwidth || vr >= fullheight)
						continue;

					// collapse all the labels to one if single-ROI mde is requested
					if (env.singleROI)
						label = 1;

					// cache this voxel 
					auto inten = dataI[i];
					LR& r = env.roiData[label];
					feed_pixel_2_cache_3D_LR (vc, vr, vz, dataI[i], r);
				}
			}

			// Close the image pair
			env.theImLoader.close();
		}

		// Dump ROI pixel clouds to the output directory
		VERBOSLVL5 (env.get_verbosity_level(), dump_roi_pixels(env.dim(), Nyxus::get_temp_dir_path(), batch_labels, label_fpath, env.uniqueLabels, env.roiData));

			return true;
	}

	bool processTrivialRois_25D (Environment & env, const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit, const std::vector<std::string>& z_indices)
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
				Pending.push_back(lab);
				batchDemand += itemFootprint;
			}
			else
			{
				// Scan pixels of pending trivial ROIs 
				std::sort(Pending.begin(), Pending.end());
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of total " << env.uniqueLabels.size() << " ROIs\n");
				VERBOSLVL2 (env.get_verbosity_level(),
						if (Pending.size() == 1)
							std::cout << ">>> (single ROI label " << Pending[0] << ")\n";
						else
							std::cout << ">>> (ROI labels " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
					);

					if (env.anisoOptions.customized() == false)
					{
						scanTrivialRois_25D (env, Pending, intens_fpath, label_fpath, z_indices);
					}
					else
					{
						double	ax = env.anisoOptions.get_aniso_x(),
							ay = env.anisoOptions.get_aniso_y(),
							az = env.anisoOptions.get_aniso_z();
						scanTrivialRois_25D_anisotropic (env, Pending, intens_fpath, label_fpath, z_indices, ax, ay, az);
					}

				// Allocate memory
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tallocating ROI buffers\n");
				allocateTrivialRoisBuffers_3D (Pending, env.roiData, env.hostCache);

				// Reduce them
				VERBOSLVL2(env.get_verbosity_level(), std::cout << "\treducing ROIs\n");
				// reduce_trivial_rois(Pending):
				reduce_trivial_rois_manual (Pending, env);

				// Free memory
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tfreeing ROI buffers\n");
				freeTrivialRoisBuffers_3D (Pending, env.roiData);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()

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
			// Read raw pixels of pending trivial ROIs 
			std::sort(Pending.begin(), Pending.end());
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of " << env.uniqueLabels.size() << " all ROIs\n");
			VERBOSLVL2 (env.get_verbosity_level(),
					if (Pending.size() == 1)
						std::cout << ">>> (single ROI " << Pending[0] << ")\n";
					else
						std::cout << ">>> (ROIs " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				);
			if (env.anisoOptions.customized() == false)
			{
				scanTrivialRois_25D (env, Pending, intens_fpath, label_fpath, z_indices);
			}
			else
			{
				double	ax = env.anisoOptions.get_aniso_x(),
					ay = env.anisoOptions.get_aniso_y(),
					az = env.anisoOptions.get_aniso_z();
				scanTrivialRois_25D_anisotropic (env, Pending, intens_fpath, label_fpath, z_indices, ax, ay, az);
			}

			// Allocate memory
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tallocating ROI buffers\n");
			allocateTrivialRoisBuffers_3D (Pending, env.roiData, env.hostCache);

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
			freeTrivialRoisBuffers_3D (Pending, env.roiData);

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

}