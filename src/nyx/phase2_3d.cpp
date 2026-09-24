#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
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
#include "helpers/helpers.h"		// Nyxus::near_eq for the physical-spacing resolver
#include "helpers/fsystem.h"
#include "helpers/timing.h"

namespace Nyxus
{
	// The effective 3D voxel spacing of a slide. Explicit --aniso* (anisoOptions.customized())
	// always wins. Otherwise, when --use-physical-spacing is on, the slide's OME PhysicalSize*
	// ratio-normalized so the smallest axis == 1 (the anisotropic path resamples by the
	// multiplier, so ratios - not absolute units - are what correct for non-cubic voxels).
	// Returns false + (1,1,1) when the grid is isotropic. Every pass over a volume resolves its
	// spacing here -- the prescan and phase-1 bounding boxes, the memory estimate, the in-RAM
	// voxel caches and the out-of-core clouds -- so one slide cannot mix resampled and
	// unresampled geometry.
	bool resolve_anisotropy (const AnisotropyOptions& aniso, bool use_physical_spacing, const SlideProps& p, double& ax, double& ay, double& az)
	{
		ax = ay = az = 1.0;

		if (aniso.customized())
		{
			ax = aniso.get_aniso_x();
			ay = aniso.get_aniso_y();
			az = aniso.get_aniso_z();
			return true;
		}

		if (use_physical_spacing)
		{
			double sx = p.phys_x, sy = p.phys_y, sz = p.phys_z;
			double mn = std::min(sx, std::min(sy, sz));
			if (mn > 0.0)
			{
				ax = sx / mn; ay = sy / mn; az = sz / mn;
				// only take the anisotropic path if the voxels are actually non-cubic
				if (! (Nyxus::near_eq(ax, 1.0) && Nyxus::near_eq(ay, 1.0) && Nyxus::near_eq(az, 1.0)))
					return true;
			}
			ax = ay = az = 1.0;
		}

		return false;
	}

	bool resolve_slide_anisotropy (const Environment& env, size_t sidx, double& ax, double& ay, double& az)
	{
		if (sidx >= env.dataset.dataset_props.size())
		{
			SlideProps unknown;
			return resolve_anisotropy (env.anisoOptions, false, unknown, ax, ay, az);
		}
		return resolve_anisotropy (env.anisoOptions, env.use_physical_spacing(), env.dataset.dataset_props[sidx], ax, ay, az);
	}

	bool stream_volume_checked (ImageLoader& ilo, size_t channel, size_t t_index, const std::string& intens_fpath, const std::string& mask_fpath,
		const std::function<void(size_t z, const std::vector<uint32_t>& int_plane, const std::vector<uint32_t>& seg_plane)>& sink)
	{
		// ImageLoader::open() guarantees that the mask's and intensity's width, height and depth
		// match. They may differ only in the number of time frames: 1:1, 1:N and N:1 pair up.
		size_t frameSize = ilo.get_full_width() * ilo.get_full_height() * ilo.get_full_depth(),
			nVoxI = frameSize * ilo.get_inten_time(),
			nVoxM = frameSize * ilo.get_mask_time();	// 0 in whole-volume mode

		if (nVoxI >= nVoxM)
		{
			ilo.stream_volume_planes (channel, t_index, sink);
			return true;
		}

		std::string erm = "Error: unsupported shape - intensity file " + intens_fpath + ": " + std::to_string(nVoxI)
			+ " voxels, mask file " + mask_fpath + ": " + std::to_string(nVoxM);
#ifdef WITH_PYTHON_H
		throw std::runtime_error (erm);
#endif
		std::cerr << erm << "\n";
		return false;
	}
	bool open_scanned_pair (Environment& env, int slide_idx, const std::string& intens_fpath, const std::string& label_fpath)
	{
		SlideProps p (intens_fpath, label_fpath);
		const SlideProps * scanned = env.dataset.scanned_slide (slide_idx);
		if (scanned)
			p.inherit_intensity_domain (*scanned);
		if (! env.theImLoader.open(p, env.fpimageOptions))
		{
			// open() allocates the intensity loader before the mask one and returns false from
			// either half, so a failed open leaves loaders behind unless they are released here
			env.theImLoader.close();
			std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
			return false;
		}
		return true;
	}

	//
	// Loads ROI voxels into voxel clouds
	//
	bool scanTrivialRois_3D (Environment & env, const std::vector<int>& batch_labels, const std::string& intens_fpath, const std::string& label_fpath, size_t t_index, size_t channel)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		if (! open_scanned_pair (env, batch_labels.empty() ? -1 : env.roiData[batch_labels[0]].slide_idx, intens_fpath, label_fpath))
			return false;

		const size_t w = env.theImLoader.get_full_width(),
			h = env.theImLoader.get_full_height();

		// Stream this (channel, timeframe)'s volume plane by plane, with the mask plane that pairs
		// with it, caching the batch's ROI voxels
		bool ok = stream_volume_checked (env.theImLoader, channel, t_index, intens_fpath, label_fpath,
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

						// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
						if (! env.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
							continue;

						// Collapse all the labels to one if single-ROI mde is requested
						if (env.singleROI)
							label = 1;

						// Cache this pixel
						LR& r = env.roiData[label];
						feed_pixel_2_cache_3D_LR ((int) x, (int) y, (int) z, dataI[i], r);
					}
			});
		if (! ok)
		{
			// the pair this function opened is its to release on the way out too
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

		// Dump ROI pixel clouds to the output directory
		VERBOSLVL5 (env.get_verbosity_level(), dump_roi_pixels(env.dim(), Nyxus::get_temp_dir_path(), batch_labels, label_fpath, env.uniqueLabels, env.roiData));

		return true;
	}

	//
	// Loads ROI voxels into voxel clouds
	//
	bool scanTrivialRois_3D_anisotropic(
		Environment& env,
		const std::vector<int>& batch_labels,
		const std::string& intens_fpath,
		const std::string& label_fpath,
		size_t t_index,
		size_t channel,
		double aniso_x,
		double aniso_y,
		double aniso_z)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort (whiteList.begin(), whiteList.end());

		if (! open_scanned_pair (env, batch_labels.empty() ? -1 : env.roiData[batch_labels[0]].slide_idx, intens_fpath, label_fpath))
			return false;

		const size_t
			w = env.theImLoader.get_full_width(),
			h = env.theImLoader.get_full_height(),
			d = env.theImLoader.get_full_depth();

		// virtual dimensions
		const size_t virt_h = h * aniso_y,
			virt_w = w * aniso_x,
			virt_d = d * aniso_z;

		// Stream this (channel, timeframe)'s volume plane by plane and fill each virtual voxel with
		// the physical voxel nearest to it. A virtual plane's physical plane never decreases with
		// its index, so each virtual plane is filled when its physical plane arrives.
		size_t vZ = 0;	// the next virtual plane to fill
		bool ok = stream_volume_checked (env.theImLoader, channel, t_index, intens_fpath, label_fpath,
			[&](size_t z, const std::vector<uint32_t>& dataI, const std::vector<uint32_t>& dataL)
			{
				for (; vZ < virt_d; vZ++)
				{
					const size_t pZ = vZ / aniso_z + 0.5;
					if (pZ > z)
						break;		// its physical plane is still to come
					if (pZ < z)
						continue;

					for (size_t vY = 0; vY < virt_h; vY++)
						for (size_t vX = 0; vX < virt_w; vX++)
						{
							// physical position; casting from virtual to physical can land outside
							// the physical bounds
							const size_t pY = vY / aniso_y + 0.5,
								pX = vX / aniso_x + 0.5;
							if (pX >= w || pY >= h)
								continue;
							const size_t i = pY * w + pX;

							// skip non-mask pixels
							auto lbl = dataL[i];
							if (!lbl)
								continue;

							// skip this ROI if the label isn't in the pending set of a multi-ROI mode
							if (!env.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), lbl))
								continue;

							// collapse all the labels to one if single-ROI mde is requested
							if (env.singleROI)
								lbl = 1;

							// cache this voxel
							LR& r = env.roiData[lbl];
							feed_pixel_2_cache_3D_LR (vX, vY, vZ, dataI[i], r);
						}
				}
			});
		if (! ok)
		{
			// the pair this function opened is its to release on the way out too
			env.theImLoader.close();
			return false;
		}

		// The scan is finished and nothing below reads the pair, so it is released here rather
		// than after the interrupt check, which leaves by a throw.
		env.theImLoader.close();

	#ifdef WITH_PYTHON_H
		// allow keyboard interrupt
		if (PyErr_CheckSignals() != 0)
			throw pybind11::error_already_set();
	#endif
		return true;
	}
	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi'
	//
	bool scan_trivial_wholevolume (
		LR& vroi,
		const std::string& intens_fpath,
		ImageLoader& ilo,
		size_t channel,
		size_t timeframe)
	{
		const size_t fullW = ilo.get_full_width(),
			fullH = ilo.get_full_height();

		// Stream the X*Y*Z volume of this (channel, timeframe) plane by plane into the virtual ROI.
		// Whole-slide has no mask.
		return stream_volume_checked (ilo, channel, timeframe, intens_fpath, "",
			[&](size_t z, const std::vector<uint32_t>& dataI, const std::vector<uint32_t>&)
			{
				for (size_t y = 0; y < fullH; y++)
					for (size_t x = 0; x < fullW; x++)
						feed_pixel_2_cache_3D_LR ((int) x, (int) y, (int) z, dataI[y * fullW + x], vroi);
			});
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi'
	//
	bool scan_trivial_wholevolume_anisotropic (
		LR& vroi,
		const std::string& intens_fpath,
		ImageLoader& ilo,
		double aniso_x,
		double aniso_y,
		double aniso_z,
		size_t channel,
		size_t timeframe)
	{
		const size_t fullW = ilo.get_full_width(),
			fullH = ilo.get_full_height(),
			fullD = ilo.get_full_depth();

		const size_t vh = (size_t) (double(fullH) * aniso_y),
			vw = (size_t) (double(fullW) * aniso_x),
			vd = (size_t) (double(fullD) * aniso_z);

		// Stream the X*Y*Z volume of this (channel, timeframe) plane by plane and fill each virtual
		// voxel with the physical voxel it falls in, clamped against float rounding at the ratio
		// boundary. A virtual plane's physical plane never decreases with its index, so each
		// virtual plane is filled when its physical plane arrives. Whole-slide has no mask.
		size_t z = 0;	// the next virtual plane to fill
		return stream_volume_checked (ilo, channel, timeframe, intens_fpath, "",
			[&](size_t pz, const std::vector<uint32_t>& dataI, const std::vector<uint32_t>&)
			{
				for (; z < vd; z++)
				{
					const size_t ph_z = (std::min<size_t>) ((size_t) (double(z) / aniso_z), fullD - 1);
					if (ph_z > pz)
						break;		// its physical plane is still to come
					if (ph_z < pz)
						continue;

					for (size_t y = 0; y < vh; y++)
						for (size_t x = 0; x < vw; x++)
						{
							const size_t ph_x = (std::min<size_t>) ((size_t) (double(x) / aniso_x), fullW - 1),
								ph_y = (std::min<size_t>) ((size_t) (double(y) / aniso_y), fullH - 1);
							feed_pixel_2_cache_3D_LR ((int) x, (int) y, (int) z, dataI[ph_y * fullW + ph_x], vroi);
						}
				}
			});
	}


	bool processTrivialRois_3D (Environment & env, size_t sidx, size_t t_index, size_t channel, const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit)
	{
		std::vector<int> Pending;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = env.roiData[lab];

			size_t itemFootprint = r.get_ram_footprint_estimate (Pending.size());

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

				VERBOSLVL2 (env.get_verbosity_level(), std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of total " << env.uniqueLabels.size() << " ROIs\n");
				VERBOSLVL2 (env.get_verbosity_level(),
					if (Pending.size() == 1)
						std::cout << ">>> (single ROI label " << Pending[0] << ")\n";
					else
						std::cout << ">>> (ROI labels " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				);

				// --aniso* (explicit) or opt-in OME physical spacing selects the anisotropic path
				double ax, ay, az;
				if (! resolve_slide_anisotropy (env, sidx, ax, ay, az))
				{
					if (! scanTrivialRois_3D (env, Pending, intens_fpath, label_fpath, t_index, channel))
						return false;
				}
				else
				{
					if (! scanTrivialRois_3D_anisotropic (env, Pending, intens_fpath, label_fpath, t_index, channel, ax, ay, az))
						return false;

					// The ROI's extent and voxel count describe the cloud that was just cached.
					// gatherRoisMetrics_3D recorded them from the PHYSICAL grid, and the anisotropic
					// scan caches the resampled (virtual) cloud, which has both a different extent and
					// a different voxel count -- and aux_area divides every feature that averages.
					for (auto lbl : Pending)
					{
						LR& r = env.roiData[lbl];
						r.aabb.update_from_voxelcloud (r.raw_pixels_3D);
						r.aux_area = (unsigned int) r.raw_pixels_3D.size();
					}
				}

				// Allocate memory
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tallocating ROI buffers\n";)
					allocateTrivialRoisBuffers_3D (Pending, env.roiData, env.hostCache);

				// Reduce them
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\treducing ROIs\n";)
					// reduce_trivial_rois(Pending);	
					reduce_trivial_rois_manual (Pending, env);

				// Free memory
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tfreeing ROI buffers\n";)
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
			
			VERBOSLVL2(env.get_verbosity_level(),
				std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << "(" << env.uniqueLabels.size() << ") ROIs\n";
				std::cout << ">>> (labels " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				);

			// --aniso* (explicit) or opt-in OME physical spacing selects the anisotropic path
			double ax, ay, az;
			if (! resolve_slide_anisotropy (env, sidx, ax, ay, az))
			{
				if (! scanTrivialRois_3D (env, Pending, intens_fpath, label_fpath, t_index, channel))
					return false;
			}
			else
			{
				if (! scanTrivialRois_3D_anisotropic (env, Pending, intens_fpath, label_fpath, t_index, channel, ax, ay, az))
					return false;

				// rescan and update ROI's AABB and voxel count -- see the identical fix (and
				// its rationale) in the main batch loop above.
				for (auto lbl : Pending)
				{
					LR& r = env.roiData[lbl];
					r.aabb.update_from_voxelcloud(r.raw_pixels_3D);
					r.aux_area = (unsigned int) r.raw_pixels_3D.size();
				}
			}

			for (auto lab : Pending)
			{
				LR& r = env.roiData[lab];
				for (Pixel3& vox : r.raw_pixels_3D)
				{
					assert (vox.x >= r.aabb.get_xmin());
					assert (vox.x <= r.aabb.get_xmax());
					assert (vox.y >= r.aabb.get_ymin());
					assert (vox.y <= r.aabb.get_ymax());
					assert (vox.z >= r.aabb.get_zmin());
					assert (vox.z <= r.aabb.get_zmax());
				}
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
			//reduce_trivial_rois(Pending);	
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

	void allocateTrivialRoisBuffers_3D (const std::vector<int>& roi_labels, Roidata& roiData, CpusideCache & cache)
	{
		// Calculate the total memory demand (in # of items) of all segments' image matrices
		cache.imageMatrixBufferLen = 0;
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			size_t w = r.aabb.get_width(),
				h = r.aabb.get_height(),
				d = r.aabb.get_z_depth(),
				v = w * h * d;
			cache.imageMatrixBufferLen += v;

			cache.largest_roi_imatr_buf_len = cache.largest_roi_imatr_buf_len == 0 ? v : std::max (cache.largest_roi_imatr_buf_len, v);
		}

		//
		// Preallocate image matrices and cubes here (in the future).
		//
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			r.aux_image_cube.calculate_from_pixelcloud(r.raw_pixels_3D, r.aabb);
		}
	}

}
