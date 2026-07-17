#include <algorithm>
#include <cassert>
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
#include "helpers/helpers.h"		// FIX (IO): Nyxus::near_eq for the physical-spacing resolver
#include "helpers/fsystem.h"
#include "helpers/timing.h"

namespace Nyxus
{
	// FIX (IO): resolve the effective 3D voxel spacing for slide `sidx`. Explicit --aniso*
	// (anisoOptions.customized()) always wins. Otherwise, when --use-physical-spacing is on,
	// use the slide's OME PhysicalSize* ratio-normalized so the smallest axis == 1 (Nyxus's
	// anisotropic path resamples by the multiplier, so ratios - not absolute units - are what
	// correct for non-cubic voxels). Returns false + (1,1,1) when the grid is isotropic.
	bool resolve_slide_anisotropy (const Environment& env, size_t sidx, double& ax, double& ay, double& az)
	{
		ax = ay = az = 1.0;

		if (env.anisoOptions.customized())
		{
			ax = env.anisoOptions.get_aniso_x();
			ay = env.anisoOptions.get_aniso_y();
			az = env.anisoOptions.get_aniso_z();
			return true;
		}

		if (env.use_physical_spacing() && sidx < env.dataset.dataset_props.size())
		{
			const SlideProps& p = env.dataset.dataset_props[sidx];
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

	//
	// Loads ROI voxels into voxel clouds
	//
	bool scanTrivialRois_3D (Environment & env, const std::vector<int>& batch_labels, const std::string& intens_fpath, const std::string& label_fpath, size_t t_index, size_t channel)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		// Scan this Z intensity-mask pair
		SlideProps p (intens_fpath, label_fpath);
		// FIX: preserve_hu offset base — a bare SlideProps defaults min_preroi_inten to -1, which
		// would make the load-time HU offset disagree with the prescan (and the whole-slide pass),
		// producing out-of-range grey bins and a crash. Inherit the scanned HU-domain min/max + flag.
		// Guarded on preserve_hu so the default (non-HU) path is byte-for-byte unchanged.
		if (env.fpimageOptions.preserve_hu() && ! batch_labels.empty())
		{
			const SlideProps& scanned = env.dataset.dataset_props [env.roiData[batch_labels[0]].slide_idx];
			p.min_preroi_inten = scanned.min_preroi_inten;
			p.max_preroi_inten = scanned.max_preroi_inten;
			p.preserve_hu = scanned.preserve_hu;
		}
		if (! env.theImLoader.open(p, env.fpimageOptions))
		{
			std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
			return false;
		}

		// thanks to ImageLoader::open() we are guaranteed that the mask and intensity volumes' 
		// width, height, and depth match. Mask and intensity may only differ in the number of 
		// time frames: 1:1, 1:N, and N:1 cases are permitted.
		size_t 
			/*
			* we don't need these in the 3D scenario:
			* 
			nth = env.theImLoader.get_num_tiles_hor(),
			ntv = env.theImLoader.get_num_tiles_vert(),
			fw = env.theImLoader.get_tile_width(),
			th = env.theImLoader.get_tile_height(),
			tw = env.theImLoader.get_tile_width(),
			tileSize = env.theImLoader.get_tile_size(),
			*/
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

		// FIX (IO): assemble the whole X*Y*Z volume for this timeframe via load_volume,
		// which loops Z internally. The previous load_tile(0,0) delivered only plane
		// z=0 for plane-by-plane loaders (OME-Zarr, multi-page/OME-TIFF), so those never
		// read a real 3D volume. The mask sits on frame t_index (1:1) or frame 0 (the
		// 1-mask : N-intensity case); the intensity sits on frame t_index.
		const size_t maskFrame = (timeI == timeM) ? t_index : 0;
		bool ok = env.theImLoader.load_volume (channel, t_index, maskFrame);
		if (!ok)
		{
			std::string erm = "Error fetching segmented data from " + intens_fpath + "(I) " + label_fpath + "(M)";
#ifdef WITH_PYTHON_H
			throw erm;
#endif
			std::cerr << erm << "\n";
			return false;
		}

		// Both buffers now hold exactly this timeframe's X*Y*Z volume (the frame was
		// already selected by load_volume), so no base offset is needed.
		const std::vector<uint32_t>& dataI = env.theImLoader.get_int_volume_buffer();
		const std::vector<uint32_t>& dataL = env.theImLoader.get_seg_volume_buffer();

		// Iterate the single-frame volume's voxels
		for (size_t i = 0; i < timeFrameSize; i++)
		{
			// Skip non-mask pixels
			auto label = dataL[i];
			if (!label)
				continue;

			// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
			if (! env.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
				continue;

			int z = i / sliceSize,
				y = (i - z * sliceSize) / w,
				x = (i - z * sliceSize) % w;

			// Skip tile buffer pixels beyond the image's bounds
			if (x >= w || y >= h || z >= d)
				continue;

			// Collapse all the labels to one if single-ROI mde is requested
			if (env.singleROI)
				label = 1;

			// Cache this pixel
			LR& r = env.roiData[label];
			feed_pixel_2_cache_3D_LR (x, y, z, dataI[i], r);
		}

#ifdef WITH_PYTHON_H
		if (PyErr_CheckSignals() != 0)
			throw pybind11::error_already_set();
#endif

		// Close the image pair
		env.theImLoader.close();

		// Dump ROI pixel clouds to the output directory
		VERBOSLVL5 (env.get_verbosity_level(), dump_roi_pixels(env.dim(), Nyxus::get_temp_dir_path(), batch_labels, label_fpath, env.uniqueLabels, env.roiData));

		return true;
	}

	bool scanTrivialRois_3D_anisotropic__BEFORE4D(
		Environment & env,
		const std::vector<int>& batch_labels,
		const std::string& intens_fpath,
		const std::string& label_fpath,
		double aniso_x,
		double aniso_y,
		double aniso_z)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		int lvl = 0,	// pyramid level
			lyr = 0;	//	layer

		// Scan this Z intensity-mask pair 
		SlideProps p (intens_fpath, label_fpath);
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
			fullW = env.theImLoader.get_full_width(),
			fullH = env.theImLoader.get_full_height(),
			fullD = env.theImLoader.get_full_depth();

		size_t vD = (size_t)(double(fullD) * aniso_z);	// virtual depth

		for (size_t vz = 0; vz < vD; vz++)
		{
			size_t z = size_t(double(vz) / aniso_z);	// physical z

			// virtual slide properties
			size_t vh = (size_t)(double(fullH) * aniso_y),
				vw = (size_t)(double(fullW) * aniso_x),
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
					if (vc >= fullW || vr >= fullH)
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

		// temp slideprops instance to pass some into to ImageLoader
		SlideProps p (intens_fpath, label_fpath);
		// FIX: preserve_hu offset base — inherit the scanned HU-domain slide min/max + flag so the
		// load-time HU offset matches the prescan (bare SlideProps defaults min to -1 -> bad bins/crash).
		// Guarded on preserve_hu so the default (non-HU) path is byte-for-byte unchanged.
		if (env.fpimageOptions.preserve_hu() && ! batch_labels.empty())
		{
			const SlideProps& scanned = env.dataset.dataset_props [env.roiData[batch_labels[0]].slide_idx];
			p.min_preroi_inten = scanned.min_preroi_inten;
			p.max_preroi_inten = scanned.max_preroi_inten;
			p.preserve_hu = scanned.preserve_hu;
		}
		if (!env.theImLoader.open(p, env.fpimageOptions))
		{
			std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
			return false;
		}

		// thanks to ImageLoader::open() we are guaranteed that the mask's and intensity's  
		// W, H, and D match. Mask and intensity may only differ in the number of 
		// time frames: 1:1, 1:N, and N:1 cases are permitted
		size_t
			w = env.theImLoader.get_full_width(),
			h = env.theImLoader.get_full_height(),
			d = env.theImLoader.get_full_depth(),
			slice = w * h,
			timeFrameSize = slice * d,
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

		// FIX (IO): assemble the whole X*Y*Z volume for this timeframe (load_volume
		// loops Z), so plane-by-plane loaders (OME-Zarr, multi-page/OME-TIFF) read a
		// real 3D volume here too. Mask on frame t_index (1:1) or frame 0 (1 mask : N).
		const size_t maskFrame = (timeI == timeM) ? t_index : 0;
		if (!env.theImLoader.load_volume (channel, t_index, maskFrame))
		{
			std::string erm = "Error fetching data from file pair " + intens_fpath + "(I) " + label_fpath + "(M)";
	#ifdef WITH_PYTHON_H
			throw erm;
	#endif
			std::cerr << erm << "\n";
			return false;
		}

		// Both buffers hold exactly this timeframe's X*Y*Z volume (frame already selected).
		const std::vector<uint32_t>& dataI = env.theImLoader.get_int_volume_buffer();
		const std::vector<uint32_t>& dataL = env.theImLoader.get_seg_volume_buffer();

		// virtual dimensions
		size_t virt_h = h * aniso_y,
			virt_w = w * aniso_x,
			virt_d = d * aniso_z;
		size_t vSliceLen = virt_h * virt_w,
			virt_v = vSliceLen * virt_d;

		// iterate virtual voxels and fill them with corresponding physical intensities
		for (size_t vIdx = 0; vIdx < virt_v; vIdx++)
		{
			// virtual Cartesian position
			size_t vZ = vIdx / vSliceLen, 
				vLastSliceLen = vIdx % vSliceLen,
				vY = vLastSliceLen / virt_w,
				vX = vLastSliceLen % virt_w;

			// physical Cartesian position
			size_t pZ = vZ / aniso_z + 0.5,
				pY = vY / aniso_y + 0.5,
				pX = vX / aniso_x + 0.5;

			// skip a position outside the bounds
			// (since we are casting coorinates from virtual to physical,
			// we may get positions outside the physical bounds)
			if (pX >= w || pY >= h || pZ >= d)
				continue;

			// physical offset
			size_t i = pZ * slice + pY * w + pX;

			//
			// interpret the mask intensity
			//

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

#if !defined(NDEBUG)
			if (vZ >= virt_d)
				std::cout << "vZ=" << vZ << " < virt_d =" << virt_d << "\n";
			assert(vZ < virt_d);
			if (vY >= virt_h)
				std::cout << "vY=" << vY << " < virt_h =" << virt_h << "\n";
			assert(vY < virt_h);
			if (vX >= virt_w)
				std::cout << "vX=" << vX << " < virt_w =" << virt_w << "\n";
			assert(vX < virt_w);
#endif

			// cache this voxel
			auto inten = dataI[i];
			LR& r = env.roiData[lbl];
			feed_pixel_2_cache_3D_LR (vX, vY, vZ, inten, r);
		}

	#ifdef WITH_PYTHON_H
		// allow keyboard interrupt
		if (PyErr_CheckSignals() != 0)
			throw pybind11::error_already_set();
	#endif

		// Close the image pair
		env.theImLoader.close();
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
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs

		size_t fullwidth = ilo.get_full_width(),
			fullheight = ilo.get_full_height(),
			fullD = ilo.get_full_depth(),
			sliceSize = fullwidth * fullheight,
			nVox = sliceSize * fullD;

		// FIX (IO): assemble the whole X*Y*Z volume (load_volume loops Z) for the requested
		// (channel, timeframe) so plane-by-plane loaders (OME-Zarr, multi-page/OME-TIFF) read
		// the full volume of THIS C/T plane, not just plane z=0 of (c=0,t=0). No mask in WSI.
		if (!ilo.load_volume(channel, timeframe, timeframe))
		{
#ifdef WITH_PYTHON_H
			throw "Error fetching tile";
#endif
			std::cerr << "Error fetching tile\n";
			return false;
		}

		// Get ahold of the assembled volume buffer
		const std::vector<uint32_t>& dataI = ilo.get_int_volume_buffer();

		// iterate abstract tiles (in a tiled slide /e.g. tiled tiff/ they correspond to physical tiles, in a nontiled slide /e.g. scanline tiff or strip tiff/ they correspond to )
		int cnt = 1;

		// iterate voxels
		for (size_t i = 0; i < nVox; i++)
		{
			int z = i / sliceSize,
				y = (i - z * sliceSize) / fullwidth,
				x = (i - z * sliceSize) % fullwidth;

			// Skip tile buffer pixels beyond the image's bounds
			if (x >= fullwidth || y >= fullheight || z >= fullD)
				continue;

			// dynamic range within- and off-ROI
			auto inten = dataI[i];

			// Cache this pixel 
			feed_pixel_2_cache_3D_LR (x, y, z, inten, vroi);

		} //- all voxels

		return true;
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
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs

		size_t fullW = ilo.get_full_width(),
			fullH = ilo.get_full_height(),
			fullD = ilo.get_full_depth(),
			sliceSize = fullW * fullH;

		size_t vh = (size_t) (double(fullH) * aniso_y),
			vw = (size_t) (double(fullW) * aniso_x),
			vd = (size_t) (double(fullD) * aniso_z);

		// FIX (IO): assemble the whole X*Y*Z volume (load_volume loops Z) for the requested
		// (channel, timeframe) so plane-by-plane loaders read the full volume of this C/T plane.
		if (! ilo.load_volume(channel, timeframe, timeframe))
		{
#ifdef WITH_PYTHON_H
			throw "Error loading volume data";
#endif
			std::cerr << "Error loading volume data\n";
			return false;
		}

		// Get ahold of the assembled volume buffer
		const std::vector<uint32_t>& dataI = ilo.get_int_volume_buffer();

		// iterate virtual voxels
		size_t vSliceSize = vh * vw, 
			nVox = vh * vw * vd;
		for (size_t i = 0; i < nVox; i++)
		{
			// virtual voxel position
			int z = i / vSliceSize,
				y = (i - z * vSliceSize) / vw,
				x = (i - z * vSliceSize) % vw;

			// physical voxel position
			size_t ph_x = (size_t) (double(x) / aniso_x),
				ph_y = (size_t) (double(y) / aniso_y),
				ph_z = (size_t) (double(z) / aniso_z);
				i = ph_z * sliceSize + ph_y * fullH + ph_x;

			// Cache this pixel 
			feed_pixel_2_cache_3D_LR (x, y, z, dataI[i], vroi);

		}

		return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi' 
	// performing anisotropy correction
	//
	bool scan_trivial_wholevolume_anisotropic__OLD(
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

				// Cache this pixel
				feed_pixel_2_cache_LR(vc, vr, dataI[i], vroi);
			}
		}

		return true;
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

				// FIX (IO): --aniso* (explicit) or opt-in OME physical spacing selects the anisotropic path
				double ax, ay, az;
				if (! resolve_slide_anisotropy (env, sidx, ax, ay, az))
				{
					scanTrivialRois_3D (env, Pending, intens_fpath, label_fpath, t_index, channel);
				}
				else
				{
					scanTrivialRois_3D_anisotropic (env, Pending, intens_fpath, label_fpath, t_index, channel, ax, ay, az);

					// rescan and update ROI's AABB
					for (auto lbl : Pending)
					{
						LR& r = env.roiData[lbl];
						r.aabb.update_from_voxelcloud (r.raw_pixels_3D);
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

			// FIX (IO): --aniso* (explicit) or opt-in OME physical spacing selects the anisotropic path
			double ax, ay, az;
			if (! resolve_slide_anisotropy (env, sidx, ax, ay, az))
			{
				scanTrivialRois_3D (env, Pending, intens_fpath, label_fpath, t_index, channel);
			}
			else
			{
				scanTrivialRois_3D_anisotropic (env, Pending, intens_fpath, label_fpath, t_index, channel, ax, ay, az);

				// rescan and update ROI's AABB
				for (auto lbl : Pending)
				{
					LR& r = env.roiData[lbl];
					r.aabb.update_from_voxelcloud(r.raw_pixels_3D);
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