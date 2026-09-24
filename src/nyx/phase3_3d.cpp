#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
#endif

#include "environment.h"
#include "feature_mgr.h"
#include "globals.h"
#include "features/3d_intensity.h"
#include "features/3d_surface.h"
#include "features/3d_glcm.h"
#include "features/3d_gldm.h"
#include "features/3d_ngldm.h"
#include "features/3d_ngtdm.h"
#include "features/3d_glrlm.h"
#include "features/3d_glszm.h"
#include "features/3d_gldzm.h"
#include "features/pixel.h"
#include "helpers/helpers.h"		// Nyxus::near_eq
#include "helpers/timing.h"

namespace Nyxus
{
	/// @brief Returns true if 'f' is one of the 3D feature classes whose osized_calculate() streams
	/// from the disk-backed voxel cloud (r.raw_voxels_NT) instead of reading the in-memory cube.
	/// Every other 3D FeatureMethod (e.g. a future feature added without a streaming path) would
	/// silently read an empty cube if allowed through, so processNontrivialRois_3D()'s per-feature
	/// guard calls this and fails loudly on false. Kept as a standalone function (rather than inline
	/// in the guard) so the allow-list itself is unit-testable independent of the streaming plumbing.
	bool is_3d_ooc_supported (FeatureMethod* f)
	{
		return dynamic_cast<D3_VoxelIntensityFeatures*>(f) != nullptr
			|| dynamic_cast<D3_SurfaceFeature*>(f) != nullptr
			|| dynamic_cast<D3_GLCM_feature*>(f) != nullptr
			|| dynamic_cast<D3_GLDM_feature*>(f) != nullptr
			|| dynamic_cast<D3_NGLDM_feature*>(f) != nullptr
			|| dynamic_cast<D3_NGTDM_feature*>(f) != nullptr
			|| dynamic_cast<D3_GLRLM_feature*>(f) != nullptr
			|| dynamic_cast<D3_GLSZM_feature*>(f) != nullptr
			|| dynamic_cast<D3_GLDZM_feature*>(f) != nullptr;
	}

	std::string ooc_unstreamable_reason (ImageLoader& imlo)
	{
		// Not "this format cannot stream": a chunked OME-Zarr reaches here too, when its Z chunk
		// spans the whole volume, and re-chunking it is the fix. What the refusal turns on is the
		// number of planes one read delivers, and which of the pair delivers them -- the mask can
		// be the unstreamable one, and it is the file the user would have to re-chunk.
		size_t planes = 0;
		bool of_mask = false;
		if (! imlo.unstreamable_read (planes, of_mask))
			return "";	// this pair streams -- the callers read that off the empty string

		return std::string ("a single read of the ") + (of_mask ? "mask" : "intensity")
			+ " input delivers " + std::to_string (planes)
			+ " planes at once, so streaming it cannot bound the footprint below that. Re-chunk that"
			  " input along Z (OME-Zarr), convert it to a Z-stack of TIFF planes, raise --ramLimit,"
			  " or add RAM";
	}

	/// @brief Populates r.raw_voxels_NT by streaming the volume plane-by-plane. When 'wholevolume'
	/// is false (the segmented-ROI case), only voxels whose mask value equals r.label are kept --
	/// segPlane must be non-empty (a real mask loader). When 'wholevolume' is true, every voxel is
	/// kept regardless of any mask (there is none: workflow_3d_whole.cpp opens the loader with an
	/// empty label path, so ImageLoader::stream_volume_planes's segPlane is empty in that case).
	/// (ax,ay,az) is the voxel spacing Nyxus::resolve_anisotropy resolved for the slide: on a
	/// non-cubic grid the cloud is resampled onto the virtual grid the in-RAM scans build, and
	/// r.aabb and r.aux_area are updated to that geometry (the physical ones the prescan recorded
	/// describe a different extent and a different voxel count).
	/// Shared by the segmented (processNontrivialRois_3D) and whole-volume out-of-core paths so
	/// both stream through the exact same primitive. The file paths name the pair in the messages
	/// Nyxus::stream_volume_checked raises, which is how this path gets the intensity-vs-mask voxel
	/// count check the in-RAM passes make; 'mask_fpath' is empty in whole-volume mode, which has no
	/// mask to check.
	bool populate_3d_voxel_cloud (ImageLoader& imlo, LR& r, size_t channel, size_t timeframe, bool wholevolume,
		double ax, double ay, double az, const std::string& intens_fpath, const std::string& mask_fpath)
	{
		// A loader that hands the whole cube back in one read (NIfTI) cannot be streamed within a
		// bounded footprint: a "tile layer" of it IS the volume. Out-of-core exists to bound that
		// footprint, so decline rather than allocate the cube the caller could not afford.
		if (! imlo.streams_bounded())
			return false;

		r.raw_voxels_NT.init (r.label, "raw_voxels_NT");

		const size_t W = imlo.get_full_width(),
			H = imlo.get_full_height(),
			D = imlo.get_full_depth();
		const uint32_t want = (uint32_t) r.label;

		const bool anisotropic = ! (Nyxus::near_eq (ax, 1.0) && Nyxus::near_eq (ay, 1.0) && Nyxus::near_eq (az, 1.0));
		if (! anisotropic)
		{
			return stream_volume_checked (imlo, channel, timeframe, intens_fpath, mask_fpath,
				[&](size_t z, const std::vector<uint32_t>& intPlane, const std::vector<uint32_t>& segPlane)
				{
					r.raw_voxels_NT.begin_slab (z);
					for (size_t y = 0; y < H; y++)
						for (size_t x = 0; x < W; x++)
						{
							size_t i = y * W + x;
							if (wholevolume || segPlane[i] == want)
								r.raw_voxels_NT.add_voxel (Pixel3(x, y, z, intPlane[i]));
						}
				});
		}

		// The virtual grid of the in-RAM anisotropic scans: the whole-volume scan takes the
		// physical voxel a virtual one falls in (truncated, clamped at the far edge), the
		// segmented scan the nearest one, and skips a virtual voxel that lands outside.
		const size_t vW = (size_t) (double(W) * ax),
			vH = (size_t) (double(H) * ay),
			vD = (size_t) (double(D) * az);
		bool any = false;
		StatsInt minx = 0, maxx = 0, miny = 0, maxy = 0, minz = 0, maxz = 0;

		size_t vz = 0;	// the next virtual plane to fill
		if (! stream_volume_checked (imlo, channel, timeframe, intens_fpath, mask_fpath,
			[&](size_t z, const std::vector<uint32_t>& intPlane, const std::vector<uint32_t>& segPlane)
			{
				for (; vz < vD; vz++)
				{
					const size_t pz = wholevolume
						? (std::min<size_t>) ((size_t) (double(vz) / az), D - 1)
						: (size_t) (vz / az + 0.5);
					if (pz > z)
						break;		// its physical plane is still to come
					if (pz < z)
						continue;

					r.raw_voxels_NT.begin_slab (vz);
					for (size_t vy = 0; vy < vH; vy++)
						for (size_t vx = 0; vx < vW; vx++)
						{
							size_t px, py;
							if (wholevolume)
							{
								px = (std::min<size_t>) ((size_t) (double(vx) / ax), W - 1);
								py = (std::min<size_t>) ((size_t) (double(vy) / ay), H - 1);
							}
							else
							{
								px = (size_t) (vx / ax + 0.5);
								py = (size_t) (vy / ay + 0.5);
								if (px >= W || py >= H)
									continue;
							}

							const size_t i = py * W + px;
							if (! (wholevolume || segPlane[i] == want))
								continue;

							r.raw_voxels_NT.add_voxel (Pixel3(vx, vy, vz, intPlane[i]));

							if (! any)
							{
								any = true;
								minx = maxx = (StatsInt) vx; miny = maxy = (StatsInt) vy; minz = maxz = (StatsInt) vz;
							}
							else
							{
								minx = (std::min) (minx, (StatsInt) vx); maxx = (std::max) (maxx, (StatsInt) vx);
								miny = (std::min) (miny, (StatsInt) vy); maxy = (std::max) (maxy, (StatsInt) vy);
								minz = (std::min) (minz, (StatsInt) vz); maxz = (std::max) (maxz, (StatsInt) vz);
							}
						}
				}
			}))
			return false;

		// the resampled cloud's own extent and voxel count, as the in-RAM anisotropic scans
		// recompute them after resampling
		if (any)
		{
			r.aabb.init_x (minx); r.aabb.update_x (maxx);
			r.aabb.init_y (miny); r.aabb.update_y (maxy);
			r.aabb.init_z (minz); r.aabb.update_z (maxz);
		}
		r.aux_area = (unsigned int) r.raw_voxels_NT.size();
		return true;
	}
	/// @brief Runs every requested feature's out-of-core path over an already-populated
	/// r.raw_voxels_NT, guarded by is_3d_ooc_supported(); writes results into r.fvals via
	/// save_value(). Shared by the segmented and whole-volume out-of-core paths. 'imloader' is
	/// passed through to match FeatureMethod's signature but is not read by any of the streaming
	/// 3D features (they read raw_voxels_NT exclusively), so either path's own loader instance works.
	void run_3d_ooc_features (Environment& env, LR& r, ImageLoader& imloader)
	{
		int nrf = env.theFeatureMgr.get_num_requested_features();
		for (int i = 0; i < nrf; i++)
		{
			auto f = env.theFeatureMgr.get_feature_method (i);

			try
			{
				// Every 3D feature that streams from raw_voxels_NT is covered by
				// is_3d_ooc_supported(); anything else would silently read an empty cube, so fail
				// loudly per-feature instead of emitting wrong values. Under the CLI this logs and
				// moves on (the supported features still compute); under Python it raises.
				if (! is_3d_ooc_supported (f))
					throw std::runtime_error("feature '" + f->feature_info
						+ "' is not yet supported out-of-core for oversized 3D ROIs; "
						+ "segment into smaller ROIs, raise --ramLimit, or add RAM");

				const Fsettings& s = env.get_feature_settings (typeid(f));
				f->osized_scan_whole_image (r, s, env.dataset, imloader);
			}
			catch (std::exception const& e)
			{
				std::string erm = "Error while computing feature " + f->feature_info + " over oversized 3D ROI " + std::to_string(r.label) + " : " + e.what();
#ifdef WITH_PYTHON_H
				throw std::runtime_error(erm);
#endif
				std::cerr << erm << "\n";
			}

			f->cleanup_instance();
		}
	}

	/// @brief Processes oversized volumetric (3D) ROIs out-of-core: streams the voxel cloud to
	///        disk one Z-plane at a time (bounded memory) instead of holding the whole cube.
	/// @param nontrivRoiLabels Labels of ROIs whose in-memory footprint exceeds the RAM limit
	/// @param intens_fpath Intensity image path
	/// @param label_fpath Mask image path
	/// @param channel Intensity channel being featurized
	/// @param timeframe Timeframe being featurized (the mask is read at the frame it pairs with)
	/// @return Success status
	///
	bool processNontrivialRois_3D (Environment& env, const std::vector<int>& nontrivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t channel, size_t timeframe)
	{
		// Sort labels for reproducibility with the trivial counterpart
		auto L = nontrivRoiLabels;
		std::sort (L.begin(), L.end());

		for (auto lab : L)
		{
			LR& r = env.roiData[lab];

			VERBOSLVL1 (env.get_verbosity_level(), std::cout << "processing oversized 3D ROI " << lab << "\n");

			// Scan one label-intensity pair, mapped into the intensity domain the prescan recorded
			// for this slide, as the in-RAM passes are
			if (! open_scanned_pair (env, r.slide_idx, intens_fpath, label_fpath))
				return false;

			// Populate the ROI's disk-backed voxel cloud by streaming the volume plane-by-plane.
			// Only this ROI's label voxels are written; z is preserved. Peak memory is one tile
			// layer of intensity and mask plus this plane's ROI voxels, never the whole cube.
			// The cloud is resampled onto the same virtual grid the in-RAM scans build.
			double ax, ay, az;
			resolve_slide_anisotropy (env, (size_t) r.slide_idx, ax, ay, az);
			if (! populate_3d_voxel_cloud (env.theImLoader, r, channel, timeframe, /*wholevolume=*/ false, ax, ay, az,
				intens_fpath, label_fpath))
			{
				// Why this loader has no bounded streaming path, read before the pair is released -- the
				// answer comes from the loader itself, and is empty when the pair streams (this refusal
				// then belongs to the shape check, which reports itself inside the streaming call). The
				// refusal leaves by a throw under Python and by the return on the CLI, and the pair has
				// to be freed on either.
				const std::string why = ooc_unstreamable_reason (env.theImLoader);
				env.theImLoader.close();

				// Fail the run rather than carry on: continuing would leave this ROI's features at
				// their initialized zeros and write that row out as if it had been measured, with
				// nyxus exiting 0. The whole-volume counterpart returns false for the same reason.
				// A shape mismatch between the intensity volume and its mask reports itself inside
				// the streaming call, so only the unstreamable loader is this site's to explain.
				if (! why.empty())
				{
					std::string erm = "Error: cannot featurize oversized 3D ROI " + std::to_string(r.label)
						+ " out-of-core: " + why
						+ ". Segmenting into smaller ROIs keeps them on the in-RAM path too.";
#ifdef WITH_PYTHON_H
					throw std::runtime_error(erm);
#endif
					std::cerr << erm << "\n";
				}
				return false;
			}

			//=== Reduce features over the streamed voxel cloud
			run_3d_ooc_features (env, r, env.theImLoader);

			//=== Clean the ROI's cache
			r.raw_voxels_NT.clear();

			// Close the image pair this ROI was streamed from: the loop opens one per ROI, and
			// ImageLoader::open() allocates loaders without releasing any it already holds
			env.theImLoader.close();

			#ifdef WITH_PYTHON_H
			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
			#endif
		}

		return true;
	}
}
