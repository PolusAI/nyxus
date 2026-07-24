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

	/// @brief Populates r.raw_voxels_NT by streaming the volume plane-by-plane. When 'wholevolume'
	/// is false (the segmented-ROI case), only voxels whose mask value equals r.label are kept --
	/// segPlane must be non-empty (a real mask loader). When 'wholevolume' is true, every voxel is
	/// kept regardless of any mask (there is none: workflow_3d_whole.cpp opens the loader with an
	/// empty label path, so ImageLoader::stream_volume_planes's segPlane is empty in that case).
	/// Shared by the segmented (processNontrivialRois_3D) and whole-volume out-of-core paths so
	/// both stream through the exact same primitive. Returns false (cloud cleared) if the loader
	/// can't deliver the volume plane-by-plane (e.g. NIfTI, a whole-4D-in-one-read format).
	bool populate_3d_voxel_cloud (ImageLoader& imlo, LR& r, size_t channel, size_t timeframe, bool wholevolume)
	{
		r.raw_voxels_NT.init (r.label, "raw_voxels_NT");

		const size_t W = imlo.get_full_width(),
			H = imlo.get_full_height();
		const uint32_t want = (uint32_t) r.label;

		bool streamed = imlo.stream_volume_planes (channel, timeframe, timeframe,
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
				r.raw_voxels_NT.end_slab (z);
			});

		if (! streamed)
			r.raw_voxels_NT.clear();
		return streamed;
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
	/// @param timeframe Timeframe being featurized (used for both intensity and mask)
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

			// Scan one label-intensity pair
			SlideProps p (intens_fpath, label_fpath);
			// Preserve the scanned HU-domain slide min/max + flag so load-time HU offset matches the
			// prescan (bare SlideProps defaults min to -1). Guarded on preserve_hu so the default path
			// is unchanged. Mirrors the 2D processNontrivialRois.
			if (env.fpimageOptions.preserve_hu() && r.slide_idx >= 0)
			{
				const SlideProps& scanned = env.dataset.dataset_props [r.slide_idx];
				p.min_preroi_inten = scanned.min_preroi_inten;
				p.max_preroi_inten = scanned.max_preroi_inten;
				p.preserve_hu = scanned.preserve_hu;
			}
			if (! env.theImLoader.open(p, env.fpimageOptions))
			{
				std::cout << "Terminating\n";
				return false;
			}

			// Populate the ROI's disk-backed voxel cloud by streaming the volume plane-by-plane.
			// Only this ROI's label voxels are written; z is preserved. Peak memory is two X*Y
			// planes (intensity + mask) plus this plane's ROI voxels, never the whole cube.
			if (! populate_3d_voxel_cloud (env.theImLoader, r, channel, timeframe, /*wholevolume=*/ false))
			{
				std::string erm = "Error: out-of-core featurization of oversized 3D ROI " + std::to_string(r.label)
					+ " is not supported for this input format (the volume is not delivered plane-by-plane). "
					+ "Segment into smaller ROIs, raise --ramLimit, or add RAM.";
#ifdef WITH_PYTHON_H
				throw std::runtime_error(erm);
#endif
				std::cerr << erm << "\n";
				continue;
			}

			//=== Reduce features over the streamed voxel cloud
			run_3d_ooc_features (env, r, env.theImLoader);

			//=== Clean the ROI's cache
			r.raw_voxels_NT.clear();

			#ifdef WITH_PYTHON_H
			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
			#endif
		}

		return true;
	}
}
