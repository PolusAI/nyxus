#pragma once

#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "../src/nyx/environment.h"				// Environment
#include "../src/nyx/feature_settings.h"			// Fsettings
#include "../src/nyx/featureset.h"				// Nyxus::Feature3D
#include "../src/nyx/helpers/fsystem.h"			// fs::exists
#include "../src/nyx/slideprops.h"				// SlideProps, scan_slide_props
#include "../src/nyx/features/3d_intensity.h"	// D3_VoxelIntensityFeatures
#include "test_main_nyxus.h"					// agrees_gt; also supplies LR (roi_cache.h) and the 3D workflow (globals.h)
#include "test_ref_vals.h"						// ref_vals_map

// Shared extraction for the 3D first-order family: the oracle file and the drift-guard file assert
// the same values from the same run, so the workflow mock lives here rather than in either of them.
//
// Defined in test_3d_glcm_pyradiomics.h. Forward-declared rather than defined so that this header
// can be included alongside it in the single test_all.cc translation unit.
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// Settings are left default-constructed on purpose: that leaves STNGS_MISSING true, which is what
// puts the histogram-derived statistics on DEFAULT_NUM_HISTO_BINS (src/nyx/constants.h) rather than
// on a bin count a test picked. The reference tables are generated against that same bin count.
static void assert_3d_firstorder_feature (
	const std::string& fname,
	const Nyxus::Feature3D& expecting_fcode,
	double expected,
	double frac_tolerance)
{
	Fsettings s;

	// segment to measure
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	ASSERT_TRUE (fs::exists(ipath));
	ASSERT_TRUE (fs::exists(mpath));

	// mock the 3D workflow
	Environment e;

	// (1) slide -> dataset -> prescan
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back (ipath, mpath);
	ASSERT_TRUE (scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();

	// (2) properties of specific ROIs sitting in 'e.uniqueLabels'
	clear_slide_rois (e.uniqueLabels, e.roiData);
	ASSERT_TRUE (gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

	// (3) voxel clouds
	std::vector<int> batch = { label };	// expecting this roi label after metrics gathering
	ASSERT_TRUE (scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

	// (4) buffers
	ASSERT_NO_THROW (allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

	// (5) feature extraction

	// make it find the feature code by name
	int fcode = -1;
	ASSERT_TRUE (e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	// ... and that it's the feature we expect
	ASSERT_TRUE ((int)expecting_fcode == fcode);

	LR& r = e.roiData[label];
	ASSERT_NO_THROW (r.initialize_fvals());
	D3_VoxelIntensityFeatures f;
	ASSERT_NO_THROW (f.calculate(r, s, e.dataset));

	// (6) verdict -- no subfeatures, so subfeature [0]
	f.save_value (r.fvals);
	ASSERT_TRUE (agrees_gt(r.fvals[fcode][0], expected, frac_tolerance)) << fname;
}
