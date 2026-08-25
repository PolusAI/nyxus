#pragma once

// Shared fixture for the 3D GLSZM tests: the settings recipes and the mock 3D workflow that turns a
// phantom plus a settings vector into computed feature values.
//
// Fixtures only, no reference data (SPEC 6.3.1) -- the oracle file and the regression file each keep
// their own table beside the assertions that read it.

// What this header includes is what it spells. <string>, <tuple> and <vector> stay because nothing
// here supplies them explicitly -- they are reachable only by accident through roi_cache.h, which is
// the case the include-hygiene rule says to keep rather than strip. The files that include this one
// repeat none of it.
#include <gtest/gtest.h>
#include <string>                             // the helpers' parameters and return types
#include <tuple>                              // the phantom accessors' return type
#include <vector>

#include "../src/nyx/features/3d_glszm.h"     // D3_GLSZM_feature, and SimpleCube / PixIntens with it
#include "../src/nyx/helpers/fsystem.h"       // fs::path, fs::exists
#include "test_main_nyxus.h"                  // agrees_gt, <cmath>, <iostream>, and the Environment / globals / roi_cache graph

// Both 3D phantoms are defined once, in test_3d_glcm_pyradiomics.h, and every 3D family reaches them
// by declaration -- test_all.cc compiles the headers into one translation unit. get_3d_compat_phantom
// is bench_compat_liver_3d (compat_int/compat_int_mri.nii + compat_seg/compat_seg_liver.nii, label 1),
// get_3d_segmented_phantom is bench_ut_phantom_3d (phantoms/ut_inten.nii + ut_mask57.nii, label 57).
static std::tuple<std::string, std::string, int> get_3d_compat_phantom();
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// The settings a 3D GLSZM assertion runs on. 'glszm_greydepth' is the family's own binning and its
// sign selects the scheme: negative is a PyRadiomics-style bin count (magnitude = binCount), positive
// is a MATLAB-style level count, and 0 is no binning at all, which is the IBSI reading of the raw
// levels. A settings vector built here starts zero-filled, so every caller states its own value.
static Fsettings make_glszm3d_settings (int greydepth, int glszm_greydepth)
{
	Fsettings s;
	s.resize((int)NyxSetting::__COUNT__);
	s[(int)NyxSetting::SOFTNAN].rval = 0.0;
	s[(int)NyxSetting::TINY].rval = 0.0;
	s[(int)NyxSetting::SINGLEROI].bval = false;
	s[(int)NyxSetting::GREYDEPTH].ival = greydepth;
	s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
	s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
	s[(int)NyxSetting::USEGPU].bval = false;
	s[(int)NyxSetting::VERBOSLVL].ival = 0;
	s[(int)NyxSetting::IBSI].bval = false;
	s[(int)NyxSetting::GLSZM_GREYDEPTH].ival = glszm_greydepth;
	return s;
}

// Mocks the 3D workflow on one phantom ROI, copying the computed feature values into 'fvals' (indexed
// by Nyxus 3D feature code) and the ROI's voxel cube and intensity extrema into 'cube' / 'lo' / 'hi'.
// One place for the four-step prescan / metrics / voxel-cloud / buffer sequence, so the oracle and
// regression assertions cannot drift apart in it.
//
// The cube and the extrema are handed back so an assertion about the size-zone matrix works from the
// voxels this featurisation actually read and bins them exactly as calculate() did, rather than from
// a second, hand-written copy of them: two assertions that describe one run have to be reading one
// run (SPEC 5.2).
static void extract_3d_glszm (
	std::vector<std::vector<double>>& fvals,
	SimpleCube<PixIntens>& cube,
	PixIntens& lo,
	PixIntens& hi,
	const std::string& ipath,
	const std::string& mpath,
	int label,
	const Fsettings& s)
{
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	Environment e;

	// (1) slide -> dataset -> prescan
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
	ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();

	// (2) properties of specific ROIs sitting in 'e.uniqueLabels'
	clear_slide_rois(e.uniqueLabels, e.roiData);
	ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

	// (3) voxel clouds
	std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
	ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

	// (4) buffers
	ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

	// (5) feature extraction
	LR& r = e.roiData[label];
	ASSERT_NO_THROW(r.initialize_fvals());
	D3_GLSZM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	fvals = r.fvals;
	cube = r.aux_image_cube;
	lo = r.aux_min;
	hi = r.aux_max;
}

// Resolves a 3D feature name to its code and checks it is the one the caller expects, so a renamed
// or re-ordered enum cannot silently move an assertion onto a different feature.
static void resolve_3d_glszm_fcode (int& fcode, const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	Environment e;
	fcode = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	ASSERT_TRUE((int)expecting_fcode == fcode);
}
