#pragma once

// Shared fixture for the 3D GLDZM tests: the phantom the oracle assertions run on, the settings
// recipes, and the mock 3D workflow that turns a phantom plus a settings vector into computed
// feature values.
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

#include "../src/nyx/features/3d_gldzm.h"     // D3_GLDZM_feature, and SimpleCube / PixIntens with it
#include "../src/nyx/helpers/fsystem.h"       // fs::path, fs::exists
#include "test_main_nyxus.h"                  // agrees_gt, <cmath>, <iostream>, and the Environment / globals / roi_cache graph

// bench_ut57_3d, defined once in test_3d_glcm_pyradiomics.h and reached by declaration -- test_all.cc
// compiles the headers into one translation unit. It is phantoms/ut_inten.nii + ut_mask57.nii,
// label 57, the CT-like volume the family's drift guard runs on.
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// bench_compat_gldzm_3d -- the 16x16x16 GLDZM compatibility phantom, all of it labelled 57 inside a
// two-voxel background margin. Both sides of a MIRP comparison read its voxel values as grey levels
// directly: Nyxus at IBSI=true does no binning, and MIRP is run with base_discretisation_method
// "none", so the discretisation the two tools would otherwise disagree about is out of the
// comparison (tests/vetting/audit/gldzm_3d_mirp_vetting_report.md).
//
// The ROI is a 12x12x12 cube with a 6x6x6 corner cut out of it, filled with 2x2x2 bricks whose grey
// level is 1 + 4*(bz%2) + 2*(by%2) + (bx%2) over the brick's coordinates, and the brick at (3,3,3)
// carries level 1 instead of the 8 that rule gives it. Every property this family gets wrong is
// separated by one of those three choices:
//
//   the background margin  the mask is what says which voxels are the ROI's, and the phantom has
//                          background on all six sides to be wrong about
//   the corner cut         it makes the ROI non-convex, so for voxels near the cut the shortest way
//                          out is diagonal and a distance measured along the axes overstates it
//   the (3,3,3) implant    that brick touches eight level-1 bricks at a corner and nowhere else, so
//                          it is one zone with them at 26-connectivity and a zone of its own at 6-
//                          or 18-connectivity
static std::tuple<std::string, std::string, int> get_3d_compat_gldzm_phantom()
{
	// physical paths of the phantoms
	fs::path this_fpath(__FILE__);
	fs::path pp = this_fpath.parent_path();

	fs::path f1("/data/nifti/compat_int/compat_int_gldzm_3d.nii");
	fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

	fs::path f2("/data/nifti/compat_seg/compat_seg_gldzm_3d.nii");
	fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

	std::string ipath = i_phys_path.string(),
		mpath = m_phys_path.string();

	return { ipath, mpath, 57 };
}

// bench_gldzm_zerolevel_3d -- an 8x8x8 volume whose 4x4x4 ROI carries raw intensity 0 on the voxels
// with an even x+y+z and 5 on the rest, inside a two-voxel background margin. Its zeros are ROI
// voxels, and the binning schemes that do not remap a zero hand them to the GLDZM as grey level 0,
// which is not a valid level. It is the family's only fixture that reaches that case: the
// compatibility phantom's levels are 1..8 and bench_ut57_3d is binned MATLAB-style, which sends 0 to
// level 1.
//
// Built by oracles/gen_gldzm3d_mirp.py --write-phantom, like the compatibility phantom, and checked
// against its rule on every ordinary run of that generator. It carries no oracle goldens: MIRP has
// the same problem with a level 0 in its input, so test_3d_gldzm_mechanics.h derives what to expect
// by hand instead.
static std::tuple<std::string, std::string, int> get_3d_gldzm_zerolevel_phantom()
{
	// physical paths of the phantoms
	fs::path this_fpath(__FILE__);
	fs::path pp = this_fpath.parent_path();

	fs::path f1("/data/nifti/phantoms/gldzm_zerolevel_inten.nii");
	fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

	fs::path f2("/data/nifti/phantoms/gldzm_zerolevel_mask.nii");
	fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

	std::string ipath = i_phys_path.string(),
		mpath = m_phys_path.string();

	return { ipath, mpath, 57 };
}

// The settings a 3D GLDZM assertion runs on. The family reads two of them, and they are not
// independent: at IBSI=true prepare_GLDZM_matrix_kit overwrites the grey depth with 0 whatever was
// passed, which is the no-binning reading of the raw levels. A settings vector built here starts
// zero-filled, so every caller states its own values.
static Fsettings make_gldzm3d_settings (int greydepth, bool ibsi)
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
	s[(int)NyxSetting::IBSI].bval = ibsi;
	return s;
}

// Mocks the 3D workflow on one phantom ROI, copying the computed feature values into 'fvals' (indexed
// by Nyxus 3D feature code). One place for the four-step prescan / metrics / voxel-cloud / buffer
// sequence, so the oracle and regression assertions cannot drift apart in it.
//
static void extract_3d_gldzm (
	std::vector<std::vector<double>>& fvals,
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
	D3_GLDZM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	fvals = r.fvals;
}
