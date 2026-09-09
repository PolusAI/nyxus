#pragma once

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>
#include "../src/nyx/environment.h"           // Environment
#include "../src/nyx/feature_settings.h"      // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"            // Nyxus::Feature3D
#include "../src/nyx/globals.h"               // clear_slide_rois, gatherRoisMetrics_3D, scanTrivialRois_3D, allocateTrivialRoisBuffers_3D
#include "../src/nyx/roi_cache.h"             // LR
#include "../src/nyx/slideprops.h"            // SlideProps, scan_slide_props
#include "../src/nyx/features/3d_ngldm.h"     // D3_NGLDM_feature
#include "../src/nyx/helpers/fsystem.h"       // fs::exists
#include "test_main_nyxus.h"                  // agrees_gt, which every judging file in this family calls

// get_3d_segmented_phantom is bench_ut57_3d (phantoms/ut_inten.nii + ut_mask57.nii, label 57).
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// Fixture (SPEC 6.3.1): loads the segmented phantom, runs D3_NGLDM_feature at GREYDEPTH=64 /
// IBSI=false and hands one feature's value back. Both judging files in this family -- the MIRP
// oracle and the regression drift guards -- read the SAME Nyxus run through this helper, which is
// what makes the two comparisons in
// tests/vetting/audit/ngldm_3d_mirp_vetting_report.md comparisons of one number rather than of two
// runs that could drift apart. Goldens stay in their respective oracle/regression files.
void calculate_3d_ngldm_feature_value (const std::string& fname, const Nyxus::Feature3D& expecting_fcode, double& out, bool ibsi = false)
{
	// get segment info
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	// mock the 3D workflow
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

	// (5) feature settings. IBSI is the family's second axis and reaches to_grayscale as
	// disable_binning, so ibsi=false is the GREYDEPTH=64 ladder both mirp_samelevels and the
	// regression pins use, and ibsi=true turns binning off and makes the raw intensity the grey
	// level -- the Nyxus half of ngldm3d.mirp_ibsi_rawlevels.
	Fsettings s;
	s.resize((int)NyxSetting::__COUNT__);
	s[(int)NyxSetting::SOFTNAN].rval = 0.0;
	s[(int)NyxSetting::TINY].rval = 0.0;
	s[(int)NyxSetting::SINGLEROI].bval = false;
	s[(int)NyxSetting::GREYDEPTH].ival = 64;
	s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
	s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
	s[(int)NyxSetting::USEGPU].bval = false;
	s[(int)NyxSetting::VERBOSLVL].ival = 0;
	s[(int)NyxSetting::IBSI].bval = ibsi;

	// (6) feature extraction

	// make it find the feature code by name
	int fcode = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	// ... and that it's the feature we expect
	ASSERT_TRUE((int)expecting_fcode == fcode);

	// extract the feature
	LR& r = e.roiData[label];
	ASSERT_NO_THROW(r.initialize_fvals());
	D3_NGLDM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));

	// (7) saving values

	f.save_value(r.fvals);

	// we have just 1 value, no need to aggregate subfeatures
	out = r.fvals[fcode][0];
}
