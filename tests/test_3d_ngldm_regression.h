#pragma once

#include <gtest/gtest.h>
#include <iomanip>
#include "test_3d_ngldm_common.h"        // the fixture, and the headers the mocked 3D workflow needs
#include "test_ref_vals.h"               // ref_vals_map, and the <string> / <vector> it already includes

// Drift guards on the segmented phantom (ut_inten.nii + ut_mask57.nii, label 57) at 64 grey levels,
// ibsi=false. Nyxus' own output, so these claim no oracle (SPEC 1).
//
// Regenerate with test_3d_ngldm_dump_regression() below.
//
// THREE features, not the family's nineteen. The other sixteen are asserted against MIRP in
// test_3d_ngldm_mirp.h, on this same fixture and config, so a snapshot of them here would pin the
// same Nyxus run to a second literal and add no path or config coverage. These three cannot be
// judged by any tool:
//
//   3NGLDM_DCP  -- Nyxus hard-codes f_DCP = 1, and MIRP returns 1 on any input where every voxel has
//                  a same-level neighbour, so an oracle row would agree at a degenerate constant.
//   3NGLDM_GLM  -- MIRP's NGLDM emits no gl_mean column.
//   3NGLDM_DCM  -- MIRP's NGLDM emits no dc_mean column.
static const ref_vals_map<double> ngldm_3d_regression_ref_vals{
		{ "3NGLDM_DCP",	1.0 },
		{ "3NGLDM_GLM",	41.474725979477633 },
		{ "3NGLDM_DCM",	5.10127098880597 }
};

void assert_3d_ngldm_feature_regression (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
	// the table is const and read through .at(), so a missing key throws rather than being
	// default-inserted as a 0 golden and compared against; check it up front to fail by name
	ASSERT_TRUE(ngldm_3d_regression_ref_vals.count(fname) > 0) << fname;

	double atot = 0.0;
	calculate_3d_ngldm_feature_value (fname, expecting_fcode, atot);

	// verdict. frac_tolerance = 1e9, i.e. rel=1e-9: Nyxus' own values pinned to full precision, so the
	// guard catches any change at all. That is tighter than the rel=1e-3 the MIRP oracle beside it
	// asserts, and deliberately so: an oracle band has to survive every CI platform's float, while a
	// snapshot of this build's own output does not. Why this band and not a looser one:
	// tests/vetting/audit/ngldm_3d_golden_regen.md, "Regression drift guards".
	ASSERT_TRUE(agrees_gt(atot, ngldm_3d_regression_ref_vals.at(fname), 1e9))
		<< fname << " actual=" << std::setprecision(17) << atot;
}

// Regenerates every golden in ngldm_3d_regression_ref_vals at full precision, in the exact shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_NGLDM_DUMP_REGRESSION*
// and paste the output over the table above. It uses the same settings the shared assert helper
// sets, so the two cannot drift apart.
void test_3d_ngldm_dump_regression()
{
	auto [ipath, mpath, label] = get_3d_segmented_phantom();
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	Environment e;
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
	ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();

	clear_slide_rois(e.uniqueLabels, e.roiData);
	ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

	std::vector<int> batch = { label };
	ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
	ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

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
	s[(int)NyxSetting::IBSI].bval = false;

	LR& r = e.roiData[label];
	ASSERT_NO_THROW(r.initialize_fvals());
	D3_NGLDM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	std::cout << "[3DNGLDM-REGEN]\n";
	for (const auto& nv : ngldm_3d_regression_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DNGLDM-REGEN]\t\t{ \"" << nv.first << "\",\t"
		          << std::setprecision(17) << r.fvals[fcode][0] << " },\n";
	}
}

void test_3d_ngldm_dcp_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCP", Feature3D::NGLDM_DCP);
}

void test_3d_ngldm_glm_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_GLM", Feature3D::NGLDM_GLM);
}

void test_3d_ngldm_dcm_regression() {
	assert_3d_ngldm_feature_regression ("3NGLDM_DCM", Feature3D::NGLDM_DCM);
}

