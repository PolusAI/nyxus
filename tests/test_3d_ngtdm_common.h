#pragma once

// Shared fixture for the 3D NGTDM tests: the phantom the oracle assertions run on, the settings
// recipes, and the mock 3D workflow that turns a phantom plus a settings vector into computed
// feature values.
//
// Fixtures only, no reference data (SPEC 6.3.1) -- the oracle file and the regression file each keep
// their own table beside the assertions that read it.

// What this header includes is what it spells. <string>, <tuple> and <vector> stay because nothing
// here supplies them explicitly -- they are reachable only by accident through roi_cache.h, which is
// the case the include-hygiene rule says to keep rather than strip. The three files that include
// this one repeat none of it.
#include <gtest/gtest.h>
#include <string>                             // the helpers' parameters and return types
#include <tuple>                              // get_3d_compat_ngtdm_phantom's return type
#include <vector>

#include "../src/nyx/features/3d_ngtdm.h"     // D3_NGTDM_feature, and SimpleCube / PixIntens with it
#include "../src/nyx/helpers/fsystem.h"       // fs::path, fs::exists
#include "test_main_nyxus.h"                  // agrees_gt, <cmath>, <iostream>, and the Environment / globals / roi_cache graph

// The 4x4x3 NGTDM compatibility phantom: one populated slice of 16 voxels between two all-zero
// slices, every voxel labelled 57. Its intensities are the discrete levels 0..5, and binWidth=1
// discretisation maps those to 1..6 -- the same levels Nyxus' zero-min correction produces, which is
// what makes a PyRadiomics run and a Nyxus run directly comparable on it.
//
// The mask has no background voxel at all, which PyRadiomics' own loader rejects; the generator
// reaches RadiomicsNGTDM directly for that reason
// (tests/vetting/audit/ngtdm_3d_pyradiomics_vetting_report.md).
static std::tuple<std::string, std::string, int> get_3d_compat_ngtdm_phantom()
{
	// physical paths of the phantoms
	fs::path this_fpath(__FILE__);
	fs::path pp = this_fpath.parent_path();

	fs::path f1("/data/nifti/compat_int/compat_int_ngtdm_3d.nii");
	fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

	fs::path f2("/data/nifti/compat_seg/compat_seg_ngtdm_3d.nii");
	fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

	std::string ipath = i_phys_path.string(),
		mpath = m_phys_path.string();

	return { ipath, mpath, 57 };
}

// The settings a 3D NGTDM assertion runs on. 'ngtdm_greydepth' is the family's own binning
// (0 = none, i.e. the raw levels; a positive value is a MATLAB-style bin count) and 'ngtdm_radius'
// is the Chebyshev radius of the neighbourhood.
//
// NGTDM_RADIUS must be >= 1. At 0 the neighbourhood loop visits only the centre voxel, which it
// skips, so no voxel gets a dependency, the matrix stays empty and every feature comes back NaN.
// A settings vector built here starts zero-filled, so every caller states the radius; the default a
// real run gets is Environment::compile_feature_settings()' business, and
// test_3d_ngtdm_default_radius_mechanics is what holds it to 1.
static Fsettings make_ngtdm3d_settings (int greydepth, int ngtdm_greydepth, int ngtdm_radius)
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
	s[(int)NyxSetting::NGTDM_GREYDEPTH].ival = ngtdm_greydepth;
	s[(int)NyxSetting::NGTDM_RADIUS].ival = ngtdm_radius;
	return s;
}

// The NGTD matrix a run built, copied out of the feature object: the grey levels it indexed, their
// n_i, p_i and s_i, and the count of voxels having a neighbour. The five features are contractions
// of this table, so an assertion on it is an assertion on what produced them.
struct Ngtdm3dMatrix
{
	std::vector<PixIntens> I;
	std::vector<int> N;
	std::vector<double> P, S;
	int Nvp = 0;
};

// Mocks the 3D workflow on one phantom ROI, copying the computed feature values into 'fvals' (indexed
// by Nyxus 3D feature code) and the ROI's voxel cube into 'cube'. One place for the four-step prescan
// / metrics / voxel-cloud / buffer sequence, so the oracle and regression assertions cannot drift
// apart in it.
//
// The cube is handed back so an assertion about the NGTD matrix works from the voxels this
// featurisation actually read, rather than from a second, hand-written copy of them: two assertions
// that describe one run have to be reading one run (SPEC 5.2). 'matrix', when asked for, is the
// matrix that run built -- the state the five feature values came out of, not a rebuild of it.
static void extract_3d_ngtdm (
	std::vector<std::vector<double>>& fvals,
	SimpleCube<PixIntens>& cube,
	const std::string& ipath,
	const std::string& mpath,
	int label,
	const Fsettings& s,
	Ngtdm3dMatrix* matrix = nullptr)
{
	ASSERT_TRUE(fs::exists(ipath));
	ASSERT_TRUE(fs::exists(mpath));

	Environment e;

	// (1) slide -> dataset -> prescan
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
	ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.fpimageOptions, e.resultOptions.need_annotation()));
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
	D3_NGTDM_feature f;
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	fvals = r.fvals;
	cube = r.aux_image_cube;

	if (matrix)
		*matrix = { f.get_levels(), f.get_N(), f.get_P(), f.get_S(), f.get_Nvp() };
}

// Resolves a 3D feature name to its code and checks it is the one the caller expects, so a renamed
// or re-ordered enum cannot silently move an assertion onto a different feature.
static void resolve_3d_ngtdm_fcode (int& fcode, const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	Environment e;
	fcode = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
	ASSERT_TRUE((int)expecting_fcode == fcode);
}
