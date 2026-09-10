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
// get_3d_segmented_phantom is bench_ut57_3d (phantoms/ut_inten.nii + ut_mask57.nii, label 57).
static std::tuple<std::string, std::string, int> get_3d_compat_phantom();
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// The settings a 3D GLSZM assertion runs on. 'glszm_greydepth' is the family's own binning and its
// sign selects the scheme: negative is a PyRadiomics-style bin count (magnitude = binCount), positive
// is a MATLAB-style level count, and 0 is no binning at all, which is the IBSI reading of the raw
// levels. A settings vector built here starts zero-filled, so every caller states its own value.
//
// 'ibsi' is the second of the two settings calculate() reads, and it is not independent of the first:
// at IBSI=true calculate() overwrites the family's binning with 0 whatever was passed, which is the
// whole content of that half of the config matrix (tests/vetting/matrix/glszm3d.md).
static Fsettings make_glszm3d_settings (int greydepth, int glszm_greydepth, bool ibsi = false)
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
	const Fsettings& s,
	D3_GLSZM_feature& f)
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
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);

	fvals = r.fvals;
	cube = r.aux_image_cube;
	lo = r.aux_min;
	hi = r.aux_max;
}

// The same run for the callers that only want the numbers. 'f' outlives calculate() in the overload
// above so an assertion can read the size-zone matrix that run built; a caller with no such assertion
// says so by not naming one.
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
	D3_GLSZM_feature f;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, s, f));
}

// bench_cube4x4x3_zcross -- a 4x4x3 volume whose zones cross the slices, small enough that every one
// of them can be read off by eye. It is the fixture behind the family's connectivity check: the 26
// offsets gather_size_zones() walks, separated from the three neighbourhoods they could be confused
// with. The nine zones, by (z, y, x):
//
//   level 1  {(0,0,0), (1,0,0), (2,0,0)}   a vertical run, dz=1 with dy=dx=0, walked both ways
//   level 1  {(1,2,2), (1,3,3)}            in-slice diagonal, the 8-neighbour case
//   level 2  {(0,0,2), (1,0,3)}            a z-edge join: dz=1, dy=0, dx=1
//   level 2  {(2,3,0)}                     alone
//   level 3  {(0,2,0), (1,3,1)}            a z-corner join: dz=1, dy=1, dx=1 -- the only 26-only link
//   level 3  {(2,1,0), (2,1,1), (2,1,2)}   an in-slice run
//   level 4  {(2,0,2), (2,1,3)}            in-slice diagonal
//   level 4  {(0,3,3)} and {(2,3,3)}       same (y,x), two slices apart: NOT one zone
//
// The last pair is the control that keeps the check honest in the other direction -- an
// implementation that joined a whole column would merge them. Counting zones under the
// neighbourhoods this one could be mistaken for gives a different table every time: 26-connectivity
// 9 zones, 18-connectivity 10 (the z-corner link breaks), 6-connectivity 13, and a purely 2D
// 8-neighbour pass 13. The earlier version of this fixture had one populated slice between two empty
// ones, so every dz != 0 neighbour of every voxel was background: 26-, 18- and 2D 8-connectivity all
// produced the same nine zones there, and only 6-connectivity differed.
static const std::vector<PixIntens> glszm_3d_zcross_volume
{
	// z=0
	1, 0, 2, 0,
	0, 0, 0, 0,
	3, 0, 0, 0,
	0, 0, 0, 4,
	// z=1
	1, 0, 0, 2,
	0, 0, 0, 0,
	0, 0, 1, 0,
	0, 3, 0, 1,
	// z=2
	1, 0, 4, 0,
	3, 3, 3, 4,
	0, 0, 0, 0,
	2, 0, 0, 4
};

// bench_cube3_gapped_levels -- a 3x3x3 volume carrying grey levels 1, 3 and 5, with 2 and 4 absent.
// It is the fixture for the half of the config matrix IBSI=true selects, which nothing else in the
// tree runs: calculate() overwrites GLSZM_GREYDEPTH with 0 there, and at 0 the row a zone lands in
// becomes 'zone.first - 1' rather than the position of that level in I, while Ng becomes max(I)
// rather than I.size(). A fixture whose levels are contiguous from 1 cannot tell those apart. This
// one can, because the position of a level and the level itself are different numbers in it.
//
// The six zones, by (z, y, x):
//
//   level 1  {(0,0,0), (1,0,0)}                   vertical
//   level 1  {(2,2,2)}                            alone
//   level 3  {(0,0,2), (1,1,1), (2,0,2)}          two z-corner steps, dz=1 with dy and dx both +-1
//   level 5  {(0,2,0)}, {(0,2,2)}, {(2,2,0)}      pairwise non-adjacent: dx=2 in-slice, dz=2 across
static const std::vector<PixIntens> glszm_3d_gapped_volume
{
	// z=0
	1, 0, 3,
	0, 0, 0,
	5, 0, 5,
	// z=1
	1, 0, 0,
	0, 3, 0,
	0, 0, 0,
	// z=2
	0, 0, 3,
	0, 0, 0,
	5, 0, 1
};

// bench_cube2_constant -- a 2x2x2 volume of one non-background intensity, which is the smallest ROI
// that reaches calculate()'s aux_min == aux_max intercept. It is fully populated rather than blank:
// at no binning it has one grey level, one 26-connected zone of eight voxels, a single populated
// cell, and sixteen finite features over it. The intercept returns the soft-NaN sentinel for all
// sixteen instead, which is what test_3d_glszm_constant_roi_regression pins. The intensity is 7
// rather than 1 so that a zone landing on the wrong row is a visibly wrong level.
static const std::vector<PixIntens> glszm_3d_constant_volume (8, 7);

// Runs the family on a literal volume rather than on a phantom read from disk: an LR carrying that
// volume as its voxel cube, its extrema, and one Pixel3 per non-background voxel, which is the Np
// ZonePercentage divides by. It is the same calculate() the phantom assertions run -- the fixture is
// the only thing that changes -- so a matrix pinned here is pinned on the production path, not on a
// re-implementation of it.
static void run_3d_glszm_on_volume (
	std::vector<std::vector<double>>& fvals,
	const std::vector<PixIntens>& voxels,
	int w, int h, int d,
	const Fsettings& s,
	D3_GLSZM_feature& f)
{
	ASSERT_EQ (voxels.size(), size_t(w) * size_t(h) * size_t(d));

	// An LR starts at the empty running-extrema state (aux_min = +inf, aux_max = 0), which is what
	// the loop below accumulates into -- the same state the real prescan hands calculate().
	LR r;
	r.aux_image_cube = SimpleCube<PixIntens> (voxels, w, h, d);
	for (int z = 0; z < d; z++)
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				PixIntens v = r.aux_image_cube.zyx (z, y, x);
				if (!v)
					continue;   // background, outside the ROI
				r.raw_pixels_3D.push_back (Pixel3 (x, y, z, v));
				if (v < r.aux_min)
					r.aux_min = v;
				if (v > r.aux_max)
					r.aux_max = v;
			}

	ASSERT_NO_THROW(r.initialize_fvals());
	ASSERT_NO_THROW(f.calculate(r, s));
	f.save_value(r.fvals);
	fvals = r.fvals;
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
