#pragma once

// 3D GLSZM vs PyRadiomics 3.0.1 on the compat phantom.
//
// Recipe glszm3d.pyradiomics_bincount20: bench_compat_liver_3d (label 1) with binCount=20,
// no resampling, weightingNorm=None, imageType=Original; on the Nyxus side GREYDEPTH=100,
// IBSI=false and GLSZM_GREYDEPTH=-20, whose negative sign selects the same binCount binning.
//
// Goldens and their reproduction: tests/vetting/oracles/gen_glszm3d_pyradiomics.py, which also
// re-verifies every pin below and recomputes all sixteen features from the pinned matrix.
// Measurements: tests/vetting/audit/glszm_3d_pyradiomics_vetting_report.md.

// Only what nothing this file already includes supplies: <algorithm> for the std::find that turns a
// pinned grey level into the row calculate() put it in, and <iomanip> for the dump helper's
// setprecision. <iostream>, <string>, <vector> and gtest arrive through the common header and are
// not repeated.
#include <algorithm>
#include <iomanip>

#include "test_3d_glszm_common.h"  // gtest, <iostream>, the phantoms, the settings recipe, extract_3d_glszm, agrees_gt
#include "test_ref_vals.h"         // ref_vals_map, ref_vals_list

static const ref_vals_map<double> glszm_3d_pyradiomics_ref_vals
{
	{"3GLSZM_GLN", 61.77441860465116},           // original_glszm_GrayLevelNonUniformity
	{"3GLSZM_GLNN", 0.07183071930773391},        // original_glszm_GrayLevelNonUniformityNormalized
	{"3GLSZM_GLV", 14.965087885343427},          // original_glszm_GrayLevelVariance
	{"3GLSZM_HGLZE", 134.6639534883721},         // original_glszm_HighGrayLevelZoneEmphasis
	{"3GLSZM_LAE", 723.7093023255813},           // original_glszm_LargeAreaEmphasis
	{"3GLSZM_LAHGLE", 87509.9523255814},         // original_glszm_LargeAreaHighGrayLevelEmphasis
	{"3GLSZM_LALGLE", 6.280653691016313},        // original_glszm_LargeAreaLowGrayLevelEmphasis
	{"3GLSZM_LGLZE", 0.016482439101794737},      // original_glszm_LowGrayLevelZoneEmphasis
	{"3GLSZM_SAE", 0.5306840085503507},          // original_glszm_SmallAreaEmphasis
	{"3GLSZM_SAHGLE", 72.65640040229414},        // original_glszm_SmallAreaHighGrayLevelEmphasis
	{"3GLSZM_SALGLE", 0.008788101239865679},     // original_glszm_SmallAreaLowGrayLevelEmphasis
	{"3GLSZM_SZN", 231.4279069767442},           // original_glszm_SizeZoneNonUniformity
	{"3GLSZM_SZNN", 0.2691022174148188},         // original_glszm_SizeZoneNonUniformityNormalized
	{"3GLSZM_ZE", 6.426417026786065},            // original_glszm_ZoneEntropy
	{"3GLSZM_ZP", 0.17916666666666667},          // original_glszm_ZonePercentage
	{"3GLSZM_ZV", 692.5573282855598}             // original_glszm_ZoneVariance
};

// agrees_gt divides the golden by this, so a larger argument is a tighter band. Nyxus and
// PyRadiomics bin this phantom into the same twenty levels and find the same connected components,
// so there is no convention residual to accommodate: over the fifteen features that take no
// logarithm the measured worst residual is 1.2e-15, which is float summation order and nothing else.
static const double glszm_3d_pyradiomics_frac_tolerance = 1e9;

// 3GLSZM_ZE is the family's only sum over logarithms, and Nyxus takes it through fast_log10() -- a
// float-precision log10 approximation divided by a ten-digit LOG10_2 -- where PyRadiomics uses
// numpy.log2. That is a deliberate fast path and its error is a convention of this codebase, so it
// belongs in the band rather than in a defect report; rel=1e-3 is the measured residual rounded up.
static const double glszm_3d_pyradiomics_ze_frac_tolerance = 1e3;

// The same fast path, measured again on the gapped-level fixture, where it costs more: 1.1e-3 there
// against 1.9e-4 on the phantom. Zone entropy is a sum of -p*log2(p) over the matrix, so on six zones
// each term carries the approximation's error at full weight, while on the phantom's 860 zones the
// errors are spread over 186 terms and partly cancel. Band rel=2e-3, the measured residual rounded up.
static const double glszm_3d_pyradiomics_gapped_ze_frac_tolerance = 5e2;

// One non-empty cell of a size-zone matrix: the grey level, the size of the zones counted, and how
// many zones of that level and size the ROI holds.
struct Glszm3dMatrixCell
{
	unsigned int level;
	int size;
	int count;
};

// Dimensions of the phantom's size-zone matrix. 'ns' is the largest zone the ROI holds and is the
// width Nyxus allocates; PyRadiomics reports a narrower matrix because it deletes the columns for
// zone sizes nothing occupies, and carries the surviving sizes in its jvector. The cells below are
// keyed by the size itself, so the two representations line up entry for entry.
static const int glszm_3d_pyradiomics_matrix_ng = 20;
static const int glszm_3d_pyradiomics_matrix_ns = 634;
static const int glszm_3d_pyradiomics_matrix_nz = 860;
static const int glszm_3d_pyradiomics_matrix_np = 4800;

// The size-zone matrix of the phantom itself, from PyRadiomics' P_glszm array -- the table it builds
// before any feature formula runs. All sixteen features above are contractions of these numbers, so
// a scalar assertion alone cannot tell a correct matrix from two errors in it that cancel. Every
// cell not listed is empty, which the assertion checks by counting.
static const ref_vals_list<Glszm3dMatrixCell> glszm_3d_pyradiomics_matrix_ref_vals
{
	{ 1, 5, 1 },
	{ 2, 1, 6 },
	{ 3, 1, 4 },
	{ 3, 2, 3 },
	{ 3, 3, 2 },
	{ 3, 5, 1 },
	{ 4, 1, 10 },
	{ 4, 2, 7 },
	{ 4, 3, 3 },
	{ 4, 4, 2 },
	{ 4, 5, 1 },
	{ 4, 8, 1 },
	{ 5, 1, 14 },
	{ 5, 2, 6 },
	{ 5, 3, 6 },
	{ 5, 4, 1 },
	{ 5, 6, 2 },
	{ 5, 7, 1 },
	{ 5, 13, 1 },
	{ 6, 1, 15 },
	{ 6, 2, 11 },
	{ 6, 3, 7 },
	{ 6, 4, 3 },
	{ 6, 5, 1 },
	{ 6, 6, 1 },
	{ 6, 8, 1 },
	{ 6, 19, 1 },
	{ 6, 28, 1 },
	{ 7, 1, 30 },
	{ 7, 2, 10 },
	{ 7, 3, 8 },
	{ 7, 4, 4 },
	{ 7, 5, 3 },
	{ 7, 6, 1 },
	{ 7, 7, 1 },
	{ 7, 8, 2 },
	{ 7, 9, 1 },
	{ 7, 10, 1 },
	{ 7, 34, 1 },
	{ 7, 36, 1 },
	{ 8, 1, 35 },
	{ 8, 2, 21 },
	{ 8, 3, 10 },
	{ 8, 4, 3 },
	{ 8, 5, 1 },
	{ 8, 7, 1 },
	{ 8, 8, 4 },
	{ 8, 9, 2 },
	{ 8, 10, 1 },
	{ 8, 13, 1 },
	{ 8, 16, 1 },
	{ 8, 25, 1 },
	{ 8, 26, 1 },
	{ 8, 59, 1 },
	{ 9, 1, 34 },
	{ 9, 2, 16 },
	{ 9, 3, 5 },
	{ 9, 4, 1 },
	{ 9, 5, 5 },
	{ 9, 6, 1 },
	{ 9, 7, 2 },
	{ 9, 8, 2 },
	{ 9, 9, 2 },
	{ 9, 12, 1 },
	{ 9, 15, 2 },
	{ 9, 20, 1 },
	{ 9, 22, 1 },
	{ 9, 28, 1 },
	{ 9, 31, 1 },
	{ 9, 80, 1 },
	{ 9, 131, 1 },
	{ 10, 1, 26 },
	{ 10, 2, 9 },
	{ 10, 3, 7 },
	{ 10, 4, 2 },
	{ 10, 5, 3 },
	{ 10, 7, 2 },
	{ 10, 9, 1 },
	{ 10, 11, 2 },
	{ 10, 12, 1 },
	{ 10, 14, 1 },
	{ 10, 29, 1 },
	{ 10, 34, 1 },
	{ 10, 65, 1 },
	{ 10, 137, 1 },
	{ 10, 249, 1 },
	{ 11, 1, 31 },
	{ 11, 2, 10 },
	{ 11, 3, 5 },
	{ 11, 4, 5 },
	{ 11, 5, 1 },
	{ 11, 6, 2 },
	{ 11, 7, 1 },
	{ 11, 9, 1 },
	{ 11, 11, 2 },
	{ 11, 12, 1 },
	{ 11, 634, 1 },
	{ 12, 1, 38 },
	{ 12, 2, 16 },
	{ 12, 3, 5 },
	{ 12, 4, 3 },
	{ 12, 5, 4 },
	{ 12, 6, 2 },
	{ 12, 7, 4 },
	{ 12, 9, 2 },
	{ 12, 11, 1 },
	{ 12, 13, 1 },
	{ 12, 16, 1 },
	{ 12, 22, 1 },
	{ 12, 61, 1 },
	{ 12, 123, 1 },
	{ 12, 199, 1 },
	{ 13, 1, 40 },
	{ 13, 2, 12 },
	{ 13, 3, 6 },
	{ 13, 4, 5 },
	{ 13, 5, 3 },
	{ 13, 7, 3 },
	{ 13, 9, 1 },
	{ 13, 10, 1 },
	{ 13, 12, 1 },
	{ 13, 15, 1 },
	{ 13, 23, 1 },
	{ 13, 51, 1 },
	{ 13, 88, 1 },
	{ 13, 96, 1 },
	{ 14, 1, 40 },
	{ 14, 2, 15 },
	{ 14, 3, 11 },
	{ 14, 4, 1 },
	{ 14, 5, 1 },
	{ 14, 6, 1 },
	{ 14, 7, 1 },
	{ 14, 9, 4 },
	{ 14, 10, 2 },
	{ 14, 12, 2 },
	{ 14, 13, 1 },
	{ 14, 16, 1 },
	{ 14, 18, 1 },
	{ 14, 24, 1 },
	{ 14, 27, 1 },
	{ 14, 44, 1 },
	{ 15, 1, 25 },
	{ 15, 2, 11 },
	{ 15, 3, 6 },
	{ 15, 4, 4 },
	{ 15, 5, 1 },
	{ 15, 7, 2 },
	{ 15, 8, 2 },
	{ 15, 9, 1 },
	{ 15, 10, 1 },
	{ 15, 11, 2 },
	{ 15, 13, 1 },
	{ 15, 19, 1 },
	{ 15, 32, 1 },
	{ 16, 1, 16 },
	{ 16, 2, 7 },
	{ 16, 3, 6 },
	{ 16, 4, 2 },
	{ 16, 5, 1 },
	{ 16, 6, 3 },
	{ 16, 7, 3 },
	{ 16, 10, 1 },
	{ 16, 44, 1 },
	{ 17, 1, 17 },
	{ 17, 2, 4 },
	{ 17, 3, 1 },
	{ 17, 4, 4 },
	{ 17, 5, 2 },
	{ 17, 6, 1 },
	{ 17, 25, 1 },
	{ 18, 1, 8 },
	{ 18, 2, 3 },
	{ 18, 3, 2 },
	{ 18, 4, 1 },
	{ 18, 5, 1 },
	{ 18, 6, 1 },
	{ 18, 10, 1 },
	{ 18, 21, 1 },
	{ 19, 1, 8 },
	{ 19, 2, 1 },
	{ 19, 4, 1 },
	{ 19, 5, 1 },
	{ 19, 6, 1 },
	{ 20, 1, 3 },
	{ 20, 2, 2 },
};

// Dimensions and cells of the connectivity fixture's matrix. Its nine zones are listed with the
// fixture in the common header; what is pinned here is what PyRadiomics makes of them.
static const int glszm_3d_pyradiomics_smallmatrix_ng = 4;
static const int glszm_3d_pyradiomics_smallmatrix_ns = 3;
static const int glszm_3d_pyradiomics_smallmatrix_nz = 9;
static const int glszm_3d_pyradiomics_smallmatrix_np = 17;

// The size-zone matrix of glszm_3d_zcross_volume, as PyRadiomics reports it:
//
//              [size=1 size=2 size=3]
//    [level=1]      0      1      1
//    [level=2]      1      1      0
//    [level=3]      0      1      1
//    [level=4]      2      1      0
static const ref_vals_list<Glszm3dMatrixCell> glszm_3d_pyradiomics_smallmatrix_ref_vals
{
	{ 1, 2, 1 },
	{ 1, 3, 1 },
	{ 2, 1, 1 },
	{ 2, 2, 1 },
	{ 3, 2, 1 },
	{ 3, 3, 1 },
	{ 4, 1, 2 },
	{ 4, 2, 1 },
};

// PyRadiomics 3.0.1 on glszm_3d_gapped_volume at binWidth=1, which leaves levels 1, 3 and 5 where
// they are. Nyxus reaches the same three levels through IBSI=true, where calculate() forces the
// family's binning to 0 and reads the volume's own values. Reproduced by
// gen_glszm3d_pyradiomics.py under recipe glszm3d.pyradiomics_ibsi_gapped.
static const ref_vals_map<double> glszm_3d_pyradiomics_gapped_ref_vals
{
	{"3GLSZM_GLN", 2.3333333333333335},          // original_glszm_GrayLevelNonUniformity
	{"3GLSZM_GLNN", 0.3888888888888889},         // original_glszm_GrayLevelNonUniformityNormalized
	{"3GLSZM_GLV", 3.222222222222222},           // original_glszm_GrayLevelVariance
	{"3GLSZM_HGLZE", 14.333333333333334},        // original_glszm_HighGrayLevelZoneEmphasis
	{"3GLSZM_LAE", 2.8333333333333335},          // original_glszm_LargeAreaEmphasis
	{"3GLSZM_LAHGLE", 26.833333333333332},       // original_glszm_LargeAreaHighGrayLevelEmphasis
	{"3GLSZM_LALGLE", 1.02},                     // original_glszm_LargeAreaLowGrayLevelEmphasis
	{"3GLSZM_LGLZE", 0.3718518518518519},        // original_glszm_LowGrayLevelZoneEmphasis
	{"3GLSZM_SAE", 0.7268518518518517},          // original_glszm_SmallAreaEmphasis
	{"3GLSZM_SAHGLE", 12.875},                   // original_glszm_SmallAreaHighGrayLevelEmphasis
	{"3GLSZM_SALGLE", 0.2303909465020576},       // original_glszm_SmallAreaLowGrayLevelEmphasis
	{"3GLSZM_SZN", 3.0},                         // original_glszm_SizeZoneNonUniformity
	{"3GLSZM_SZNN", 0.5},                        // original_glszm_SizeZoneNonUniformityNormalized
	{"3GLSZM_ZE", 1.7924812503605767},           // original_glszm_ZoneEntropy
	{"3GLSZM_ZP", 0.6666666666666666},           // original_glszm_ZonePercentage
	{"3GLSZM_ZV", 0.5833333333333333}            // original_glszm_ZoneVariance
};

// Ng is 5, not 3: both sides count the levels up to the largest one, so rows 2 and 4 exist and are
// empty. That is the whole point of the fixture -- an implementation that packed the three occupied
// levels into rows 0..2 would produce the same sixteen scalars only if it also renumbered the levels,
// and it does not.
static const int glszm_3d_pyradiomics_gappedmatrix_ng = 5;
static const int glszm_3d_pyradiomics_gappedmatrix_ns = 3;
static const int glszm_3d_pyradiomics_gappedmatrix_nz = 6;
static const int glszm_3d_pyradiomics_gappedmatrix_np = 9;

// The size-zone matrix of glszm_3d_gapped_volume, as PyRadiomics reports it:
//
//              [size=1 size=2 size=3]
//    [level=1]      1      1      0
//    [level=2]      0      0      0
//    [level=3]      0      0      1
//    [level=4]      0      0      0
//    [level=5]      3      0      0
static const ref_vals_list<Glszm3dMatrixCell> glszm_3d_pyradiomics_gappedmatrix_ref_vals
{
	{ 1, 1, 1 },
	{ 1, 2, 1 },
	{ 3, 3, 1 },
	{ 5, 1, 3 },
};

void assert_3d_glszm_feature_pyradiomics (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	// a name with no golden is a failure, not a comparison against whatever a lookup would invent
	auto iter = glszm_3d_pyradiomics_ref_vals.find(fname);
	ASSERT_TRUE(iter != glszm_3d_pyradiomics_ref_vals.end());

	int fcode = -1;
	ASSERT_NO_FATAL_FAILURE(resolve_3d_glszm_fcode (fcode, expecting_fcode, fname));

	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_glszm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, s));

	double band = fname == "3GLSZM_ZE" ? glszm_3d_pyradiomics_ze_frac_tolerance
	                                   : glszm_3d_pyradiomics_frac_tolerance;
	ASSERT_TRUE (agrees_gt (fvals[fcode][0], iter->second, band)) << fname;
}

// Asserts the size-zone matrix a run built against a pinned oracle table.
//
// It reads the P the feature object holds after calculate() returned, not a table rebuilt beside it:
// the sixteen scalars are contractions of exactly this matrix, so an error in the row mapping, in
// Ng/Ns, in the allocation or in the fill loop has to be visible here, or the pin is not pinning the
// production table. The row a grey level sits in is looked up in the same I calculate() indexed by.
static void assert_3d_glszm_matrix_pyradiomics (
	const D3_GLSZM_feature& f,
	const ref_vals_list<Glszm3dMatrixCell>& expected,
	int expect_ng, int expect_ns, int expect_nz, int expect_np)
{
	const SimpleMatrix<int>& P = f.get_P();
	const std::vector<PixIntens>& I = f.get_I();

	ASSERT_EQ (f.get_Ng(), expect_ng);
	ASSERT_EQ (f.get_Ns(), expect_ns);
	ASSERT_EQ (f.get_Nz(), expect_nz);
	ASSERT_EQ (f.get_Np(), expect_np);

	// the table is the one those dimensions describe, so a mis-sized allocation is not read as a
	// matrix whose missing cells are merely empty
	ASSERT_EQ (P.width(), expect_ns);
	ASSERT_EQ (P.height(), expect_ng);

	int pinned_zones = 0;
	for (const Glszm3dMatrixCell& c : expected)
	{
		SCOPED_TRACE ("grey level " + std::to_string(c.level) + ", zone size " + std::to_string(c.size));
		auto it = std::find (I.begin(), I.end(), (PixIntens)c.level);
		ASSERT_TRUE (it != I.end());
		ASSERT_EQ (P.yx (int(it - I.begin()), c.size - 1), c.count);
		pinned_zones += c.count;
	}

	// nothing outside the pinned cells: the table accounts for every zone, and for every cell that
	// holds one. Without this a matrix with an extra populated cell would still pass every line above.
	int nonempty = 0;
	for (auto v : P)
		if (v)
			nonempty++;
	ASSERT_EQ (nonempty, (int)expected.size());
	ASSERT_EQ (pinned_zones, expect_nz);
}

// The matrix under the sixteen features, read off the feature object the sixteen oracle assertions
// run through: this assertion and those describe one run of one phantom rather than two, and a
// hand-written copy of the phantom would keep passing after the loader started producing something
// else.
void test_3d_glszm_matrix_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_glszm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	D3_GLSZM_feature f;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, s, f));

	ASSERT_NO_FATAL_FAILURE(assert_3d_glszm_matrix_pyradiomics (
		f, glszm_3d_pyradiomics_matrix_ref_vals,
		glszm_3d_pyradiomics_matrix_ng, glszm_3d_pyradiomics_matrix_ns,
		glszm_3d_pyradiomics_matrix_nz, glszm_3d_pyradiomics_matrix_np));
}

// The connectivity check: nine zones that can be counted by eye, on a volume small enough to print.
// It runs the same calculate() the phantom assertion does, at the family's no-binning setting, so the
// matrix is indexed by the literal's own levels 1..4 -- which is what PyRadiomics reads at binWidth=1.
void test_3d_glszm_smallmatrix_pyradiomics()
{
	Fsettings s = make_glszm3d_settings (64/*greydepth, inert here*/, 0/*no binning: the raw levels*/);
	std::vector<std::vector<double>> fvals;
	D3_GLSZM_feature f;
	ASSERT_NO_FATAL_FAILURE(run_3d_glszm_on_volume (
		fvals, glszm_3d_zcross_volume, 4/*width*/, 4/*height*/, 3/*depth*/, s, f));

	ASSERT_NO_FATAL_FAILURE(assert_3d_glszm_matrix_pyradiomics (
		f, glszm_3d_pyradiomics_smallmatrix_ref_vals,
		glszm_3d_pyradiomics_smallmatrix_ng, glszm_3d_pyradiomics_smallmatrix_ns,
		glszm_3d_pyradiomics_smallmatrix_nz, glszm_3d_pyradiomics_smallmatrix_np));
}

// The IBSI half of the config matrix, measured rather than assumed: the sixteen features and the
// size-zone matrix under them, at IBSI=true on the gapped-level volume.
//
// GLSZM_GREYDEPTH is passed as 64 and has to be ignored -- calculate() overwrites it with 0 whenever
// IBSI is on -- so a run that honoured it would bin levels 1..5 into 64 MATLAB bins and miss every
// pin below. That makes this assertion the measurement of the overwrite as well as of the branch.
void test_3d_glszm_ibsi_gapped_pyradiomics()
{
	Fsettings s = make_glszm3d_settings (64/*greydepth*/, 64/*overwritten with 0 by IBSI*/, true/*ibsi*/);
	std::vector<std::vector<double>> fvals;
	D3_GLSZM_feature f;
	ASSERT_NO_FATAL_FAILURE(run_3d_glszm_on_volume (
		fvals, glszm_3d_gapped_volume, 3/*width*/, 3/*height*/, 3/*depth*/, s, f));

	Environment e;
	for (const auto& nv : glszm_3d_pyradiomics_gapped_ref_vals)
	{
		SCOPED_TRACE (nv.first);
		int fcode = -1;
		ASSERT_TRUE (e.theFeatureSet.find_3D_FeatureByString (nv.first, fcode));
		double band = nv.first == "3GLSZM_ZE" ? glszm_3d_pyradiomics_gapped_ze_frac_tolerance
		                                      : glszm_3d_pyradiomics_frac_tolerance;
		ASSERT_TRUE (agrees_gt (fvals[fcode][0], nv.second, band)) << nv.first;
	}

	ASSERT_NO_FATAL_FAILURE(assert_3d_glszm_matrix_pyradiomics (
		f, glszm_3d_pyradiomics_gappedmatrix_ref_vals,
		glszm_3d_pyradiomics_gappedmatrix_ng, glszm_3d_pyradiomics_gappedmatrix_ns,
		glszm_3d_pyradiomics_gappedmatrix_nz, glszm_3d_pyradiomics_gappedmatrix_np));
}

// Regenerates every golden of both oracle recipes at full precision, in the shape the tables want.
// Run it with
//     runAllTests --gtest_filter=*3D_GLSZM_DUMP_PYRADIOMICS*
// It prints Nyxus' own values at the recipes the assertions use, which is what makes the residual
// against the pinned PyRadiomics goldens readable without a debugger.
void test_3d_glszm_dump_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_glszm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, s));

	Environment e;
	std::cout << "[3DGLSZM-PYRAD] cube " << cube.width() << "x" << cube.height() << "x" << cube.depth()
	          << ", intensity range [" << lo << ", " << hi << "]\n";
	for (const auto& nv : glszm_3d_pyradiomics_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DGLSZM-PYRAD]\t{\"" << nv.first << "\", "
		          << std::setprecision(17) << fvals[fcode][0] << "},\tpinned "
		          << std::setprecision(17) << nv.second << "\n";
	}

	Fsettings s_gap = make_glszm3d_settings (64/*greydepth*/, 64/*overwritten with 0*/, true/*ibsi*/);
	std::vector<std::vector<double>> gap_vals;
	D3_GLSZM_feature f_gap;
	ASSERT_NO_FATAL_FAILURE(run_3d_glszm_on_volume (
		gap_vals, glszm_3d_gapped_volume, 3/*width*/, 3/*height*/, 3/*depth*/, s_gap, f_gap));
	std::cout << "[3DGLSZM-GAPPED] Ng " << f_gap.get_Ng() << ", Ns " << f_gap.get_Ns()
	          << ", Nz " << f_gap.get_Nz() << ", Np " << f_gap.get_Np() << "\n";
	for (const auto& nv : glszm_3d_pyradiomics_gapped_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DGLSZM-GAPPED]\t{\"" << nv.first << "\", "
		          << std::setprecision(17) << gap_vals[fcode][0] << "},\tpinned "
		          << std::setprecision(17) << nv.second << "\n";
	}
}

void test_3d_glszm_sae_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_SAE, "3GLSZM_SAE");
}

void test_3d_glszm_lae_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_LAE, "3GLSZM_LAE");
}

void test_3d_glszm_lglze_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_LGLZE, "3GLSZM_LGLZE");
}

void test_3d_glszm_hglze_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_HGLZE, "3GLSZM_HGLZE");
}

void test_3d_glszm_salgle_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_SALGLE, "3GLSZM_SALGLE");
}

void test_3d_glszm_sahgle_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_SAHGLE, "3GLSZM_SAHGLE");
}

void test_3d_glszm_lalgle_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_LALGLE, "3GLSZM_LALGLE");
}

void test_3d_glszm_lahgle_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_LAHGLE, "3GLSZM_LAHGLE");
}

void test_3d_glszm_gln_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_GLN, "3GLSZM_GLN");
}

void test_3d_glszm_glnn_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_GLNN, "3GLSZM_GLNN");
}

void test_3d_glszm_szn_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_SZN, "3GLSZM_SZN");
}

void test_3d_glszm_sznn_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_SZNN, "3GLSZM_SZNN");
}

void test_3d_glszm_zp_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_ZP, "3GLSZM_ZP");
}

void test_3d_glszm_glv_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_GLV, "3GLSZM_GLV");
}

void test_3d_glszm_zv_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_ZV, "3GLSZM_ZV");
}

void test_3d_glszm_ze_pyradiomics() {
	assert_3d_glszm_feature_pyradiomics (Nyxus::Feature3D::GLSZM_ZE, "3GLSZM_ZE");
}
