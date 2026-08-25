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

// Only what nothing this file already includes supplies: <algorithm> for sort / find / max, and
// <unordered_set> for the unique-level list the matrix assertions build. <iomanip> for the dump
// helper's setprecision. <iostream>, <string>, <vector> and gtest arrive through the common header
// and are not repeated.
#include <algorithm>
#include <iomanip>
#include <unordered_set>

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

// A 4x4x3 volume with one populated slice between two empty ones, small enough that its zones can be
// read off by eye and every one of them is a 2D 8-connected component of the middle slice. It is the
// direct check that the 26 offsets gather_size_zones() walks agree with PyRadiomics' neighbourhood:
// the phantom above would hide a connectivity difference in a matrix nobody can check by hand.
//
// The volume is the fixture here, not a copy of one -- there is no file to read it from, and the
// generator runs PyRadiomics on this same literal at binWidth=1.
static const std::vector<PixIntens> glszm_3d_pyradiomics_small_volume
{
	// z=0
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
	// z=1
	1, 2, 3, 4,
	1, 3, 4, 4,
	3, 2, 2, 2,
	4, 1, 4, 1,
	// z=2
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0
};

static const int glszm_3d_pyradiomics_smallmatrix_ng = 4;
static const int glszm_3d_pyradiomics_smallmatrix_ns = 3;
static const int glszm_3d_pyradiomics_smallmatrix_nz = 9;

// Its size-zone matrix, as PyRadiomics reports it:
//
//              [size=1 size=2 size=3]
//    [level=1]      2      1      0
//    [level=2]      1      0      1
//    [level=3]      0      0      1
//    [level=4]      2      0      1
static const ref_vals_list<Glszm3dMatrixCell> glszm_3d_pyradiomics_smallmatrix_ref_vals
{
	{ 1, 1, 2 },
	{ 1, 2, 1 },
	{ 2, 1, 1 },
	{ 2, 3, 1 },
	{ 3, 3, 1 },
	{ 4, 1, 2 },
	{ 4, 3, 1 },
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

// Asserts the size-zone matrix a grey-binned volume produces against a pinned oracle table.
//
// 'binned' is taken by value because gather_size_zones() marks visited voxels in place, so the
// caller's copy would come back destroyed.
static void assert_3d_glszm_matrix_pyradiomics (
	SimpleCube<PixIntens> binned,
	const ref_vals_list<Glszm3dMatrixCell>& expected,
	int expect_ng, int expect_ns, int expect_nz)
{
	// sorted unique non-background intensities, which is the row order calculate() indexes into
	std::unordered_set<PixIntens> U (binned.begin(), binned.end());
	U.erase (0);
	std::vector<PixIntens> I (U.begin(), U.end());
	std::sort (I.begin(), I.end());
	ASSERT_EQ ((int)I.size(), expect_ng);

	std::vector<std::pair<PixIntens, int>> zones;
	D3_GLSZM_feature::gather_size_zones (zones, binned, 0/*zeroI at radiomics and no binning*/);
	ASSERT_EQ ((int)zones.size(), expect_nz);

	int ns = 0;
	for (const auto& z : zones)
		ns = (std::max) (ns, z.second);
	ASSERT_EQ (ns, expect_ns);

	SimpleMatrix<int> P;
	P.allocate (ns/*width*/, (int)I.size()/*height*/);
	P.fill (0);
	for (const auto& z : zones)
	{
		auto it = std::find (I.begin(), I.end(), z.first);
		ASSERT_TRUE (it != I.end());
		P.xy (z.second - 1, int(it - I.begin()))++;
	}

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

// The matrix under the sixteen features, on the voxels the featurisation actually read and binned
// by the same bin_intensities_3d() calculate() calls. It takes the cube and the intensity extrema
// back out of the same extract_3d_glszm() the sixteen oracle assertions run, so this assertion and
// those describe one run rather than two -- a hand-written copy of the phantom would keep passing
// after the loader started producing something else.
void test_3d_glszm_matrix_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_glszm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_glszm (fvals, cube, lo, hi, ipath, mpath, label, s));

	D3_GLSZM_feature f;
	SimpleCube<PixIntens> binned;
	binned.allocate (cube.width(), cube.height(), cube.depth());
	f.bin_intensities_3d (binned, cube, lo, hi, -20);

	ASSERT_NO_FATAL_FAILURE(assert_3d_glszm_matrix_pyradiomics (
		binned, glszm_3d_pyradiomics_matrix_ref_vals,
		glszm_3d_pyradiomics_matrix_ng, glszm_3d_pyradiomics_matrix_ns, glszm_3d_pyradiomics_matrix_nz));

	// the ROI's voxel count, which is what ZonePercentage divides the zone count by
	int occupied = 0;
	for (auto v : binned)
		if (v)
			occupied++;
	ASSERT_EQ (occupied, glszm_3d_pyradiomics_matrix_np);
}

// The connectivity check: eight zones that can be counted by eye, on a volume small enough to print.
void test_3d_glszm_smallmatrix_pyradiomics()
{
	SimpleCube<PixIntens> D (glszm_3d_pyradiomics_small_volume, 4/*width*/, 4/*height*/, 3/*depth*/);
	ASSERT_NO_FATAL_FAILURE(assert_3d_glszm_matrix_pyradiomics (
		D, glszm_3d_pyradiomics_smallmatrix_ref_vals,
		glszm_3d_pyradiomics_smallmatrix_ng, glszm_3d_pyradiomics_smallmatrix_ns,
		glszm_3d_pyradiomics_smallmatrix_nz));
}

// Regenerates every golden in glszm_3d_pyradiomics_ref_vals at full precision, in the shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_GLSZM_DUMP_PYRADIOMICS*
// It prints Nyxus' own values at the recipe the assertions use, which is what makes the residual
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
