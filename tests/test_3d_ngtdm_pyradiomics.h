#pragma once

// 3D NGTDM vs PyRadiomics 3.0.1 on the NGTDM compatibility phantom.
//
// Recipe ngtdm3d.pyradiomics_binwidth1: bench_compat_ngtdm_3d (label 57) with binWidth=1,
// distances=[1], no resampling, imageType=Original; on the Nyxus side GREYDEPTH=100, IBSI=false,
// NGTDM_GREYDEPTH=0 (no binning) and NGTDM_RADIUS=1. Both sides work on the grey levels 1..6.
//
// Goldens and their reproduction: tests/vetting/oracles/gen_ngtdm3d_pyradiomics.py, which also
// re-verifies every pin below. Measurements and the PyRadiomics-loader limitation this phantom runs
// into: tests/vetting/audit/ngtdm_3d_pyradiomics_vetting_report.md.

// Only what nothing this file already includes supplies: <algorithm> and <unordered_set> for the
// sorted-unique-level list the matrix assertion builds, <iomanip> for the dump helper's
// setprecision. <iostream>, <string>, <vector> and gtest arrive through the common header and are
// not repeated.
#include <algorithm>
#include <iomanip>
#include <unordered_set>

#include "test_3d_ngtdm_common.h"  // gtest, <iostream>, the phantom, the settings recipe, extract_3d_ngtdm, agrees_gt
#include "test_ref_vals.h"         // ref_vals_map, ref_vals_list

static const ref_vals_map<double> ngtdm_3d_pyradiomics_ref_vals
{
	{"3NGTDM_BUSYNESS", 4.553401556426767},         // original_ngtdm_Busyness
	{"3NGTDM_COARSENESS", 0.030118770647251797},    // original_ngtdm_Coarseness
	{"3NGTDM_COMPLEXITY", 32.13037220400344},       // original_ngtdm_Complexity
	{"3NGTDM_CONTRAST", 0.23138014315250832},       // original_ngtdm_Contrast
	{"3NGTDM_STRENGTH", 1.245800596888454}          // original_ngtdm_Strength
};

// One row of an NGTDM: the grey level, the number of voxels carrying it that have at least one
// neighbour, that count as a fraction of all such voxels, and the sum over them of the absolute
// difference between the level and its neighbourhood mean.
struct Ngtdm3dMatrixRow
{
	unsigned int level;
	int n;
	double p;
	double s;
};

// The NGTDM of the phantom itself, from PyRadiomics' P_ngtdm array -- the table it builds before any
// feature formula runs. All five features above are contractions of these eighteen numbers, so a
// scalar assertion alone cannot tell a correct matrix from two errors in it that cancel.
static const ref_vals_list<Ngtdm3dMatrixRow> ngtdm_3d_pyradiomics_matrix_ref_vals
{
	{ 1, 32, 0.6666666666666666, 47.05118411000764 },
	{ 2, 6, 0.125, 0.8266145619086798 },
	{ 3, 2, 0.041666666666666664, 2.358288770053476 },
	{ 4, 4, 0.08333333333333333, 9.10880296174414 },
	{ 5, 1, 0.020833333333333332, 3.1923076923076925 },
	{ 6, 3, 0.0625, 12.916289592760181 }
};

// The 4x4 image PyRadiomics' NGTDM documentation works through by hand, driven here as a
// single-slice volume. Its published s_i carry three significant figures; these are the
// full-precision values of a PyRadiomics run on the same image, which the generator also reproduces
// from the IBSI definition in exact rational arithmetic. Grey level 4 is absent from the image, and
// both tools drop empty levels, so the table has four rows and not five.
static const ref_vals_list<Ngtdm3dMatrixRow> ngtdm_3d_pyradiomics_docmatrix_ref_vals
{
	{ 1, 6, 0.375, 13.35 },
	{ 2, 2, 0.125, 2.0 },
	{ 3, 4, 0.25, 3.033333333333333 },
	{ 5, 4, 0.25, 10.075 }
};

// The 4x4 image the table above is the NGTDM of.
static const std::vector<PixIntens> ngtdm_3d_pyradiomics_doc_image
{
	1, 2, 5, 2,
	3, 5, 1, 3,
	1, 3, 5, 5,
	3, 1, 1, 1
};

// rel=1e-9: agrees_gt divides the golden by this, so a larger argument is a tighter band. Nyxus and
// PyRadiomics discretise this phantom to the same six levels and build the same neighbourhood, so
// there is no convention residual to accommodate; the measured worst case over the five features
// and the eighteen matrix entries is at the last bit.
static const double ngtdm_3d_pyradiomics_frac_tolerance = 1e9;

void assert_3d_ngtdm_feature_pyradiomics (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	// a name with no golden is a failure, not a comparison against whatever a lookup would invent
	auto iter = ngtdm_3d_pyradiomics_ref_vals.find(fname);
	ASSERT_TRUE(iter != ngtdm_3d_pyradiomics_ref_vals.end());

	int fcode = -1;
	ASSERT_NO_FATAL_FAILURE(resolve_3d_ngtdm_fcode (fcode, expecting_fcode, fname));

	auto [ipath, mpath, label] = get_3d_compat_ngtdm_phantom();
	Fsettings s = make_ngtdm3d_settings (100/*greydepth*/, 0/*no ngtdm binning*/, 1/*radius*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, label, s));

	ASSERT_TRUE (agrees_gt (fvals[fcode][0], iter->second,
	                        ngtdm_3d_pyradiomics_frac_tolerance)) << fname;
}

// Asserts one NGTDM the family's matrix builder produces against a pinned oracle table.
static void assert_3d_ngtdm_matrix_pyradiomics (
	const std::vector<PixIntens>& raw,
	int width, int height, int depth,
	const ref_vals_list<Ngtdm3dMatrixRow>& expected)
{
	SimpleCube<PixIntens> D (raw, width, height, depth);

	// sorted unique intensities, which is the row order calc_NGTDM indexes into
	std::unordered_set<PixIntens> U (raw.begin(), raw.end());
	std::vector<PixIntens> I (U.begin(), U.end());
	std::sort (I.begin(), I.end());
	ASSERT_EQ (I.size(), expected.size());

	std::vector<std::pair<PixIntens, double>> Zones;
	D3_NGTDM_feature::gather_zones (Zones, D, 1/*radius*/, 0/*zeroI: no level is background here*/);

	std::vector<int> N;
	std::vector<double> P, S;
	double Nvp = D3_NGTDM_feature::calc_NGTDM (N, P, S, Zones, I);

	ASSERT_EQ (N.size(), expected.size());
	ASSERT_EQ (P.size(), expected.size());
	ASSERT_EQ (S.size(), expected.size());

	// every voxel of these fixtures has a neighbour, so the valid-voxel count is the voxel count
	ASSERT_EQ ((size_t)Nvp, raw.size());

	for (size_t k = 0; k < expected.size(); k++)
	{
		const Ngtdm3dMatrixRow& row = expected[k];
		SCOPED_TRACE ("grey level " + std::to_string (row.level));
		ASSERT_EQ (I[k], row.level);
		ASSERT_EQ (N[k], row.n);
		ASSERT_TRUE (agrees_gt (P[k], row.p, ngtdm_3d_pyradiomics_frac_tolerance));
		ASSERT_TRUE (agrees_gt (S[k], row.s, ngtdm_3d_pyradiomics_frac_tolerance));
	}
}

// The matrix under the five features, on the voxels the featurisation actually read. It takes the
// cube back out of the same extract_3d_ngtdm() the five oracle assertions run, so this assertion and
// those describe one run rather than two -- a hand-written copy of the phantom would keep passing
// after the loader started producing something else.
//
// The one step reproduced here is calculate()'s zero-min correction: when the minimum level is 0 it
// shifts every level by one, which at NGTDM_GREYDEPTH=0 (no binning) is the whole of its preamble.
// The minimum is asserted rather than assumed, since the shift is conditional on it.
void test_3d_ngtdm_matrix_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_ngtdm_phantom();
	Fsettings s = make_ngtdm3d_settings (100/*greydepth*/, 0/*no ngtdm binning*/, 1/*radius*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, label, s));

	ASSERT_EQ (cube.size(), size_t(4 * 4 * 3));
	ASSERT_EQ (*std::min_element (cube.begin(), cube.end()), PixIntens(0));

	std::vector<PixIntens> levels (cube.begin(), cube.end());
	for (auto& lev : levels)
		lev += 1;

	assert_3d_ngtdm_matrix_pyradiomics (levels, cube.width(), cube.height(), cube.depth(),
	                                    ngtdm_3d_pyradiomics_matrix_ref_vals);
}

// The doc example keeps a literal, and for it that is the right shape: the 4x4 image IS the
// fixture -- it is published, there is no file to read it from, and the generator runs
// PyRadiomics on the same literal.
void test_3d_ngtdm_docmatrix_pyradiomics()
{
	assert_3d_ngtdm_matrix_pyradiomics (ngtdm_3d_pyradiomics_doc_image, 4, 4, 1,
	                                    ngtdm_3d_pyradiomics_docmatrix_ref_vals);
}

// Regenerates every golden in ngtdm_3d_pyradiomics_ref_vals at full precision, in the shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_NGTDM_DUMP_PYRADIOMICS*
// It prints Nyxus' own values at the recipe the assertions use, which is what makes the residual
// against the pinned PyRadiomics goldens readable without a debugger.
void test_3d_ngtdm_dump_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_ngtdm_phantom();
	Fsettings s = make_ngtdm3d_settings (100/*greydepth*/, 0/*no ngtdm binning*/, 1/*radius*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, label, s));

	Environment e;
	std::cout << "[3DNGTDM-PYRAD]\n";
	for (const auto& nv : ngtdm_3d_pyradiomics_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		std::cout << "[3DNGTDM-PYRAD]\t{\"" << nv.first << "\", "
		          << std::setprecision(17) << fvals[fcode][0] << "},\tpinned "
		          << std::setprecision(17) << nv.second << "\n";
	}
}

void test_3d_ngtdm_busyness_pyradiomics() {
	assert_3d_ngtdm_feature_pyradiomics (Nyxus::Feature3D::NGTDM_BUSYNESS, "3NGTDM_BUSYNESS");
}

void test_3d_ngtdm_coarseness_pyradiomics() {
	assert_3d_ngtdm_feature_pyradiomics (Nyxus::Feature3D::NGTDM_COARSENESS, "3NGTDM_COARSENESS");
}

void test_3d_ngtdm_complexity_pyradiomics() {
	assert_3d_ngtdm_feature_pyradiomics (Nyxus::Feature3D::NGTDM_COMPLEXITY, "3NGTDM_COMPLEXITY");
}

void test_3d_ngtdm_contrast_pyradiomics() {
	assert_3d_ngtdm_feature_pyradiomics (Nyxus::Feature3D::NGTDM_CONTRAST, "3NGTDM_CONTRAST");
}

void test_3d_ngtdm_strength_pyradiomics() {
	assert_3d_ngtdm_feature_pyradiomics (Nyxus::Feature3D::NGTDM_STRENGTH, "3NGTDM_STRENGTH");
}
