#pragma once

// 3D NGTDM vs MIRP on a ROI that mixes voxels with a ROI neighbour and a voxel without one.
//
// Recipe ngtdm3d.mirp_fbn_mixed: label 59 of the NGTDM ball phantom (bench_compat_ngtdm_3d_ball) --
// three voxels in a chain at levels 3, 0, 3, and a lone voxel at level 2 that no other voxel of the
// ROI reaches at radius 1. On the Nyxus side GREYDEPTH=100, IBSI=false, NGTDM_GREYDEPTH=0 (no
// binning), NGTDM_RADIUS=1, so the levels are the raw ones lifted by one: 1, 3, 4. MIRP: by_slice=false,
// distance 1, base_discretisation_method="fixed_bin_number" with 4 bins, which maps the ROI's levels
// 0..3 to the same 1..4; the generator asserts that identity.
//
// WHY MIRP: IBSI leaves a voxel with no neighbour out of the NGTDM -- it is in no row and counts
// towards neither n_i, Nvp nor Ngp -- and MIRP and Nyxus both follow it. PyRadiomics keeps such a voxel
// as a row with s_i = 0, so it parts from both on every feature of any ROI that holds one, and cannot
// be the oracle here.
//
// The lone voxel's level is carried by no other voxel, so its matrix row is empty: Ngp, the number of
// non-empty rows, is 2 while the ROI has 3 levels. Contrast divides by Ngp (Ngp - 1), which is how
// that difference reaches a feature value.
//
// Goldens and their reproduction: tests/vetting/oracles/gen_ngtdm3d_mirp.py, which re-verifies every
// pin below against MIRP and against an independent numpy NGTDM.
//
// Provenance (SPEC 6.4):
//   tool      = mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3, Python 3.11)
//   config    = by_slice=false, base_feature_families="ngtdm", fixed_bin_number n=4, distance 1,
//               native 1x1x1 spacing
//   fixture   = tests/data/nifti/compat_int/compat_int_ngtdm_3d_ball.nii +
//               compat_seg/compat_seg_ngtdm_3d_ball.nii, label 59
//   recipe    = ngtdm3d.mirp_fbn_mixed
//   generator = tests/vetting/oracles/gen_ngtdm3d_mirp.py

// Only what nothing this file already includes supplies: <iomanip> for setprecision. gtest, <string>
// and <vector> arrive through the common header.
#include <iomanip>

#include "test_3d_ngtdm_common.h"  // gtest, the phantom, the settings recipe, extract_3d_ngtdm, Ngtdm3dMatrixRow, agrees_gt
#include "test_ref_vals.h"         // ref_vals_map, ref_vals_list

static const ref_vals_map<double> ngtdm_3d_mirp_mixed_ref_vals
{
	{"3NGTDM_BUSYNESS", 1.0714285714285716},        // ngt_busyness_3d_fbn_n4
	{"3NGTDM_COARSENESS", 0.2},                     // ngt_coarseness_3d_fbn_n4
	{"3NGTDM_COMPLEXITY", 10.0},                    // ngt_complexity_3d_fbn_n4
	{"3NGTDM_CONTRAST", 6.0},                       // ngt_contrast_3d_fbn_n4
	{"3NGTDM_STRENGTH", 2.0}                        // ngt_strength_3d_fbn_n4
};

// Label 59's NGTDM in Nyxus' layout: one row per level the ROI carries, so the lone voxel's level 3
// is a row with n_i = 0. From the generator's numpy NGTDM; the generator recomputes the five MIRP
// values above from this table, which ties it to MIRP as well.
static const ref_vals_list<Ngtdm3dMatrixRow> ngtdm_3d_mirp_mixed_matrix_ref_vals
{
	{ 1, 1, 0.3333333333333333, 3.0 },
	{ 3, 0, 0.0, 0.0 },
	{ 4, 2, 0.6666666666666666, 6.0 }
};

// rel=1e-9: the two tools work on the same four levels with the same neighbourhood, and the measured
// residual is 0. agrees_gt divides the golden by this, so a larger argument is a tighter band.
static const double ngtdm_3d_mirp_frac_tolerance = 1e9;

static void assert_3d_ngtdm_feature_mixed_mirp (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	SCOPED_TRACE (std::string("MIRP_ORACLE_MIXED__") + fname);
	auto iter = ngtdm_3d_mirp_mixed_ref_vals.find (fname);
	ASSERT_TRUE (iter != ngtdm_3d_mirp_mixed_ref_vals.end()) << fname;

	int fcode = -1;
	ASSERT_NO_FATAL_FAILURE(resolve_3d_ngtdm_fcode (fcode, expecting_fcode, fname));

	auto [ipath, mpath, label, isolated_label] = get_3d_ngtdm_ball_phantom();
	Fsettings s = make_ngtdm3d_settings (100/*greydepth*/, 0/*no ngtdm binning*/, 1/*radius*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, get_3d_ngtdm_ball_phantom_mixed_label(), s));

	ASSERT_TRUE (agrees_gt (fvals[fcode][0], iter->second, ngtdm_3d_mirp_frac_tolerance))
		<< fname << " actual=" << std::setprecision(17) << fvals[fcode][0] << " mirp=" << iter->second;
}

void test_3d_ngtdm_busyness_mixed_mirp()
{
	assert_3d_ngtdm_feature_mixed_mirp (Nyxus::Feature3D::NGTDM_BUSYNESS, "3NGTDM_BUSYNESS");
}

void test_3d_ngtdm_coarseness_mixed_mirp()
{
	assert_3d_ngtdm_feature_mixed_mirp (Nyxus::Feature3D::NGTDM_COARSENESS, "3NGTDM_COARSENESS");
}

void test_3d_ngtdm_complexity_mixed_mirp()
{
	assert_3d_ngtdm_feature_mixed_mirp (Nyxus::Feature3D::NGTDM_COMPLEXITY, "3NGTDM_COMPLEXITY");
}

void test_3d_ngtdm_contrast_mixed_mirp()
{
	assert_3d_ngtdm_feature_mixed_mirp (Nyxus::Feature3D::NGTDM_CONTRAST, "3NGTDM_CONTRAST");
}

void test_3d_ngtdm_strength_mixed_mirp()
{
	assert_3d_ngtdm_feature_mixed_mirp (Nyxus::Feature3D::NGTDM_STRENGTH, "3NGTDM_STRENGTH");
}

// The matrix the five values above are contractions of, from the same run. The lone voxel's level
// keeps its row with n_i = 0, the n_i add up to the three voxels that have a neighbour, and Ngp is the
// two non-empty rows -- not the ROI's three levels, which is the count Contrast would otherwise divide by.
void test_3d_ngtdm_matrix_mixed_mirp()
{
	auto [ipath, mpath, label, isolated_label] = get_3d_ngtdm_ball_phantom();
	Fsettings s = make_ngtdm3d_settings (100/*greydepth*/, 0/*no ngtdm binning*/, 1/*radius*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	Ngtdm3dMatrix m;
	ASSERT_NO_FATAL_FAILURE(extract_3d_ngtdm (fvals, cube, ipath, mpath, get_3d_ngtdm_ball_phantom_mixed_label(), s, &m));

	const auto& expected = ngtdm_3d_mirp_mixed_matrix_ref_vals;
	ASSERT_EQ (m.I.size(), expected.size());
	ASSERT_EQ (m.N.size(), expected.size());
	ASSERT_EQ (m.P.size(), expected.size());
	ASSERT_EQ (m.S.size(), expected.size());
	ASSERT_EQ (m.Nvp, 3);
	ASSERT_EQ (m.Ngp, 2);

	int nonempty = 0;
	for (size_t k = 0; k < expected.size(); k++)
	{
		const Ngtdm3dMatrixRow& row = expected[k];
		SCOPED_TRACE ("grey level " + std::to_string (row.level));
		ASSERT_EQ (m.I[k], row.level);
		ASSERT_EQ (m.N[k], row.n);
		ASSERT_NEAR (m.P[k], row.p, 1e-15);
		ASSERT_NEAR (m.S[k], row.s, 1e-15);
		nonempty += m.N[k] > 0;
	}
	ASSERT_EQ (m.Ngp, nonempty);
}
