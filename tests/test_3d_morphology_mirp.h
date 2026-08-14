#pragma once

#include <string>
#include "test_3d_morphology_common.h"
#include "test_ref_vals.h"

// ---------------------------------------------------------------------------------------------------
// MIRP-oracle'd 3D morphology: the five PCA axis features (registry: mirp / vetted). Shared fixture
// lives in test_3d_morphology_common.h.
// ---------------------------------------------------------------------------------------------------

// ORACLE goldens -- MIRP morphology (IBSI section 3.1 `morph_*`).
//
// Provenance (SPEC 6.4):
//   tool         = mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3)
//   config       = by_slice=false, base_feature_families="morphology",
//                  base_discretisation_method="none", native 1x1x1 spacing
//   fixture      = tests/data/nifti/phantoms/ut_inten.nii + ut_mask57.nii, label 57
//   recipe       = morphology3d.mirp_ibsi
//   generator    = tests/vetting/oracles/gen_morphology3d_mirp.py (re-verifies every pin below)
//
// MIRP names these axes by role and Nyxus by size rank; the two agree only because MAJOR is the
// largest eigenvalue, which is the correspondence the generator re-checks on every run.
//
// The rest of MIRP's morph_* block is deliberately not pinned here: its `morph_area_mesh` is a
// marching-cubes mesh area while Nyxus' 3AREA counts exposed voxel faces (46739 against 59992), and
// the area-derived features inherit that convention difference. They stay regression-only -- see
// tests/vetting/audit/morphology_3d_mirp_vetting_report.md.
static ref_vals_map<double> morphology_3d_mirp_ref_vals
{
    { "3ELONGATION", 0.8433210559938976 },      // morph_pca_elongation
    { "3FLATNESS", 0.6829975804590384 },        // morph_pca_flatness
    { "3LEAST_AXIS_LEN", 71.51449974198198 },   // morph_pca_least_axis
    { "3MAJOR_AXIS_LEN", 104.70681271508683 },  // morph_pca_maj_axis
    { "3MINOR_AXIS_LEN", 88.30145986864228 }    // morph_pca_min_axis
};

// Same definition on both sides -- 4*sqrt of the mask covariance eigenvalues, and their ratios --
// so Nyxus reproduces MIRP to double precision (measured <= 2.6e-16 on all five). frac_tolerance
// = 1e9, i.e. rel=1e-9: tighter than SPEC 7's same-definition tier because the agreement is exact,
// and a band wider than the divergence it absorbs hides drift.
static void assert_3d_morphology_feature_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MIRP_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_mirp_ref_vals.count(fname) > 0) << fname;

    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);

    ASSERT_TRUE(agrees_gt(actual, morphology_3d_mirp_ref_vals[fname], 1e9))
        << fname << " actual=" << actual << " mirp=" << morphology_3d_mirp_ref_vals[fname];
}

void test_3d_morphology_major_axis_len_mirp() {
    assert_3d_morphology_feature_mirp ("3MAJOR_AXIS_LEN", Feature3D::MAJOR_AXIS_LEN);
}

void test_3d_morphology_minor_axis_len_mirp() {
    assert_3d_morphology_feature_mirp ("3MINOR_AXIS_LEN", Feature3D::MINOR_AXIS_LEN);
}

void test_3d_morphology_least_axis_len_mirp() {
    assert_3d_morphology_feature_mirp ("3LEAST_AXIS_LEN", Feature3D::LEAST_AXIS_LEN);
}

void test_3d_morphology_elongation_mirp() {
    assert_3d_morphology_feature_mirp ("3ELONGATION", Feature3D::ELONGATION);
}

void test_3d_morphology_flatness_mirp() {
    assert_3d_morphology_feature_mirp ("3FLATNESS", Feature3D::FLATNESS);
}
