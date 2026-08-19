#pragma once

#include <cmath>
#include "test_3d_morphology_common.h"
#include "test_ref_vals.h"   // ref_vals_map, and the <string> / <vector> it already includes

// ---------------------------------------------------------------------------------------------------
// MIRP-oracle'd 3D morphology: the five PCA axis features and the two volume features (registry:
// mirp / vetted). Shared fixture lives in test_3d_morphology_common.h.
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
static ref_vals_map<double> morphology_3d_mirp_pca_ref_vals
{
    { "3ELONGATION", 0.8433210559938976 },      // morph_pca_elongation
    { "3FLATNESS", 0.6829975804590384 },        // morph_pca_flatness
    { "3LEAST_AXIS_LEN", 71.51449974198198 },   // morph_pca_least_axis
    { "3MAJOR_AXIS_LEN", 104.70681271508683 },  // morph_pca_maj_axis
    { "3MINOR_AXIS_LEN", 88.30145986864228 }    // morph_pca_min_axis
};

// ORACLE goldens -- MIRP volume quantities, same run and same recipe as the axes above.
//
// These exist because the three `matlab` rows of this family were pinned from an offline MATLAB
// session with no in-repo generator (SPEC 6.4), and MATLAB cannot be re-run from this tree: there is
// no licence here and Octave's `image` package has no `regionprops3`. MIRP computes the same two
// quantities, is runnable from the tree, and agrees with the MATLAB numbers, so the values stop
// resting on a session nobody can repeat.
//
//   3VOXEL_VOLUME       = morph_vol_approx -- IBSI "volume (voxel counting)", the ROI voxel count
//                         times the voxel volume. MATLAB regionprops3 Volume is the same definition.
//   3VOLUME_CONVEXHULL  = morph_volume / morph_vol_dens_conv_hull. IBSI defines volume density
//                         (convex hull) as V_mesh / V_convex, so dividing the mesh volume by that
//                         density backs out MIRP's convex-hull volume.
//
// Measured on this fixture:
//
//   feature              Nyxus     MIRP        MATLAB regionprops3   Nyxus vs MIRP
//   3VOXEL_VOLUME        274432    274432      274432               0
//   3VOLUME_CONVEXHULL   478516    496958.3    497824               3.71%
//
// The two independent triangulated hulls (MIRP's qhull, MATLAB's) agree with each other to 0.17%,
/// and Nyxus sits 3.71% below MIRP and 3.88% below regionprops3 -- because Nyxus builds a DISCRETE
// VOXEL hull and they triangulate.
// That is the citation behind the 5% band this file and test_3d_morphology_matlab.h both use: it is
// a definitional difference between a voxelised and a triangulated hull, now measured against two
// tools rather than one, not slack absorbing an unexplained gap.
//
// 3MESH_VOLUME is deliberately absent. Nyxus aliases it to the convex-hull volume instead of
// integrating the ROI surface mesh, so against IBSI's volume (mesh) -- MIRP morph_volume,
// 274338.34 here -- it reads 74% high. Pinning the two against each other would record that as
// agreement; the divergence is filed as a defect instead.
static ref_vals_map<double> morphology_3d_mirp_volume_ref_vals
{
    { "3VOXEL_VOLUME", 274432.0 },              // morph_vol_approx
    { "3VOLUME_CONVEXHULL", 496958.3201121965 } // morph_volume / morph_vol_dens_conv_hull
};

// Bands as the percent relative_absdiff may reach, measured per feature (SPEC 7).
static ref_vals_map<double> morphology_3d_mirp_volume_ref_tols
{
    // identical definition, exact agreement measured
    { "3VOXEL_VOLUME", 0.1 },
    // voxel hull against triangulated hull, measured 3.71%; band 5%
    { "3VOLUME_CONVEXHULL", 5.0 }
};

static void assert_3d_morphology_volume_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MIRP_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_mirp_volume_ref_vals.count(fname) > 0) << fname;
    ASSERT_TRUE(morphology_3d_mirp_volume_ref_tols.count(fname) > 0) << fname << " has a golden but no stated band";

    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);

    const double expected = morphology_3d_mirp_volume_ref_vals[fname];
    const double pct = 100.0 * std::abs(actual - expected) / std::abs(expected);
    ASSERT_LE(pct, morphology_3d_mirp_volume_ref_tols[fname])
        << fname << " actual=" << actual << " mirp=" << expected
        << " band=" << morphology_3d_mirp_volume_ref_tols[fname] << "%";
}

void test_3d_morphology_voxel_volume_mirp() {
    assert_3d_morphology_volume_mirp ("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}

void test_3d_morphology_volume_convex_hull_mirp() {
    assert_3d_morphology_volume_mirp ("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
}

// Same definition on both sides -- 4*sqrt of the mask covariance eigenvalues, and their ratios --
// so Nyxus reproduces MIRP to double precision (measured <= 2.6e-16 on all five). frac_tolerance
// = 1e9, i.e. rel=1e-9: tighter than SPEC 7's same-definition tier because the agreement is exact,
// and a band wider than the divergence it absorbs hides drift.
static void assert_3d_morphology_feature_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MIRP_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_mirp_pca_ref_vals.count(fname) > 0) << fname;

    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);

    ASSERT_TRUE(agrees_gt(actual, morphology_3d_mirp_pca_ref_vals[fname], 1e9))
        << fname << " actual=" << actual << " mirp=" << morphology_3d_mirp_pca_ref_vals[fname];
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
