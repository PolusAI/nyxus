#pragma once

#include "test_3d_morphology_common.h"   // the fixture, and the <cmath> test_main_nyxus.h brings with it
#include "test_ref_vals.h"               // ref_vals_map, and the <string> / <vector> it already includes

// ---------------------------------------------------------------------------------------------------
// MIRP-oracle'd 3D morphology: the five PCA axis features and the three volume features (registry:
// mirp / vetted). Shared fixture lives in test_3d_morphology_common.h.
//
// The voxel-counting and convex-hull volumes are also asserted separately against MATLAB
// regionprops3 in test_3d_morphology_matlab.h; the mesh volume is not, since regionprops3 has none.
// SPEC 3 permits several assertion rows for one feature and counts the
// feature as vetted when at least one feature x config x reference assertion is vetted.
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
// MIRP's `morph_area_mesh` is not pinned here: it is a marching-cubes mesh area while Nyxus' 3AREA
// counts exposed voxel faces (46739 against 59992), and the five area-derived ratios inherit that
// convention difference. They stay regression-only -- see
// tests/vetting/audit/morphology_3d_mirp_vetting_report.md. Nor are the density family
// (morph_*_dens_*), morph_diam, morph_integ_int and the spatial-autocorrelation pair (morph_moran_i,
// morph_geary_c): Nyxus implements no feature for any of them.
static const ref_vals_map<double> morphology_3d_mirp_pca_ref_vals
{
    { "3ELONGATION", 0.8433210559938976 },      // morph_pca_elongation
    { "3FLATNESS", 0.6829975804590384 },        // morph_pca_flatness
    { "3LEAST_AXIS_LEN", 71.51449974198198 },   // morph_pca_least_axis
    { "3MAJOR_AXIS_LEN", 104.70681271508683 },  // morph_pca_maj_axis
    { "3MINOR_AXIS_LEN", 88.30145986864228 }    // morph_pca_min_axis
};

// ORACLE golden -- MIRP's voxel-counting volume, same run and same recipe as the axes above.
//
//   3VOXEL_VOLUME = morph_vol_approx -- IBSI "volume (voxel counting)", the ROI voxel count times the
//                   voxel volume. MATLAB regionprops3 Volume is the same definition, asserted
//                   separately in test_3d_morphology_matlab.h.
//
// Both voxel-volume oracles return 274432.00. Nyxus' 0.5236 packing-density approximation returns
// 274431.36 on this fixture: a 2.34e-04% residual, within the existing SPEC 7 same-definition
// rel=1e-3 tier. The rows therefore agree within tolerance; they do not claim a bitwise-exact Nyxus
// result.
static const ref_vals_map<double> morphology_3d_mirp_volume_ref_vals
{
    { "3VOXEL_VOLUME", 274432.0 }                // morph_vol_approx
};

// ORACLE goldens -- the ROI surface mesh and its convex hull, same run, same recipe.
//
//   3MESH_VOLUME        = morph_volume, IBSI volume (mesh)
//   3VOLUME_CONVEXHULL  = morph_volume / morph_vol_dens_conv_hull. IBSI defines volume density
//                         (convex hull) as V_mesh / V_convex, so dividing the mesh volume by that
//                         density backs out MIRP's convex-hull volume.
//
// The same quantities on both sides: Nyxus and MIRP both build the marching-cubes surface of the mask
// at the 0.5 isolevel, integrate it, and hull its vertices, so no convention difference is left for a
// band to absorb.
static const ref_vals_map<double> morphology_3d_mirp_mesh_ref_vals
{
    { "3MESH_VOLUME", 274338.34375 },                     // morph_volume
    { "3VOLUME_CONVEXHULL", 496958.3201121965 }           // morph_volume / morph_vol_dens_conv_hull
};

// Same definition and the same surface on both sides, so what is left is MIRP's own precision: it
// carries the mesh in float32. Measured divergence: 3.8e-08 on 3MESH_VOLUME and 2.7e-08 on
// 3VOLUME_CONVEXHULL, which is that storage rather than a difference in the quantity.
// frac_tolerance = 1e6, i.e. rel=1e-6: more than an order of magnitude above the measured residuals,
// and far below any change of definition -- the hull of the voxel centres sits 3.3% below MIRP's
// hull, and the hull volume 81% above the mesh volume.
static void assert_3d_morphology_mesh_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MIRP_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_mirp_mesh_ref_vals.count(fname) > 0) << fname;

    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);

    ASSERT_TRUE(agrees_gt(actual, morphology_3d_mirp_mesh_ref_vals.at(fname), 1e6))
        << fname << " actual=" << actual << " mirp=" << morphology_3d_mirp_mesh_ref_vals.at(fname);
}

void test_3d_morphology_mesh_volume_mirp() {
    assert_3d_morphology_mesh_mirp ("3MESH_VOLUME", Feature3D::MESH_VOLUME);
}

static void assert_3d_morphology_volume_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MIRP_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_mirp_volume_ref_vals.count(fname) > 0) << fname;

    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);

    const double expected = morphology_3d_mirp_volume_ref_vals.at(fname);
    const double band = morphology_3d_volume_ref_tol_pct(fname);
    const double pct = 100.0 * std::abs(actual - expected) / std::abs(expected);
    ASSERT_LE(pct, band)
        << fname << " actual=" << actual << " mirp=" << expected
        << " band=" << band << "%";
}

void test_3d_morphology_voxel_volume_mirp() {
    assert_3d_morphology_volume_mirp ("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}

void test_3d_morphology_volume_convex_hull_mirp() {
    assert_3d_morphology_mesh_mirp ("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
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

    ASSERT_TRUE(agrees_gt(actual, morphology_3d_mirp_pca_ref_vals.at(fname), 1e9))
        << fname << " actual=" << actual << " mirp=" << morphology_3d_mirp_pca_ref_vals.at(fname);
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
