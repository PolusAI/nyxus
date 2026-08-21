#pragma once

#include "test_3d_morphology_common.h"   // the fixture, and the <cmath> test_main_nyxus.h brings with it
#include "test_ref_vals.h"               // ref_vals_map, and the <string> / <vector> it already includes

// ---------------------------------------------------------------------------------------------------
// MIRP-oracle'd 3D morphology: the five PCA axis features and the three volume features (registry:
// mirp / vetted). Shared fixture lives in test_3d_morphology_common.h.
//
// MIRP is the family's only oracle: it is runnable from this tree (gen_morphology3d_mirp.py), and
// SPEC 3 allows one oracle per assertion, so the MATLAB regionprops3 numbers quoted below are a
// corroborating measurement and nothing asserts against them. Why MATLAB is not the oracle here:
// tests/vetting/audit/morphology_3d_golden_regen.md, "Retired: the MATLAB goldens".
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
static const ref_vals_map<double> morphology_3d_mirp_pca_ref_vals
{
    { "3ELONGATION", 0.8433210559938976 },      // morph_pca_elongation
    { "3FLATNESS", 0.6829975804590384 },        // morph_pca_flatness
    { "3LEAST_AXIS_LEN", 71.51449974198198 },   // morph_pca_least_axis
    { "3MAJOR_AXIS_LEN", 104.70681271508683 },  // morph_pca_maj_axis
    { "3MINOR_AXIS_LEN", 88.30145986864228 }    // morph_pca_min_axis
};

// ORACLE goldens -- MIRP volume quantities, same run and same recipe as the axes above.
//
// These replace the family's three `matlab` goldens, which had no in-repo generator (SPEC 6.4).
// MIRP computes the same quantities, is runnable from the tree, and reproduces the MATLAB numbers,
// so the values stop resting on a session nobody can repeat.
//
//   3VOXEL_VOLUME       = morph_vol_approx -- IBSI "volume (voxel counting)", the ROI voxel count
//                         times the voxel volume. MATLAB regionprops3 Volume is the same definition.
//   3VOLUME_CONVEXHULL  = morph_volume / morph_vol_dens_conv_hull. IBSI defines volume density
//                         (convex hull) as V_mesh / V_convex, so dividing the mesh volume by that
//                         density backs out MIRP's convex-hull volume.
//
// Measured on this fixture:
//
//   feature              Nyxus       MIRP        MATLAB regionprops3   Nyxus vs MIRP
//   3VOXEL_VOLUME        274431.36   274432      274432               2.3e-4%
//   3VOLUME_CONVEXHULL   479997.83   496958.3    497824               3.41%
//   3MESH_VOLUME         479997.83   496958.3    497824               3.41%   (the same hull number --
//                                                                              see the alias note below)
//
// Both tools count the same 274432 voxels; Nyxus' 3VOXEL_VOLUME then scales the count by
// 4/3*pi*r^3 / 0.5236 (a cubic-lattice ball-packing correction that is 1 to within 2.3e-6), which is
// the whole of that first residual.
//
// The two independent triangulated hulls (MIRP's qhull, MATLAB's) agree with each other to 0.17%,
// and Nyxus sits 3.41% below MIRP and 3.58% below regionprops3 -- because Nyxus builds a DISCRETE
// VOXEL hull and they triangulate.
// That is the citation behind this file's 5% band: it is a definitional difference between a
// voxelised and a triangulated hull, measured against two tools rather than one, not slack absorbing
// an unexplained gap. The MATLAB column stays in the table above as the corroborating measurement it
// is; nothing below asserts against it.
//
// 3MESH_VOLUME is pinned against that same convex-hull number, because the hull volume is what the
// feature computes: Nyxus aliases MESH_VOLUME to the convex-hull volume instead of integrating the
// ROI surface mesh. Judged against IBSI's volume (mesh) -- MIRP morph_volume, 274338.34 here -- it
// reads 75% high, and that gap is NOT absorbed by the band below; it is filed as a defect in
// tests/vetting/not_covered.md. So the assertion judges the alias against the quantity the alias
// actually computes, and says nothing about the feature deserving its name. The registry row records
// the same, and if 3MESH_VOLUME ever becomes a real mesh integral this golden must be revisited.
static const ref_vals_map<double> morphology_3d_mirp_volume_ref_vals
{
    { "3VOXEL_VOLUME", 274432.0 },               // morph_vol_approx
    { "3VOLUME_CONVEXHULL", 496958.3201121965 }, // morph_volume / morph_vol_dens_conv_hull
    { "3MESH_VOLUME", 496958.3201121965 }        // the same hull volume: Nyxus aliases MESH_VOLUME to it
};

// Bands as the percent relative_absdiff may reach, measured per feature (SPEC 7).
static const ref_vals_map<double> morphology_3d_mirp_volume_ref_tols
{
    // same voxel count on both sides, 2.3e-4% left by the ball-packing scale factor; band 0.1%
    { "3VOXEL_VOLUME", 0.1 },
    // voxel hull against triangulated hull, measured 3.41%; band 5%
    { "3VOLUME_CONVEXHULL", 5.0 },
    // the alias inherits the hull convention and the same measured 3.41%
    { "3MESH_VOLUME", 5.0 }
};

static void assert_3d_morphology_volume_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MIRP_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_mirp_volume_ref_vals.count(fname) > 0) << fname;
    ASSERT_TRUE(morphology_3d_mirp_volume_ref_tols.count(fname) > 0) << fname << " has a golden but no stated band";

    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);

    const double expected = morphology_3d_mirp_volume_ref_vals.at(fname);
    const double pct = 100.0 * std::abs(actual - expected) / std::abs(expected);
    ASSERT_LE(pct, morphology_3d_mirp_volume_ref_tols.at(fname))
        << fname << " actual=" << actual << " mirp=" << expected
        << " band=" << morphology_3d_mirp_volume_ref_tols.at(fname) << "%";
}

void test_3d_morphology_voxel_volume_mirp() {
    assert_3d_morphology_volume_mirp ("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}

void test_3d_morphology_volume_convex_hull_mirp() {
    assert_3d_morphology_volume_mirp ("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
}

void test_3d_morphology_mesh_volume_mirp() {
    assert_3d_morphology_volume_mirp ("3MESH_VOLUME", Feature3D::MESH_VOLUME);
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
