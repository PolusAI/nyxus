#pragma once

#include <cmath>
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/helpers/helpers.h"
#include "test_3d_morphology_common.h"
#include "test_ref_vals.h"   // ref_vals_map, and the <string> / <vector> it already includes

// ---------------------------------------------------------------------------------------------------
// MATLAB-oracle'd 3D morphology:
//  - 3MESH_VOLUME / 3VOLUME_CONVEXHULL / 3VOXEL_VOLUME (registry: matlab / vetted).
//  - the covariance-matrix + eigenvalue math (Pixel3::calc_cov_matrix / Nyxus::calc_eigvals) whose
//    ground truth is produced by MATLAB cov()/eig() - the linear-algebra core behind 3D shape.
// Shared fixture lives in test_3d_morphology_common.h.
// ---------------------------------------------------------------------------------------------------

// ORACLE goldens -- MATLAB regionprops3 shape built-ins.
//
// Provenance (SPEC 6.4):
//   tool         = MATLAB R2025b, Image Processing Toolbox regionprops3
//   properties   = Volume -> 3VOXEL_VOLUME; ConvexVolume -> 3VOLUME_CONVEXHULL and 3MESH_VOLUME
//   fixture      = tests/data/nifti/phantoms/ut_inten.nii + ut_mask57.nii, label 57
//   generated    = offline. MATLAB cannot be re-run from this tree -- no licence here, and Octave's
//                  image package has no regionprops3 -- so there is no MATLAB generator to check in.
//                  What closes the SPEC 6.4 gap instead is a second tool that IS runnable:
//                  tests/vetting/oracles/gen_morphology3d_mirp.py pins MIRP's own values for the
//                  same two quantities (morph_vol_approx; morph_volume / morph_vol_dens_conv_hull)
//                  in test_3d_morphology_mirp.h, and they agree with the MATLAB numbers here --
//                  exactly for 3VOXEL_VOLUME, to 0.17% for the convex hull. The MATLAB pins below
//                  are therefore reproducible in substance even though the session that produced
//                  them is not repeatable. See tests/vetting/not_covered.md section C.
//
// 3AREA is deliberately absent: regionprops3 SurfaceArea disagrees by more than 10%, so pinning it
// here would assert an agreement that does not hold.
//
// This table is also read by the 3D coverage sweep (test_3d_coverage_common.h), the same way that
// header already reads the per-family *_3d_pyradiomics_ref_vals tables from their own oracle files.
static ref_vals_map<double> morphology_3d_matlab_ref_vals
{
	{ "3MESH_VOLUME", 497824.0 },
	{ "3VOXEL_VOLUME", 274432.0 },
	{ "3VOLUME_CONVEXHULL", 497824.0 }
};

// Per-feature agreement bands, as the percent that relative_absdiff_pct() may reach. Measured per
// feature rather than shared, because a band looser than the divergence it absorbs hides drift
// (SPEC 7).
static ref_vals_map<double> morphology_3d_matlab_ref_tols
{
	// Same definition on both sides (count of voxels x voxel volume); measured 2.3e-04%. SPEC 7
	// same-definition tier.
	{ "3VOXEL_VOLUME", 0.1 },
	// Convex-hull convention difference: Nyxus builds a discrete voxel hull, regionprops3
	// ConvexVolume triangulates. Measured 3.88% -- relative_absdiff_pct of Nyxus 478516 against the
	// 497824 pinned below -- stated band 5% (SPEC 7 known-method-divergence).
	// Citation for the band, now from two tools rather than one: MIRP's convex-hull volume for this
	// fixture, backed out of IBSI's volume density (convex hull) = V_mesh / V_convex, is 496958.3
	// against regionprops3's 497824 -- two independent triangulated hulls agreeing to 0.17%, with
	// Nyxus 3.88% below regionprops3 and 3.71% below MIRP, the 0.17% between the two tools being the
	// whole spread. The divergence is voxelised-vs-triangulated hull, not tool noise; see
	// test_3d_morphology_mirp.h and audit/morphology_3d_mirp_vetting_report.md.
	{ "3VOLUME_CONVEXHULL", 5.0 },
	// Nyxus aliases 3MESH_VOLUME to the convex-hull volume rather than integrating the surface mesh,
	// so it inherits the hull convention and the same measured 3.88%. The alias is why one golden
	// serves two keys; if 3MESH_VOLUME ever becomes a real mesh integral this band must be revisited.
	// It is not one today: IBSI's volume (mesh) for this fixture, MIRP morph_volume, is 274338.34,
	// and 3MESH_VOLUME reads 74% above it. This band judges the alias against the quantity the alias
	// actually computes; it says nothing about the feature deserving its name.
	{ "3MESH_VOLUME", 5.0 }
};

// Judges against MATLAB's number and this file's own band -- never against the snapshot table in
// test_3d_morphology_regression.h, which pins Nyxus output for the same features.
static void assert_3d_morphology_feature_matlab (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MATLAB_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_matlab_ref_vals.count(fname) > 0) << fname;
    ASSERT_TRUE(morphology_3d_matlab_ref_tols.count(fname) > 0) << fname << " has a golden but no stated band";

    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);

    const double expected = morphology_3d_matlab_ref_vals[fname];
    const double pct = std::abs(expected) == 0.0 ? (actual == expected ? 0.0 : 100.0)
                                                 : 100.0 * std::abs(actual - expected) / std::abs(expected);
    ASSERT_LE(pct, morphology_3d_matlab_ref_tols[fname])
        << fname << " actual=" << actual << " MATLAB regionprops3=" << expected
        << " band=" << morphology_3d_matlab_ref_tols[fname] << "%";
}

void test_3d_morphology_mesh_volume_matlab() {
    assert_3d_morphology_feature_matlab ("3MESH_VOLUME", Feature3D::MESH_VOLUME);
}

// The other two goldens in the table above. They had a MATLAB value and a stated band but no
// assertion of their own: the only thing comparing them against MATLAB was the parameterized sweep in
// test_3d_morphology_coverage.h, whose case names carry no oracle token, so the registry's
// oracle=matlab claim rested on a test that does not say so (SPEC 6.2).
void test_3d_morphology_voxel_volume_matlab() {
    assert_3d_morphology_feature_matlab ("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}

void test_3d_morphology_volume_convex_hull_matlab() {
    assert_3d_morphology_feature_matlab ("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
}

void test_3d_morphology_covmatrix_and_eigenvals_matlab() {
    std::vector<Pixel3> cloud =
    {
        // layout: X, Y, Z, intensity
        {9,     96,     4,      1000},
        {26,    55,     89,     1000},
        {80,    52,     91,	1000 },
        {3,     23,	80,	1000},
        {93,    49,	10,	1000},
        {73,    62,	26,	1000},
        {49,    68,	34,	1000},
        {58,    40,	68,	1000},
        {24,    37,	14,	1000},
        {46,    99,	72,	1000}
    };

    double K[3][3];
    Pixel3::calc_cov_matrix (K, cloud);

    // verdict #1 (covariance matrix)
    /*
            producing the ground truth with MATLAB:
            cloud = [
                9    96     4,
                26    55    89,
                80    52    91,
                3    23    80,
                93    49    10,
                73    62    26,
                49    68    34,
                58    40    68,
                24    37    14,
                46    99    72] ;
            K = cov(cloud)
            >>
                ans =
                1.0e+03 *
                0.9277 - 0.0093 - 0.0601
                - 0.0093    0.5952 - 0.1913
                - 0.0601 - 0.1913    1.1933
    */
    double gtK[3][3] =
    {
        { 0.9277e3,    -0.0093e3,    -0.0601e3 },
        { -0.0093e3,   0.5952e3,     -0.1913e3 },
        { -0.0601e3,   -0.1913e3,    1.1933e3 }
    };

    double tol = 1.0;
    ASSERT_TRUE (agrees_gt (K[0][0], gtK[0][0], tol));
    ASSERT_TRUE (agrees_gt (K[0][1], gtK[0][1], tol));
    ASSERT_TRUE (agrees_gt (K[0][2], gtK[0][2], tol));

    ASSERT_TRUE (agrees_gt (K[1][0], gtK[1][0], tol));
    ASSERT_TRUE (agrees_gt (K[1][1], gtK[1][1], tol));
    ASSERT_TRUE (agrees_gt (K[1][2], gtK[1][2], tol));

    ASSERT_TRUE (agrees_gt (K[2][0], gtK[2][0], tol));
    ASSERT_TRUE (agrees_gt (K[2][1], gtK[2][1], tol));
    ASSERT_TRUE (agrees_gt (K[2][2], gtK[2][2], tol));

    double L[3];
    ASSERT_TRUE(Nyxus::calc_eigvals(L, K));

    // verdict #2 (eigenvalues)
    /*
    producing the ground truth with MATLAB:
            L = eig (K)
            sort(L, 'descend')
    */
    double gtL[3] = { 1.2584e3, 0.9202e3, 0.5375e3 };
    ASSERT_TRUE (agrees_gt(L[0], gtL[0], tol));
    ASSERT_TRUE (agrees_gt(L[1], gtL[1], tol));
    ASSERT_TRUE (agrees_gt(L[2], gtL[2], tol));
}
