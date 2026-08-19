#pragma once

#include "test_3d_morphology_common.h"   // gtest, <string>, <vector>, Pixel3, calc_eigvals
#include "test_ref_vals.h"               // ref_vals_list

// Kernel mechanics for the 3D shape maths: the covariance matrix of a point cloud and its
// eigenvalues, checked directly rather than through a feature. Nothing here reads an image or names
// a feature, so this is a mechanics file (SPEC 2) and it claims no vetting for any registry row --
// the PCA features those eigenvalues feed are vetted against MIRP in test_3d_morphology_mirp.h.
//
// Provenance (SPEC 6.4). numpy is NOT a SPEC 4 oracle token and nothing here claims to be an
// oracle: this is the reference for a kernel check, and the features that consume the kernel are
// vetted against MIRP next door.
//   tool      = numpy 2.4.6 (python 3.12.13)
//   quantity  = numpy.cov(cloud.T, ddof=1) and numpy.linalg.eigvalsh, sorted descending
//   generator = tests/vetting/oracles/gen_morphology3d_covmatrix_numpy.py (re-verifies every pin)
//
// These goldens were MATLAB `cov`/`eig` output quoted to five significant figures, from a session
// that cannot be re-run from this tree. numpy computes the same two quantities -- `calc_covariance`
// normalises by n-1, which is the sample covariance both MATLAB `cov` and numpy `ddof=1` compute --
// so the reference is now reproducible and carries full precision instead of five digits. The
// numbers agree with the MATLAB ones they replace at every digit MATLAB printed.
//
// The band matters here. The old assertions passed `frac_tolerance = 1.0` to agrees_gt(), which
// makes the tolerance the ground truth itself: every one of the twelve comparisons accepted any
// value within +/-100%, so a covariance off by a factor of two -- or a normalisation switched from
// n-1 to n, which moves these entries by 10% -- would have passed. They are pinned at rel=1e-9
// below, which is what the arithmetic actually delivers.

// The point cloud under test: ten voxels, layout X, Y, Z, intensity. Intensity is uniform because
// calc_cov_matrix is a geometric moment of the coordinates and does not read it.
static const std::vector<Pixel3> morphology_3d_covmatrix_cloud =
{
    {9,     96,     4,      1000},
    {26,    55,     89,     1000},
    {80,    52,     91,     1000},
    {3,     23,     80,     1000},
    {93,    49,     10,     1000},
    {73,    62,     26,     1000},
    {49,    68,     34,     1000},
    {58,    40,     68,     1000},
    {24,    37,     14,     1000},
    {46,    99,     72,     1000}
};

// Row-major upper-and-lower 3x3, i.e. K[0][0], K[0][1], K[0][2], K[1][0], ... K[2][2].
static const ref_vals_list<double> morphology_3d_mechanics_covmatrix_ref_vals
{
     927.65555555555550,  -9.3444444444444361, -60.088888888888932,
      -9.3444444444444361, 595.21111111111100, -191.31111111111113,
     -60.088888888888932, -191.31111111111113, 1193.2888888888886
};

// Eigenvalues of the matrix above, descending -- the order calc_eigvals returns them in.
static const ref_vals_list<double> morphology_3d_mechanics_eigenvalues_ref_vals
{
    1258.4359559296070,
     920.19859231791270,
     537.52100730803570
};

void test_3d_morphology_covmatrix_and_eigenvals_mechanics()
{
    SCOPED_TRACE("MECHANICS__3d_morphology_covmatrix_and_eigenvals");

    double K[3][3];
    Pixel3::calc_cov_matrix (K, morphology_3d_covmatrix_cloud);

    // verdict #1 -- the covariance matrix, at the precision double arithmetic on ten points holds
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            const double golden = morphology_3d_mechanics_covmatrix_ref_vals[i * 3 + j];
            ASSERT_TRUE (agrees_gt (K[i][j], golden, 1e9))
                << "K[" << i << "][" << j << "] actual=" << K[i][j] << " numpy=" << golden;
        }

    // verdict #2 -- the eigenvalues, through the Jacobi solver in helpers.cpp
    double L[3];
    ASSERT_TRUE (Nyxus::calc_eigvals (L, K));

    for (int i = 0; i < 3; i++)
    {
        const double golden = morphology_3d_mechanics_eigenvalues_ref_vals[i];
        ASSERT_TRUE (agrees_gt (L[i], golden, 1e9))
            << "L[" << i << "] actual=" << L[i] << " numpy=" << golden;
    }
}
