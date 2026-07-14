#pragma once

#include "test_3d_morphology_common.h"

// ---------------------------------------------------------------------------------------------------
// Migrated from test_3d_shape.h (Wave 8). MATLAB-oracle'd 3D morphology:
//  - 3MESH_VOLUME (registry: matlab / vetted).
//  - the covariance-matrix + eigenvalue math (Pixel3::calc_cov_matrix / Nyxus::calc_eigvals) whose
//    ground truth is produced by MATLAB cov()/eig() - the linear-algebra core behind 3D shape.
// Shared fixture lives in test_3d_morphology_common.h.
// ---------------------------------------------------------------------------------------------------

void test_3shape_meshvolume() {
    test_3shape_feature ("3MESH_VOLUME", Feature3D::MESH_VOLUME);
}

void test_3shape_covmatrix_and_eigenvals() {
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
