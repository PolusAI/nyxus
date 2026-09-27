#pragma once

#include <array>
#include <map>
#include <utility>
#include "../src/nyx/features/3d_mesh.h"   // Triangle3, build_roi_surface_mesh
#include "../src/nyx/features/pixel.h"   // Pixel3
#include "../src/nyx/helpers/helpers.h"   // Nyxus::calc_eigvals
#include "test_3d_morphology_common.h"    // gtest, <string>, <vector>, agrees_gt
#include "test_ref_vals.h"                // ref_vals_list

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
// `calc_covariance` normalises by n-1, so the quantity is the sample covariance, which is what both
// numpy `ddof=1` and MATLAB `cov` compute -- the reference is the same quantity, not a near one.
// The twelve pins carry full precision and are asserted at rel=1e-9, which is what the arithmetic
// delivers: a 1e-7 relative perturbation of one eigenvalue fails the test. Where this reference
// came from: tests/vetting/audit/morphology_3d_golden_regen.md, "Covariance / eigenvalue kernel".

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

// The property the marching-cubes case table exists to have: the surface it assembles is closed.
// mesh_volume() integrates the divergence theorem over it, which is only a volume if every edge is
// traversed as often one way as the other -- an open facet would leak an arbitrary amount into the
// answer, silently. Checked over a pseudo-random blob rather than a solid, because the cases that
// can leave a cube open are the ragged ones a compact ROI never reaches.
//
// This is a mechanics check (SPEC 2): it names no feature and claims no vetting. The features the
// mesh feeds are vetted against MIRP in test_3d_morphology_mirp.h and against closed-form solids in
// test_3d_morphology_analytic.h.
void test_3d_morphology_surface_mesh_closed_mechanics()
{
    SCOPED_TRACE("MECHANICS__3d_morphology_surface_mesh_closed");

    // a deterministic pseudo-random occupancy, dense enough to exercise the ambiguous cube cases
    std::vector<Pixel3> cloud;
    unsigned int state = 20260925u;
    for (int x = 0; x < 14; x++)
        for (int y = 0; y < 14; y++)
            for (int z = 0; z < 14; z++)
            {
                state = state * 1664525u + 1013904223u;
                if ((state >> 16) & 1u)
                    cloud.push_back (Pixel3(x, y, z, 1000));
            }
    ASSERT_GT (cloud.size(), 1000u);

    std::vector<Nyxus::Triangle3> mesh;
    Nyxus::build_roi_surface_mesh (mesh, cloud);
    ASSERT_FALSE (mesh.empty());

    // count every directed edge; a closed oriented surface uses each one as often as its reverse
    std::map<std::pair<std::array<double,3>, std::array<double,3>>, int> halfedges;
    for (const auto& t : mesh)
    {
        const std::array<double,3> v[3] = {
            { t.a[0], t.a[1], t.a[2] }, { t.b[0], t.b[1], t.b[2] }, { t.c[0], t.c[1], t.c[2] } };
        for (int i = 0; i < 3; i++)
            halfedges[{ v[i], v[(i + 1) % 3] }]++;
    }

    size_t unbalanced = 0;
    for (const auto& he : halfedges)
    {
        auto back = halfedges.find ({ he.first.second, he.first.first });
        if (back == halfedges.end() || back->second != he.second)
            unbalanced++;
    }
    ASSERT_EQ (unbalanced, 0u) << mesh.size() << " triangles, " << unbalanced << " unbalanced edges";
}
