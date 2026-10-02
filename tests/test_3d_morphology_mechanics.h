#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <utility>
#include "../src/nyx/3rdparty/quickhull.hpp"   // quick_hull
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

// The convex hull kernel, quick_hull (src/nyx/3rdparty/quickhull.hpp), driven directly and the way
// D3_SurfaceFeature::build_hull drives it: double points, eps = 16 * DBL_EPSILON * the largest
// coordinate. The clouds are lattice ellipsoids, every voxel with (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1,
// because their hull faces carry many voxels exactly on a face or on an edge line -- the points the
// facet predicates must treat as within eps of a plane, not as above or below it.
//
// This is a mechanics check (SPEC 2): it names no feature and claims no vetting. 3VOLUME_CONVEXHULL,
// which is built on this kernel, is asserted against closed forms in test_3d_morphology_analytic.h.
//
// Provenance (SPEC 6.4). scipy is NOT a SPEC 4 oracle token and nothing here claims to be an oracle.
//   tool      = scipy 1.10.1 (scipy.spatial.ConvexHull, i.e. qhull; numpy 1.23.5, python 3.8.20)
//   quantity  = the volume of the convex hull of the ellipsoid's voxel centres
//   generator = tests/vetting/oracles/gen_morphology3d_quick_hull_scipy.py (re-verifies every pin)
//
// A hull of lattice points has a volume that is a multiple of 1/6; each pin is that multiple, which
// qhull reproduces to within 1e-15 relative.

// Semi-axes a, b, c of each ellipsoid, in the order of the volumes below.
static const int morphology_3d_quick_hull_ellipsoids[][3] = { {5, 9, 6}, {4, 9, 4}, {6, 9, 4} };

static const ref_vals_list<double> morphology_3d_mechanics_quick_hull_volume_ref_vals
{
    5800.0 / 6.0,
    2928.0 / 6.0,
    4584.0 / 6.0
};

// Hulls one lattice ellipsoid placed at (ox, oy, oz) and holds the facets to three properties: every
// facet has non-zero area, no input point lies more than eps outside any facet, and the facets enclose
// the pinned volume.
static void assert_quick_hull_encloses_lattice_ellipsoid (int a, int b, int c, int ox, int oy, int oz, double want)
{
    using Points = std::vector<std::array<double, 3>>;
    Points P;
    double maxcoord = 1.0;
    const long long abc2 = (long long)a * a * b * b * c * c;
    for (int x = -a; x <= a; x++)
        for (int y = -b; y <= b; y++)
            for (int z = -c; z <= c; z++)
                if ((long long)x * x * b * b * c * c + (long long)y * y * a * a * c * c + (long long)z * z * a * a * b * b <= abc2)
                {
                    P.push_back ({ double(x + ox), double(y + oy), double(z + oz) });
                    for (double v : P.back())
                        maxcoord = std::max (maxcoord, std::abs(v));
                }
    const double eps = 16.0 * std::numeric_limits<double>::epsilon() * maxcoord;

    std::ostringstream what;
    what << "ellipsoid " << a << "," << b << "," << c << " at " << ox << "," << oy << "," << oz
        << " (" << P.size() << " voxels)";

    quick_hull<Points::const_iterator> qh { 3, eps };
    qh.add_points (std::cbegin(P), std::cend(P));
    auto basis = qh.get_affine_basis();
    ASSERT_EQ (basis.size(), 4u) << what.str() << " spans no volume";
    qh.create_initial_simplex (std::cbegin(basis), std::prev(std::cend(basis)));
    qh.create_convex_hull();

    // Every product below is between integer coordinate differences, so the volume is summed exactly.
    // The facets are oriented outward, so each spans a signed tetrahedron with the reference point
    // P[0] and the signed sum is the enclosed volume.
    const auto& o = P[0];
    double volume6 = 0.;
    size_t flat = 0, outside = 0;
    for (const auto& f : qh.facets_)
    {
        const auto& p = *f.vertices_[0];
        const auto& q = *f.vertices_[1];
        const auto& r = *f.vertices_[2];
        const double pq[3] = { q[0] - p[0], q[1] - p[1], q[2] - p[2] },
            pr[3] = { r[0] - p[0], r[1] - p[1], r[2] - p[2] };
        const double n[3] = { pq[1] * pr[2] - pq[2] * pr[1],
                              pq[2] * pr[0] - pq[0] * pr[2],
                              pq[0] * pr[1] - pq[1] * pr[0] };
        // |n| is twice the facet's area; a facet between lattice points has |n| >= 1 or is flat
        if (n[0] == 0. && n[1] == 0. && n[2] == 0.)
            flat++;
        volume6 += n[0] * (p[0] - o[0]) + n[1] * (p[1] - o[1]) + n[2] * (p[2] - o[2]);
        for (const auto& v : P)
            if (eps < f.distance (std::cbegin(v)))
                outside++;
    }
    const double volume = volume6 / 6.;

    ASSERT_EQ (flat, 0u) << what.str() << ": " << flat << " of " << qh.facets_.size()
        << " facets have zero area";
    ASSERT_EQ (outside, 0u) << what.str() << ": " << outside
        << " point-facet pairs lie more than eps outside the hull";
    ASSERT_TRUE (agrees_gt (volume, want, 1e12)) << what.str() << ": hull volume actual=" << volume
        << " qhull=" << want;
}

// Each ellipsoid is hulled at six placements from the origin out to 2e4, since the facet planes'
// rounding grows with the coordinates. Which coplanar voxel quick_hull meets first follows the
// iteration order of hash sets keyed on point addresses, so one placement exercises one order; the
// eighteen hulls together are what cover the predicates rather than any one of them.
void test_3d_morphology_quick_hull_lattice_ellipsoid_mechanics()
{
    SCOPED_TRACE("MECHANICS__3d_morphology_quick_hull_lattice_ellipsoid");

    const int placements[][3] = { {0, 0, 0}, {20, 30, 10}, {500, 700, 300}, {900, 600, 400},
                                  {3000, 2000, 1000}, {20000, 15000, 9000} };

    ASSERT_EQ (std::size(morphology_3d_quick_hull_ellipsoids), morphology_3d_mechanics_quick_hull_volume_ref_vals.size());
    for (size_t i = 0; i < morphology_3d_mechanics_quick_hull_volume_ref_vals.size(); i++)
    {
        const auto& e = morphology_3d_quick_hull_ellipsoids[i];
        for (const auto& o : placements)
        {
            assert_quick_hull_encloses_lattice_ellipsoid (e[0], e[1], e[2], o[0], o[1], o[2],
                morphology_3d_mechanics_quick_hull_volume_ref_vals[i]);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

// The refusal path of the same kernel: a cloud that spans no volume has no initial simplex.
// get_affine_basis() adds a point only if it lies more than eps off the subspace the basis already
// spans, so a plane stops at three points and a line at two, and build_hull then leaves the hull
// empty. The clouds are integer points, in an axis-aligned plane, a tilted plane and a tilted line,
// far enough from the origin that a point in the subspace sits a rounding error off it rather than
// exactly on it.
void test_3d_morphology_quick_hull_degenerate_basis_mechanics()
{
    SCOPED_TRACE("MECHANICS__3d_morphology_quick_hull_degenerate_basis");

    using Points = std::vector<std::array<double, 3>>;
    struct Cloud { const char* name; Points P; size_t rank; };
    std::vector<Cloud> clouds;

    Cloud flat { "axis-aligned plane", {}, 3 };
    for (int x = 0; x < 6; x++)
        for (int y = 0; y < 4; y++)
            flat.P.push_back ({ double(x + 40), double(y + 50), 9. });
    clouds.push_back (flat);

    Cloud tilted { "tilted plane", {}, 3 };
    for (int x = 0; x < 7; x++)
        for (int y = 0; y < 5; y++)
            tilted.P.push_back ({ double(x + 500), double(y + 300), double(x + 2 * y + 700) });
    clouds.push_back (tilted);

    Cloud line { "tilted line", {}, 2 };
    for (int t = 0; t < 10; t++)
        line.P.push_back ({ double(t + 100), double(2 * t + 200), double(3 * t + 300) });
    clouds.push_back (line);

    for (const auto& c : clouds)
    {
        double maxcoord = 1.0;
        for (const auto& p : c.P)
            for (double v : p)
                maxcoord = std::max (maxcoord, std::abs(v));
        const double eps = 16.0 * std::numeric_limits<double>::epsilon() * maxcoord;

        quick_hull<Points::const_iterator> qh { 3, eps };
        qh.add_points (std::cbegin(c.P), std::cend(c.P));
        ASSERT_EQ (qh.get_affine_basis().size(), c.rank) << c.name << " of " << c.P.size() << " points";
    }
}
