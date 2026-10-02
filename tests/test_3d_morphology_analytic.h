#pragma once

#include <cmath>
#include <vector>
#include "../src/nyx/features/3d_mesh.h"      // Triangle3, LatticeBounds, build_roi_surface_mesh, mesh_volume, roi_mesh_volume
#include "../src/nyx/features/3d_surface.h"   // D3_SurfaceFeature
#include "../src/nyx/features/pixel.h"        // Pixel3
#include "../src/nyx/roi_cache.h"             // LR
#include "test_3d_morphology_common.h"        // gtest, <string>, agrees_gt

// ---------------------------------------------------------------------------------------------------
// Analytic oracle for 3MESH_VOLUME and 3VOLUME_CONVEXHULL. The goldens are closed-form
// geometry rather than another tool's output, so they hold at every precision and on every platform,
// and they establish the features independently of the MIRP rows next door.
//
// Provenance (SPEC 6.4):
//   tool      = analytic (SPEC 4 token); no external reference is involved
//   quantity  = IBSI section 3.1 volume (mesh) -- the volume enclosed by the marching-cubes surface
//               of the mask at the 0.5 isolevel, which for a binary field puts every vertex on the
//               midpoint between an in-ROI and an out-of-ROI voxel; and the volume of the convex
//               hull of the voxel centres
//   recipe   = morphology3d.analytic_lattice_solids
//   fixture   = voxel clouds built in this file; nothing is read from disk
//
// The shapes are chosen so the volume convention is the whole difference between passing and
// failing: counting voxels returns 1 for the single voxel where the octahedron encloses 1/6, and the
// convex hull of a non-convex solid encloses more than its surface does.
// ---------------------------------------------------------------------------------------------------

// Runs D3_SurfaceFeature over a synthetic voxel cloud -- the features themselves, not the mesh
// helpers, so these assertions cover the same code path the phantom fixtures do. single_roi selects
// the whole-volume branch, which takes the ROI to be its bounding box and evaluates the bevelled-box
// closed form instead of meshing the cloud. hull_volume, when given, receives 3VOLUME_CONVEXHULL.
static void calculate_3d_morphology_on_cloud (const std::vector<Pixel3>& cloud,
                                              double& mesh_volume,
                                              bool single_roi = false,
                                              double* hull_volume = nullptr)
{
    ASSERT_FALSE (cloud.empty());

    LR r;
    r.raw_pixels_3D = cloud;
    r.aabb.init_x (cloud[0].x);
    r.aabb.init_y (cloud[0].y);
    r.aabb.init_z (cloud[0].z);
    for (size_t i = 0; i < cloud.size(); i++)
    {
        r.aabb.update_x (cloud[i].x);
        r.aabb.update_y (cloud[i].y);
        r.aabb.update_z (cloud[i].z);
        r.zplanes[(int)cloud[i].z].push_back (i);
    }

    Fsettings s;
    s.resize ((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SINGLEROI].bval = single_roi;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;

    ASSERT_NO_THROW (r.initialize_fvals());
    D3_SurfaceFeature f;
    ASSERT_NO_THROW (f.calculate (r, s));
    f.save_value (r.fvals);

    mesh_volume = r.fvals[(int)Nyxus::Feature3D::MESH_VOLUME][0];
    if (hull_volume)
        *hull_volume = r.fvals[(int)Nyxus::Feature3D::VOLUME_CONVEXHULL][0];
}

// A lone voxel's 0.5-isolevel surface is the octahedron whose six vertices sit half a lattice step
// out along each axis: volume 4/3 * d^3 = 1/6. That is exact in double, so it is pinned at rel=1e-12
// rather than given a band.
void test_3d_morphology_single_voxel_mesh_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_single_voxel_mesh");

    double mesh_volume = 0.;
    calculate_3d_morphology_on_cloud ({ Pixel3(7, 11, 5, 1000) }, mesh_volume);
    if (::testing::Test::HasFatalFailure())
        return;

    ASSERT_TRUE (agrees_gt (mesh_volume, 1.0 / 6.0, 1e12)) << "3MESH_VOLUME actual=" << mesh_volume;
}

// A solid w x h x d box comes out bevelled: each of the 4*(w+h+d-3) cells along an interior edge run,
// and each of the 8 corner cells, cuts a fixed amount off the staircase. The enclosed volume is
// therefore an exact function of w, h and d. Shared by the general-path and whole-volume tests below.
static void assert_3d_morphology_boxes_match_closed_form (bool single_roi)
{
    const int boxes[][3] = { {1,1,1}, {1,4,9}, {2,2,2}, {3,5,2}, {5,7,11}, {8,8,8} };

    for (const auto& b : boxes)
    {
        const int w = b[0], h = b[1], d = b[2];
        std::vector<Pixel3> cloud;
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                for (int z = 0; z < d; z++)
                    cloud.push_back (Pixel3(x + 3, y + 3, z + 3, 1000));

        double mesh_volume = 0.;
        calculate_3d_morphology_on_cloud (cloud, mesh_volume, single_roi);
        if (::testing::Test::HasFatalFailure())
            return;

        const double edgecells = double(w) + double(h) + double(d) - 3.;
        const double want_v = double(w) * h * d - edgecells / 2. - 5. / 6.;

        ASSERT_TRUE (agrees_gt (mesh_volume, want_v, 1e12))
            << w << "x" << h << "x" << d << " 3MESH_VOLUME actual=" << mesh_volume
            << " analytic=" << want_v;
    }
}

// The general path: marching cubes over the box's voxels, integrated triangle by triangle.
void test_3d_morphology_box_mesh_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_box_mesh");
    assert_3d_morphology_boxes_match_closed_form (false);
}

// The whole-volume (SINGLEROI) branch evaluates the bevelled-box closed form from the bounding box
// in place of meshing every voxel. Asserting it against the same closed form, box for box, holds it
// to the value the general path produces above.
void test_3d_morphology_box_mesh_singleroi_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_box_mesh_singleroi");
    assert_3d_morphology_boxes_match_closed_form (true);
}

// A lattice-discretised ball. 3MESH_VOLUME converges on the smooth sphere as the radius grows --
// -3.6% at r=5, -0.15% by r=15 -- so 1% is a real bound there and not slack.
void test_3d_morphology_sphere_mesh_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_sphere_mesh");

    const int radii[] = { 15, 20 };

    for (int r : radii)
    {
        std::vector<Pixel3> cloud;
        for (int x = -r; x <= r; x++)
            for (int y = -r; y <= r; y++)
                for (int z = -r; z <= r; z++)
                    if (x * x + y * y + z * z <= r * r)
                        cloud.push_back (Pixel3(x + 64, y + 64, z + 64, 1000));

        double mesh_volume = 0.;
        calculate_3d_morphology_on_cloud (cloud, mesh_volume);
        if (::testing::Test::HasFatalFailure())
            return;

        const double want_v = 4. / 3. * M_PI * double(r) * r * r;

        ASSERT_LE (std::abs(mesh_volume - want_v) / want_v, 0.01)
            << "r=" << r << " 3MESH_VOLUME actual=" << mesh_volume << " 4/3 pi r^3=" << want_v;
    }
}

// 3VOLUME_CONVEXHULL is the volume of the convex hull of the voxel centres, so on a lattice solid
// whose extreme points are known it is exact: a w x h x d box spans (w-1)(h-1)(d-1), the octahedron
// |x|+|y|+|z| <= R has volume 4/3 R^3, and the rhombic prism |x|+|y| <= R, |z| <= H has 2R^2 * 2H.
// Every hull face carries many exactly coplanar voxels, which is the case a quickhull decides by
// its eps: a coplanar voxel taken for an outside one leaves zero-area facets behind, and the signed
// sum over the facets then misses the hull by whole voxels, by an amount that follows the hash-set
// iteration order and so differs between standard libraries. The solids sit far from the origin
// because that is where the facet plane equation's rounding is largest. The volume is a sum of
// integer determinants over six, so it is pinned at rel=1e-12.
void test_3d_morphology_lattice_hull_volume_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_lattice_hull_volume");

    struct Solid { const char* name; std::vector<Pixel3> cloud; double want; };
    std::vector<Solid> solids;

    const int boxes[][3] = { {2,2,2}, {3,5,2}, {5,7,11}, {8,8,8} };
    for (const auto& b : boxes)
    {
        Solid s { "box", {}, double(b[0] - 1) * (b[1] - 1) * (b[2] - 1) };
        for (int x = 0; x < b[0]; x++)
            for (int y = 0; y < b[1]; y++)
                for (int z = 0; z < b[2]; z++)
                    s.cloud.push_back (Pixel3(x + 500, y + 700, z + 300, 1000));
        solids.push_back (s);
    }

    for (int R : { 3, 8, 15 })
    {
        Solid s { "octahedron", {}, 4. / 3. * double(R) * R * R };
        for (int x = -R; x <= R; x++)
            for (int y = -R; y <= R; y++)
                for (int z = -R; z <= R; z++)
                    if (std::abs(x) + std::abs(y) + std::abs(z) <= R)
                        s.cloud.push_back (Pixel3(x + 900, y + 600, z + 400, 1000));
        solids.push_back (s);
    }

    for (int R : { 4, 12 })
    {
        const int H = R / 2 + 1;
        Solid s { "rhombic prism", {}, 4. * double(R) * R * H };
        for (int x = -R; x <= R; x++)
            for (int y = -R; y <= R; y++)
                for (int z = -H; z <= H; z++)
                    if (std::abs(x) + std::abs(y) <= R)
                        s.cloud.push_back (Pixel3(x + 800, y + 800, z + 800, 1000));
        solids.push_back (s);
    }

    for (const auto& s : solids)
    {
        double mesh_volume = 0., hull_volume = 0.;
        calculate_3d_morphology_on_cloud (s.cloud, mesh_volume, false, &hull_volume);
        if (::testing::Test::HasFatalFailure())
            return;

        ASSERT_TRUE (agrees_gt (hull_volume, s.want, 1e12)) << "3VOLUME_CONVEXHULL actual=" << hull_volume
            << " analytic=" << s.want << " for a " << s.name << " of " << s.cloud.size() << " voxels";
    }
}

// The refusal path for the hull: voxels in a single plane span no volume, so there is no initial
// simplex. The hull is left empty and 3VOLUME_CONVEXHULL comes back 0 rather than a volume assembled
// from facets of a degenerate simplex.
void test_3d_morphology_planar_hull_volume_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_planar_hull_volume");

    std::vector<Pixel3> cloud;
    for (int x = 0; x < 6; x++)
        for (int y = 0; y < 4; y++)
            cloud.push_back (Pixel3(x + 40, y + 50, 9, 1000));

    double mesh_volume = 0., hull_volume = -1.;
    calculate_3d_morphology_on_cloud (cloud, mesh_volume, false, &hull_volume);
    if (::testing::Test::HasFatalFailure())
        return;

    ASSERT_EQ (hull_volume, 0.0) << "3VOLUME_CONVEXHULL of a single-plane ROI";
}

// The refusal path: an ROI with no voxels has no surface. The builder must hand back an empty mesh,
// and both volume integrals a zero, rather than reading off the end of an empty cloud; bounds that
// enclose no lattice point give a zero without asking the plane source for a plane.
void test_3d_morphology_empty_roi_mesh_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_empty_roi_mesh");

    std::vector<Nyxus::Triangle3> mesh;
    ASSERT_NO_THROW (Nyxus::build_roi_surface_mesh (mesh, std::vector<Pixel3>{}));
    ASSERT_TRUE (mesh.empty());
    ASSERT_EQ (Nyxus::mesh_volume (mesh), 0.0);
    ASSERT_EQ (Nyxus::roi_mesh_volume (std::vector<Pixel3>{}), 0.0);

    int planes_asked = 0;
    const Nyxus::LatticeBounds none = { 5, 4, 0, 3, 0, 3 };
    ASSERT_EQ (Nyxus::roi_mesh_volume (none, [&planes_asked](StatsInt, std::vector<Pixel3>&) { planes_asked++; }), 0.0);
    ASSERT_EQ (planes_asked, 0);
}
