#pragma once

#include <cmath>
#include <vector>
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
//               hull of that surface's vertices
//   recipe   = morphology3d.analytic_lattice_solids
//   fixture   = voxel clouds built in this file; nothing is read from disk
//
// Every golden here is exact and pinned at rel=1e-12. The shapes are chosen so the volume convention
// is the whole difference between passing and failing: counting voxels returns 1 for the single voxel
// where the octahedron encloses 1/6, and the hull of the voxel centres falls short of the hull of the
// mesh by half a lattice step on every face.
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

// A solid w x h x d box comes out bevelled: each of the 4*(w+h+d-3) cells along an interior edge run,
// and each of the 8 corner cells, cuts a fixed amount off the staircase. The enclosed volume is
// therefore an exact function of w, h and d, and since the bevelled box is convex it is also the
// volume of the hull of its vertices.
static double morphology_3d_bevelled_box_volume (int w, int h, int d)
{
    const double edgecells = double(w) + double(h) + double(d) - 3.;
    return double(w) * h * d - edgecells / 2. - 5. / 6.;
}

static std::vector<Pixel3> morphology_3d_box_cloud (int w, int h, int d, int x0, int y0, int z0)
{
    std::vector<Pixel3> cloud;
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
            for (int z = 0; z < d; z++)
                cloud.push_back (Pixel3(x + x0, y + y0, z + z0, 1000));
    return cloud;
}

// A lone voxel's 0.5-isolevel surface is the octahedron whose six vertices sit half a lattice step
// out along each axis: volume 4/3 * d^3 = 1/6, and the octahedron is its own hull.
void test_3d_morphology_single_voxel_mesh_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_single_voxel_mesh");

    double mesh_volume = 0., hull_volume = 0.;
    calculate_3d_morphology_on_cloud ({ Pixel3(7, 11, 5, 1000) }, mesh_volume, false, &hull_volume);
    if (::testing::Test::HasFatalFailure())
        return;

    ASSERT_TRUE (agrees_gt (mesh_volume, 1.0 / 6.0, 1e12)) << "3MESH_VOLUME actual=" << mesh_volume;
    ASSERT_TRUE (agrees_gt (hull_volume, 1.0 / 6.0, 1e12)) << "3VOLUME_CONVEXHULL actual=" << hull_volume;
}

// Shared by the general-path and whole-volume tests below: the bevelled-box closed form for both the
// mesh volume and the hull volume.
static void assert_3d_morphology_boxes_match_closed_form (bool single_roi)
{
    const int boxes[][3] = { {1,1,1}, {1,4,9}, {2,2,2}, {3,5,2}, {4,5,6}, {5,7,11}, {8,8,8} };

    for (const auto& b : boxes)
    {
        const int w = b[0], h = b[1], d = b[2];

        double mesh_volume = 0., hull_volume = 0.;
        calculate_3d_morphology_on_cloud (morphology_3d_box_cloud (w, h, d, 3, 3, 3), mesh_volume, single_roi, &hull_volume);
        if (::testing::Test::HasFatalFailure())
            return;

        const double want = morphology_3d_bevelled_box_volume (w, h, d);
        ASSERT_TRUE (agrees_gt (mesh_volume, want, 1e12))
            << w << "x" << h << "x" << d << " 3MESH_VOLUME actual=" << mesh_volume << " analytic=" << want;
        ASSERT_TRUE (agrees_gt (hull_volume, want, 1e12))
            << w << "x" << h << "x" << d << " 3VOLUME_CONVEXHULL actual=" << hull_volume << " analytic=" << want;
    }
}

// The general path: marching cubes over the box's voxels, integrated triangle by triangle, and hulled.
void test_3d_morphology_box_mesh_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_box_mesh");
    assert_3d_morphology_boxes_match_closed_form (false);
}

// The whole-volume (SINGLEROI) branch evaluates the bevelled-box closed form from the bounding box
// in place of meshing every voxel. Asserting it against the same closed form, box for box, holds it
// to the values the general path produces above.
void test_3d_morphology_box_mesh_singleroi_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_box_mesh_singleroi");
    assert_3d_morphology_boxes_match_closed_form (true);
}

// 3VOLUME_CONVEXHULL is the volume of the convex hull of the mesh vertices. Every vertex is an
// in-ROI voxel moved half a step towards an out-of-ROI neighbour, so on a lattice solid whose extreme
// voxels are known the hull is exact: a w x h x d box gives the bevelled box above, the octahedron
// |x|+|y|+|z| <= R gives the octahedron of radius R + 1/2, 4/3 (R + 1/2)^3, and the rhombic prism
// |x|+|y| <= R, |z| <= H gives a prism of rhombus radius R + 1/2 and height 2H capped at each end by a
// frustum of height 1/2 narrowing to radius R. Every hull face carries many exactly coplanar vertices,
// which is the case a quickhull decides by its eps: a coplanar point taken for an outside one leaves
// zero-area facets behind, and the signed sum over the facets then misses the hull, by an amount that
// follows the hash-set iteration order and so differs between standard libraries. The solids sit far
// from the origin because that is where the facet plane equation's rounding is largest.
void test_3d_morphology_lattice_hull_volume_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_lattice_hull_volume");

    struct Solid { const char* name; std::vector<Pixel3> cloud; double want; };
    std::vector<Solid> solids;

    const int boxes[][3] = { {2,2,2}, {3,5,2}, {5,7,11}, {8,8,8} };
    for (const auto& b : boxes)
        solids.push_back ({ "box", morphology_3d_box_cloud (b[0], b[1], b[2], 500, 700, 300),
            morphology_3d_bevelled_box_volume (b[0], b[1], b[2]) });

    for (int R : { 3, 8, 15 })
    {
        const double r = R + 0.5;
        Solid s { "octahedron", {}, 4. / 3. * r * r * r };
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
        const double outer = 2. * (R + 0.5) * (R + 0.5),	// rhombus areas at radius R + 1/2 and R
            inner = 2. * double(R) * R;
        Solid s { "rhombic prism", {}, outer * 2. * H + 2. * (0.5 / 3.) * (outer + inner + std::sqrt(outer * inner)) };
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

// An ROI one slice thick still has a surface with thickness -- its vertices sit half a step above and
// below the slice -- so it has a hull volume, the bevelled 6 x 4 x 1 slab.
void test_3d_morphology_planar_hull_volume_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_planar_hull_volume");

    double mesh_volume = 0., hull_volume = -1.;
    calculate_3d_morphology_on_cloud (morphology_3d_box_cloud (6, 4, 1, 40, 50, 9), mesh_volume, false, &hull_volume);
    if (::testing::Test::HasFatalFailure())
        return;

    const double want = morphology_3d_bevelled_box_volume (6, 4, 1);
    ASSERT_TRUE (agrees_gt (hull_volume, want, 1e12)) << "3VOLUME_CONVEXHULL of a single-slice ROI actual="
        << hull_volume << " analytic=" << want;
}

// The mask decides the surface, not the intensities. A 7 x 7 x 5 box whose voxels are all 0, and one
// with two of its faces at 0, give the same bevelled box as a uniform one. An all-zero ROI is also the
// refusal path a point set too small to hull takes: it must come back as a value, not a crash.
void test_3d_morphology_zero_intensity_hull_analytic()
{
    SCOPED_TRACE("ANALYTIC_ORACLE__3d_morphology_zero_intensity_hull");

    const double want = morphology_3d_bevelled_box_volume (7, 7, 5);

    auto all_zero = morphology_3d_box_cloud (7, 7, 5, 20, 30, 40);
    for (auto& v : all_zero)
        v.inten = 0;

    auto two_faces_zero = morphology_3d_box_cloud (7, 7, 5, 20, 30, 40);
    for (auto& v : two_faces_zero)
        if (v.x == 20 || v.z == 44)
            v.inten = 0;

    for (const auto* cloud : { &all_zero, &two_faces_zero })
    {
        double mesh_volume = 0., hull_volume = 0.;
        calculate_3d_morphology_on_cloud (*cloud, mesh_volume, false, &hull_volume);
        if (::testing::Test::HasFatalFailure())
            return;

        const char* which = cloud == &all_zero ? "all voxels at 0" : "two faces at 0";
        ASSERT_TRUE (agrees_gt (mesh_volume, want, 1e12)) << which << ": 3MESH_VOLUME actual=" << mesh_volume;
        ASSERT_TRUE (agrees_gt (hull_volume, want, 1e12)) << which << ": 3VOLUME_CONVEXHULL actual=" << hull_volume;
    }
}
