#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/features/edt.h"
#include "../src/nyx/features/roi_radius.h"
#include "test_main_nyxus.h"   // roi_cache.h / featureset.h / pixel.h

// Degenerate and out-of-range inputs to the two distance engines (SPEC 2 mechanics tier -- plumbing,
// no oracle claim).
//
// The transform indexes a raster it allocates from ROI geometry, so the inputs that matter are the
// ones where that geometry is empty, is a single cell, or does not contain the sites. The exhaustive
// scan reaches the same cases through a different path and must answer the same way. Every
// assertion below states a defined outcome rather than merely "does not crash": a routine that
// returned a plausible-looking number for a ROI with no contour would pass a no-crash test and be
// wrong.

// A raster with no site inside it resolves to no distance at all, and says so rather than
// inventing one. -1 is the marked value; a finite number here would be a fabricated measurement.
inline void test_2d_edt_no_sites_mechanics()
{
    std::vector<int64_t> sq;

    // No sites at all.
    Nyxus::exact_sqedt ({}, 0, 0, 4, 3, sq);
    ASSERT_EQ ((size_t)12, sq.size());
    for (size_t i = 0; i < sq.size(); i++)
        ASSERT_EQ (-1, sq[i]) << "cell " << i << " of a site-free raster";

    // Sites exist but all fall outside the raster, which is the shape of the bug a raster sized to
    // the pixel cloud alone would have: a contour traced on a padded image can sit outside it.
    std::vector<Pixel2> outside = { Pixel2(100L, 100L, (PixIntens)1), Pixel2(-5L, 7L, (PixIntens)1) };
    Nyxus::exact_sqedt (outside, 0, 0, 4, 3, sq);
    ASSERT_EQ ((size_t)12, sq.size());
    for (size_t i = 0; i < sq.size(); i++)
        ASSERT_EQ (-1, sq[i]) << "cell " << i << " with every site outside the raster";
}

// A raster with no cells produces no cells, and does not index anything to find that out.
inline void test_2d_edt_empty_raster_mechanics()
{
    std::vector<Pixel2> sites = { Pixel2(0L, 0L, (PixIntens)1) };
    std::vector<int64_t> sq (7, 12345);   // pre-filled, so a no-op would be visible

    Nyxus::exact_sqedt (sites, 0, 0, 0, 5, sq);
    ASSERT_TRUE (sq.empty()) << "a zero-width raster should yield no cells";

    Nyxus::exact_sqedt (sites, 0, 0, 5, 0, sq);
    ASSERT_TRUE (sq.empty()) << "a zero-height raster should yield no cells";
}

// The smallest raster there is, holding its own site.
inline void test_2d_edt_single_cell_mechanics()
{
    std::vector<Pixel2> sites = { Pixel2(9L, 4L, (PixIntens)1) };
    std::vector<int64_t> sq;
    Nyxus::exact_sqedt (sites, 9, 4, 1, 1, sq);
    ASSERT_EQ ((size_t)1, sq.size());
    ASSERT_EQ (0, sq[0]) << "a cell holding a site is at distance zero from itself";
}

// One site in an open raster: every cell's answer is the squared Euclidean distance to it, so this
// pins the transform against arithmetic rather than against the scan.
inline void test_2d_edt_single_site_mechanics()
{
    const long sx = 3, sy = 2;
    std::vector<Pixel2> sites = { Pixel2(sx, sy, (PixIntens)1) };
    std::vector<int64_t> sq;
    const size_t W = 7, H = 5;
    Nyxus::exact_sqedt (sites, 0, 0, W, H, sq);

    ASSERT_EQ (W * H, sq.size());
    for (size_t y = 0; y < H; y++)
        for (size_t x = 0; x < W; x++)
        {
            const int64_t dx = (int64_t)x - sx, dy = (int64_t)y - sy;
            ASSERT_EQ (dx * dx + dy * dy, sq[y * W + x])
                << "cell (" << x << "," << y << ") against the one site at (" << sx << "," << sy << ")";
        }
}

// Sites on the raster's edges and corners, where a row or column pass has no neighbour on one side.
inline void test_2d_edt_boundary_sites_mechanics()
{
    const size_t W = 6, H = 6;
    std::vector<Pixel2> corners = {
        Pixel2(0L, 0L, (PixIntens)1), Pixel2((long)W - 1, 0L, (PixIntens)1),
        Pixel2(0L, (long)H - 1, (PixIntens)1), Pixel2((long)W - 1, (long)H - 1, (PixIntens)1)
    };
    std::vector<int64_t> sq;
    Nyxus::exact_sqedt (corners, 0, 0, W, H, sq);

    ASSERT_EQ (W * H, sq.size());
    for (size_t y = 0; y < H; y++)
        for (size_t x = 0; x < W; x++)
        {
            int64_t best = -1;
            for (const auto& s : corners)
            {
                const int64_t dx = (int64_t)x - s.x, dy = (int64_t)y - s.y, d = dx * dx + dy * dy;
                if (best < 0 || d < best)
                    best = d;
            }
            ASSERT_EQ (best, sq[y * W + x]) << "cell (" << x << "," << y << ")";
        }
}

// A ROI with no pixels, and a ROI whose contour never got built. Both report zeros rather than
// reading an empty buffer, on the in-RAM path and the out-of-core path alike.
inline void test_2d_roi_radius_empty_inputs_mechanics()
{
    Fsettings s;

    {   // no pixels, no contour
        LR roi(1);
        roi.initialize_fvals();
        RoiRadiusFeature f;
        ASSERT_NO_THROW (f.calculate (roi, s));
        f.save_value (roi.fvals);
        ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEAN][0]);
        ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MAX][0]);
        ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEDIAN][0]);
    }

    {   // pixels but no contour: nothing to measure a distance to
        LR roi(1);
        roi.initialize_fvals();
        for (long y = 0; y < 4; y++)
            for (long x = 0; x < 4; x++)
                roi.raw_pixels.push_back (Pixel2(x, y, (PixIntens)1));
        roi.make_nonanisotropic_aabb();
        RoiRadiusFeature f;
        ASSERT_NO_THROW (f.calculate (roi, s));
        f.save_value (roi.fvals);
        ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEAN][0]);
        ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MAX][0]);
        ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEDIAN][0]);
    }
}

// A one-pixel ROI is its own contour, so every statistic is zero -- the smallest case where the
// raster is a single cell and the scan compares a pixel with itself.
inline void test_2d_roi_radius_single_pixel_mechanics()
{
    Fsettings s;
    LR roi(1);
    roi.initialize_fvals();
    const Pixel2 p (17L, 23L, (PixIntens)1);
    roi.raw_pixels.push_back (p);
    roi.multicontour_.push_back ({ p });
    roi.make_nonanisotropic_aabb();

    RoiRadiusFeature f;
    ASSERT_NO_THROW (f.calculate (roi, s));
    f.save_value (roi.fvals);
    ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEAN][0]);
    ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MAX][0]);
    ASSERT_EQ (0.0, roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEDIAN][0]);
}

// A contour lying outside the ROI's own bounding box. Contours are traced on a padded image and come
// back offset from the pixel cloud, so the raster is sized to hold both; sizing it to the cloud
// alone would index past the buffer here.
inline void test_2d_roi_radius_contour_outside_bbox_mechanics()
{
    Fsettings s;
    LR roi(1);
    roi.initialize_fvals();

    std::vector<Pixel2> contour;
    for (long y = 0; y < 5; y++)
        for (long x = 0; x < 5; x++)
        {
            roi.raw_pixels.push_back (Pixel2(x, y, (PixIntens)1));
            if (x == 0 || y == 0 || x == 4 || y == 4)
                contour.push_back (Pixel2(x + 1, y + 1, (PixIntens)1));   // the (+1,+1) offset
        }
    roi.multicontour_.push_back (contour);
    roi.make_nonanisotropic_aabb();

    RoiRadiusFeature f;
    ASSERT_NO_THROW (f.calculate (roi, s));
    f.save_value (roi.fvals);

    // Every distance is finite and no larger than the diagonal of the union of the two extents.
    const double bound = std::sqrt (6.0 * 6.0 + 6.0 * 6.0);
    const double mx = roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MAX][0];
    ASSERT_GE (mx, 0.0);
    ASSERT_LE (mx, bound) << "a distance past the union's diagonal means the raster was mis-sized";
}
