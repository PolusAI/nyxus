#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/features/edt.h"
#include "../src/nyx/features/roi_radius.h"
#include "test_main_nyxus.h"   // roi_cache.h / featureset.h / pixel.h

// The two ways RoiRadiusFeature measures a pixel's distance to the contour must be the same
// measurement (SPEC 2 invariant tier -- a required relation between two implementations, not an
// oracle claim).
//
//   in-RAM, compact ROI   Nyxus::exact_sqedt over the bounding box, then a lookup per pixel
//   in-RAM, sparse ROI    Pixel2::exact_min_sqdist, the exhaustive scan
//   out-of-core           Pixel2::exact_min_sqdist, the same exhaustive scan
//
// RoiRadiusFeature::edt_is_cheaper decides which of the first two runs. Both are exact and both
// work in integer squared distances, so the choice must not be observable in any reported value --
// which is what makes the branch safe to take on a cost estimate. These assertions are what that
// claim rests on, and they are exact-equality assertions rather than tolerance ones: a transform
// that agreed only to within a tolerance would not be exact, and the claim would be false.
//
// The contours here are built by the test, not by ContourFeature, so a defect in contour tracing
// cannot mask a disagreement between the two distance engines -- it would move both sides equally.

namespace nyxus_ut_edt
{
    struct Shape
    {
        std::string name;
        std::vector<Pixel2> cloud;     // every ROI pixel
        std::vector<Pixel2> contour;   // 4-neighbour inner boundary of 'cloud'
    };

    // Turn a mask into a ROI plus its 4-connected inner boundary. A pixel is on the boundary when
    // any of its four neighbours is outside the mask, the image edge counting as outside.
    inline Shape from_mask (const std::string& name, const std::vector<char>& m, int W, int H,
                            long ox = 0, long oy = 0)
    {
        Shape s;
        s.name = name;
        auto at = [&](int x, int y) { return (x < 0 || y < 0 || x >= W || y >= H) ? char(0) : m[(size_t)y * W + x]; };
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                if (! at(x, y))
                    continue;
                Pixel2 p ((long)(ox + x), (long)(oy + y), (PixIntens)1);
                s.cloud.push_back (p);
                if (! at(x - 1, y) || ! at(x + 1, y) || ! at(x, y - 1) || ! at(x, y + 1))
                    s.contour.push_back (p);
            }
        return s;
    }

    inline Shape disk (int R, long ox = 100, long oy = 100)
    {
        int S = 2 * R + 1;
        std::vector<char> m ((size_t)S * S, 0);
        for (int y = -R; y <= R; y++)
            for (int x = -R; x <= R; x++)
                if (x * x + y * y <= R * R)
                    m[(size_t)(y + R) * S + (x + R)] = 1;
        return from_mask ("disk R=" + std::to_string(R), m, S, S, ox, oy);
    }

    // One pixel per row and column: the sparsest ROI a bounding box can hold, and the case
    // edt_is_cheaper exists to send to the scan.
    inline Shape diagonal (int L)
    {
        std::vector<char> m ((size_t)L * L, 0);
        for (int i = 0; i < L; i++)
            m[(size_t)i * L + i] = 1;
        return from_mask ("diagonal L=" + std::to_string(L), m, L, L, 100, 100);
    }

    // A large box holding few pixels, most of which are boundary -- sparse like the diagonal but
    // with a long contour, which is the combination a fill-ratio test would misjudge.
    inline Shape ring (int R, int w)
    {
        int S = 2 * R + 1, inner = R - w;
        std::vector<char> m ((size_t)S * S, 0);
        for (int y = -R; y <= R; y++)
            for (int x = -R; x <= R; x++)
            {
                int d2 = x * x + y * y;
                if (d2 <= R * R && d2 > inner * inner)
                    m[(size_t)(y + R) * S + (x + R)] = 1;
            }
        return from_mask ("ring R=" + std::to_string(R) + " w=" + std::to_string(w), m, S, S, 100, 100);
    }

    // A square with a square hole: two contours, so merge_multicontour's flattening is exercised
    // and the transform is seeded from a non-simply-connected site set.
    inline Shape square_with_hole (int S, int hole)
    {
        std::vector<char> m ((size_t)S * S, 1);
        int lo = (S - hole) / 2, hi = lo + hole;
        for (int y = lo; y < hi; y++)
            for (int x = lo; x < hi; x++)
                m[(size_t)y * S + x] = 0;
        return from_mask ("square " + std::to_string(S) + " hole " + std::to_string(hole), m, S, S, 100, 100);
    }

    inline Shape ell (int S, int arm)
    {
        std::vector<char> m ((size_t)S * S, 0);
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++)
                if (x < arm || y >= S - arm)
                    m[(size_t)y * S + x] = 1;
        return from_mask ("L " + std::to_string(S) + "/" + std::to_string(arm), m, S, S, 100, 100);
    }

    // Overlapping lobes: a contour that is neither convex nor locally unimodal, which is the family
    // the superseded hill-descent mis-handled and therefore the family worth covering.
    inline Shape blob (int S, int lobes, unsigned seed)
    {
        std::vector<char> m ((size_t)S * S, 0);
        unsigned st = seed * 2654435761u + 1u;
        auto rnd = [&st](int lo, int hi) { st = st * 1664525u + 1013904223u; return lo + (int)((st >> 16) % (unsigned)(hi - lo + 1)); };
        for (int i = 0; i < lobes; i++)
        {
            int cx = rnd(0, S - 1), cy = rnd(0, S - 1), rr = rnd(S / 12 + 1, S / 5 + 1);
            for (int y = std::max(0, cy - rr); y < std::min(S, cy + rr + 1); y++)
                for (int x = std::max(0, cx - rr); x < std::min(S, cx + rr + 1); x++)
                    if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr * rr)
                        m[(size_t)y * S + x] = 1;
        }
        return from_mask ("blob seed=" + std::to_string(seed), m, S, S, 100, 100);
    }

    // The raster both engines are compared over: the ROI's pixels and its contour together, since a
    // traced contour need not lie inside the ROI's own bounding box.
    inline void bounds (const Shape& s, StatsInt& xmin, StatsInt& ymin, size_t& W, size_t& H)
    {
        StatsInt xmax, ymax;
        xmin = xmax = s.cloud[0].x;
        ymin = ymax = s.cloud[0].y;
        for (const auto* v : { &s.cloud, &s.contour })
            for (const auto& p : *v)
            {
                xmin = std::min (xmin, p.x); xmax = std::max (xmax, p.x);
                ymin = std::min (ymin, p.y); ymax = std::max (ymax, p.y);
            }
        W = (size_t)(xmax - xmin) + 1;
        H = (size_t)(ymax - ymin) + 1;
    }

    inline void stats (std::vector<double> d, double& mean, double& mx, double& med)
    {
        if (d.empty()) { mean = mx = med = 0.0; return; }
        double sum = 0; mx = d[0];
        for (double v : d) { sum += v; if (v > mx) mx = v; }
        mean = sum / d.size();
        std::sort (d.begin(), d.end());
        size_t n = d.size();
        med = n % 2 ? d[n / 2] : (d[n / 2 - 1] + d[n / 2]) / 2.0;
    }

    inline std::vector<Shape> battery()
    {
        std::vector<Shape> v;
        for (int R : {1, 2, 3, 5, 10, 20, 40})
            v.push_back (disk (R));
        v.push_back (diagonal (40));
        v.push_back (ring (30, 3));
        v.push_back (ring (30, 12));
        v.push_back (square_with_hole (40, 12));
        v.push_back (ell (40, 9));
        for (unsigned s = 1; s <= 8; s++)
            v.push_back (blob (48, 5, s));
        return v;
    }
}

// Every ROI pixel's squared distance to the contour is the same integer either way.
inline void test_2d_morphology_edt_equals_exhaustive_scan_invariant()
{
    for (const auto& s : nyxus_ut_edt::battery())
    {
        SCOPED_TRACE (s.name);
        ASSERT_FALSE (s.cloud.empty());
        ASSERT_FALSE (s.contour.empty());

        StatsInt xmin, ymin; size_t W, H;
        nyxus_ut_edt::bounds (s, xmin, ymin, W, H);

        std::vector<int64_t> sq;
        Nyxus::exact_sqedt (s.contour, xmin, ymin, W, H, sq);

        for (const auto& p : s.cloud)
        {
            const int64_t viaEdt = sq[(size_t)(p.y - ymin) * W + (size_t)(p.x - xmin)];
            const double viaScan = p.exact_min_sqdist (s.contour);

            // Integer squared distances on both sides, so this is exact equality by construction.
            // An implementation returning a nearest-but-one site fails it; so does one whose
            // raster indexing is off by a row.
            ASSERT_EQ ((double)viaEdt, viaScan)
                << "pixel (" << p.x << "," << p.y << "): transform " << viaEdt
                << " vs exhaustive scan " << viaScan;
        }
    }
}

// The three reported statistics are bit-identical, which is the property that lets the branch be
// chosen on a cost estimate without the choice being observable.
inline void test_2d_morphology_roi_radius_engines_agree_invariant()
{
    for (const auto& s : nyxus_ut_edt::battery())
    {
        SCOPED_TRACE (s.name);

        StatsInt xmin, ymin; size_t W, H;
        nyxus_ut_edt::bounds (s, xmin, ymin, W, H);

        std::vector<int64_t> sq;
        Nyxus::exact_sqedt (s.contour, xmin, ymin, W, H, sq);

        std::vector<double> dEdt, dScan;
        dEdt.reserve (s.cloud.size());
        dScan.reserve (s.cloud.size());
        for (const auto& p : s.cloud)
        {
            dEdt.push_back (std::sqrt ((double) sq[(size_t)(p.y - ymin) * W + (size_t)(p.x - xmin)]));
            dScan.push_back (std::sqrt (p.exact_min_sqdist (s.contour)));
        }

        double m1, x1, d1, m2, x2, d2;
        nyxus_ut_edt::stats (dEdt, m1, x1, d1);
        nyxus_ut_edt::stats (dScan, m2, x2, d2);

        ASSERT_EQ (m1, m2) << "ROI_RADIUS_MEAN differs between the engines";
        ASSERT_EQ (x1, x2) << "ROI_RADIUS_MAX differs between the engines";
        ASSERT_EQ (d1, d2) << "ROI_RADIUS_MEDIAN differs between the engines";
    }
}

// The cost estimate that picks between them. Its inputs are three counts known before any distance
// is computed -- the bounding box area, the pixel count and the contour length -- so the decision
// costs nothing to reach. What is asserted here is which way it goes on the shapes whose measured
// behaviour motivates it, so a future change to the constants has to face those measurements.
inline void test_2d_morphology_roi_radius_engine_choice_invariant()
{
    // Compact ROIs: the box is mostly ROI and the contour is long, so the transform wins -- measured
    // at 12.7x on a disk of R=40 and 96x at R=400.
    for (int R : {10, 20, 40, 100, 200, 400})
    {
        const size_t side = (size_t)(2 * R + 1), cells = side * side;
        size_t pixels = 0, contour = 0;
        for (int y = -R; y <= R; y++)
            for (int x = -R; x <= R; x++)
                if (x * x + y * y <= R * R)
                {
                    pixels++;
                    auto in = [R](int a, int b) { return a * a + b * b <= R * R; };
                    if (! in(x - 1, y) || ! in(x + 1, y) || ! in(x, y - 1) || ! in(x, y + 1))
                        contour++;
                }
        EXPECT_TRUE (RoiRadiusFeature::edt_is_cheaper (cells, pixels, contour))
            << "disk R=" << R << ": " << cells << " cells, " << pixels << " pixels, "
            << contour << " contour -- the transform should be chosen here";
    }

    // A one-pixel-wide diagonal: L pixels in L*L cells with only L contour pixels, so the scan is
    // cheap and the transform is not. Measured 7.9x slower at L=700.
    for (size_t L : {200, 700, 2000})
        EXPECT_FALSE (RoiRadiusFeature::edt_is_cheaper (L * L, L, L))
            << "diagonal L=" << L << ": the exhaustive scan should be chosen here";

    // A thin ring is nearly as sparse as the diagonal by fill ratio (1.2% against 0.14%) and the
    // transform still wins on it, measured 4.9x -- because its contour is long. This is the case a
    // fill-ratio test would get wrong, and the reason the estimate uses pixels*contour instead.
    EXPECT_TRUE (RoiRadiusFeature::edt_is_cheaper (641601, 7524, 4508))
        << "thin ring: the transform should be chosen despite the low fill ratio";

    // Past the raster ceiling the transform is refused whatever the shape, because the buffer is
    // held per ROI being measured and so once per worker thread.
    EXPECT_FALSE (RoiRadiusFeature::edt_is_cheaper ((size_t)4096 * 4096, 10000000, 100000))
        << "a bounding box past the ceiling must fall back to the scan";
}

// The in-RAM and out-of-core paths report the same statistics of the same ROI.
//
// SPEC 2 invariant tier. CLAUDE.md requires the two paths to produce identical values; this asserts
// it for ROI_RADIUS_* over the shape battery above, with the in-RAM path free to take either of its
// two engines and the out-of-core path always taking the exhaustive scan. On a compact shape that
// makes it a distance transform against a scan across a path boundary, which is the widest gap the
// three implementations have to close.
//
// WHAT THIS DOES NOT COVER, deliberately. Both paths are handed the SAME contour, set on the ROI
// directly rather than traced. The two contour builders do not agree with each other -- same count
// and different pixels below the tile size (todo item 65), and a contour truncated to the first tile
// above it (item 68) -- and those are defects of contour construction, pinned by their own tests. If
// this test traced instead of supplying, it would fail for reasons that have nothing to do with the
// distance engines and would stop discriminating between them. When 65 and 68 are fixed, the
// stronger end-to-end equality belongs in tests/python/test_2d_ooc_invariant.py on a disk fixture.
inline void test_2d_morphology_roi_radius_paths_agree_invariant()
{
    Fsettings s;
    size_t n_took_transform = 0;

    for (const auto& shape : nyxus_ut_edt::battery())
    {
        SCOPED_TRACE (shape.name);

        // Record which engine the in-RAM path will use, so the comparison below is known to be a
        // cross-engine one on at least some shapes rather than the scan against itself.
        StatsInt xmin, ymin; size_t W, H;
        nyxus_ut_edt::bounds (shape, xmin, ymin, W, H);
        const bool viaTransform = RoiRadiusFeature::edt_is_cheaper (W * H, shape.cloud.size(), shape.contour.size());
        if (viaTransform)
            n_took_transform++;

        // ---- in-RAM
        double ramMean, ramMax, ramMedian;
        {
            LR roi (1);
            roi.initialize_fvals();
            roi.raw_pixels = shape.cloud;
            roi.multicontour_.push_back (shape.contour);

            RoiRadiusFeature f;
            f.calculate (roi, s);
            f.save_value (roi.fvals);
            ramMean = roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEAN][0];
            ramMax = roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MAX][0];
            ramMedian = roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEDIAN][0];
        }

        // ---- out-of-core, same pixels and the same contour, streamed from a disk-backed cloud
        double oocMean, oocMax, oocMedian;
        {
            LR roi (1);
            roi.initialize_fvals();
            roi.multicontour_.push_back (shape.contour);

            roi.raw_pixels_NT.init (1, "nyxus_ut_paths_agree");
            for (const auto& p : shape.cloud)
                roi.raw_pixels_NT.add_pixel (p);

            ImageLoader dummy;   // osized_calculate reads its pixels from the cloud, not the loader
            RoiRadiusFeature f;
            f.osized_calculate (roi, s, dummy);
            f.save_value (roi.fvals);
            oocMean = roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEAN][0];
            oocMax = roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MAX][0];
            oocMedian = roi.fvals[(int)Nyxus::Feature2D::ROI_RADIUS_MEDIAN][0];

            roi.raw_pixels_NT.close();
        }

        // Exact equality, not a tolerance: both engines work in integer squared distances and take
        // the square root once, so any difference at all means they are not the same measurement.
        ASSERT_EQ (ramMean, oocMean)
            << "ROI_RADIUS_MEAN differs across the path boundary (in-RAM took the "
            << (viaTransform ? "transform" : "scan") << ")";
        ASSERT_EQ (ramMax, oocMax)
            << "ROI_RADIUS_MAX differs across the path boundary (in-RAM took the "
            << (viaTransform ? "transform" : "scan") << ")";
        ASSERT_EQ (ramMedian, oocMedian)
            << "ROI_RADIUS_MEDIAN differs across the path boundary (in-RAM took the "
            << (viaTransform ? "transform" : "scan") << ")";
    }

    // Without this the battery could drift -- or the guard's constants could change -- until every
    // shape takes the scan on both sides, at which point the assertions above still pass while
    // comparing the scan with itself and proving nothing about the transform.
    ASSERT_GT (n_took_transform, (size_t)0)
        << "no shape in the battery routed the in-RAM path through the distance transform, so this "
           "test no longer compares the two engines";
}
