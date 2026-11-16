#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../src/nyx/feature_settings.h"   // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"         // Feature3D
#include "../src/nyx/features/3d_glcm.h"   // D3_GLCM_feature
#include "../src/nyx/roi_cache.h"          // LR

// 3D GLCM on an asymmetric co-occurrence matrix against the IBSI definitions evaluated in the test,
// the 3D counterpart of test_2d_glcm_analytic.h, which states the definitions and why MATLAB-style
// binning makes the matrix asymmetric.
//
// The fixture is a 5 x 5 x 3 box, every voxel the ROI's: slice 0 is the 5 x 5 GLCM example image of
// the PyRadiomics documentation, slice 1 its left-right mirror and slice 2 its transpose. Its
// left-right mirror is the second volume. Mirroring turns the matrix of a direction (1, dy, dz) into
// the transpose of the original's matrix of direction (1, -dy, -dz), so the features that do not
// change under a transpose take that direction's value.
//
// Kind: *analytic* per tests/vetting/SPEC.md 4: the reference is the definition evaluated here, on a
// co-occurrence matrix counted from the fixture without Nyxus.

static const int glcm_3d_asym_side = 5, glcm_3d_asym_depth = 3;
static const unsigned int glcm_3d_asym_slice[glcm_3d_asym_side][glcm_3d_asym_side] =
{
    {1, 2, 5, 2, 3},
    {3, 2, 1, 3, 1},
    {1, 3, 5, 5, 2},
    {1, 1, 1, 1, 2},
    {1, 2, 4, 3, 5}
};
static const int glcm_3d_asym_levels = 64;   // the default coarse grey depth

// The 13 directions, (dx, dy, dz), in the order 3d_glcm.cpp walks them and stores their values
static const int glcm_3d_asym_shifts[13][3] =
{
    {1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1}, {1, -1, 1}, {1, -1, 0}, {1, -1, -1},
    {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 0, 1}
};

static unsigned int glcm_3d_asym_intensity (int x, int y, int z, bool mirror)
{
    if (mirror)
        x = glcm_3d_asym_side - 1 - x;
    switch (z)
    {
    case 0: return glcm_3d_asym_slice[y][x];
    case 1: return glcm_3d_asym_slice[y][glcm_3d_asym_side - 1 - x];
    default: return glcm_3d_asym_slice[x][y];
    }
}

static std::vector<std::vector<double>> run_3d_glcm_asym (bool mirror, double softnan)
{
    LR r (1);
    r.slide_idx = -1;
    r.aabb.init_x (0); r.aabb.update_x (glcm_3d_asym_side - 1);
    r.aabb.init_y (0); r.aabb.update_y (glcm_3d_asym_side - 1);
    r.aabb.init_z (0); r.aabb.update_z (glcm_3d_asym_depth - 1);
    r.aux_min = ~(PixIntens) 0;
    r.aux_max = 0;
    for (int z = 0; z < glcm_3d_asym_depth; z++)
        for (int y = 0; y < glcm_3d_asym_side; y++)
            for (int x = 0; x < glcm_3d_asym_side; x++)
            {
                const PixIntens v = glcm_3d_asym_intensity (x, y, z, mirror);
                r.raw_pixels_3D.push_back (Pixel3 (x, y, z, v));
                r.aux_min = (std::min) (r.aux_min, v);
                r.aux_max = (std::max) (r.aux_max, v);
            }
    r.aux_area = (decltype(r.aux_area)) r.raw_pixels_3D.size();

    Fsettings s;
    s.resize ((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = softnan;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = glcm_3d_asym_levels;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = glcm_3d_asym_levels;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;

    D3_GLCM_feature::symmetric_glcm = false;
    D3_GLCM_feature f;
    f.calculate (r, s);
    r.initialize_fvals();
    f.save_value (r.fvals);
    return r.fvals;
}

struct Glcm3dIbsiValues
{
    double clutend, clushade, cluprom, sumvariance, jvar, infomeas1;
};

// The IBSI values of the fixture in direction (dx, dy, dz): P(i, j), i the MATLAB-style level of the
// voxel and j that of its neighbour, floor(n x / max + 1) clipped to 1..n over the ROI's maximum.
static Glcm3dIbsiValues glcm_3d_ibsi_values (bool mirror, int dx, int dy, int dz)
{
    const int n = glcm_3d_asym_levels, side = glcm_3d_asym_side, depth = glcm_3d_asym_depth;
    unsigned int mx = 0;
    for (int z = 0; z < depth; z++)
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++)
                mx = std::max (mx, glcm_3d_asym_intensity (x, y, z, mirror));
    auto level = [&](int x, int y, int z)
    {
        const int l = (int) std::floor (double(n) * glcm_3d_asym_intensity (x, y, z, mirror) / double(mx) + 1.);
        return std::min (std::max (l, 1), n);
    };

    std::vector<std::vector<double>> p (n + 1, std::vector<double> (n + 1, 0.));
    double total = 0;
    for (int z = 0; z < depth; z++)
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++)
            {
                const int xn = x + dx, yn = y + dy, zn = z + dz;
                if (xn < 0 || xn >= side || yn < 0 || yn >= side || zn < 0 || zn >= depth)
                    continue;
                p[level (x, y, z)][level (xn, yn, zn)] += 1;
                total += 1;
            }

    std::vector<double> px (n + 1, 0.), py (n + 1, 0.), pxpy (2 * n + 1, 0.);
    double mux = 0, muy = 0;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
        {
            p[i][j] /= total;
            px[i] += p[i][j];
            py[j] += p[i][j];
            pxpy[i + j] += p[i][j];
            mux += i * p[i][j];
            muy += j * p[i][j];
        }

    Glcm3dIbsiValues v {};
    double hxy = 0, hxy1 = 0, hx = 0, hy = 0;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
        {
            const double m = i + j - mux - muy;
            v.clutend += m * m * p[i][j];
            v.clushade += m * m * m * p[i][j];
            v.cluprom += m * m * m * m * p[i][j];
            v.jvar += (i - mux) * (i - mux) * p[i][j];
            if (p[i][j] > 0)
            {
                hxy -= p[i][j] * std::log2 (p[i][j]);
                hxy1 -= p[i][j] * std::log2 (px[i] * py[j]);
            }
        }
    for (int i = 1; i <= n; i++)
    {
        if (px[i] > 0)
            hx -= px[i] * std::log2 (px[i]);
        if (py[i] > 0)
            hy -= py[i] * std::log2 (py[i]);
    }
    v.infomeas1 = (hxy - hxy1) / std::max (hx, hy);

    double sa = 0;
    for (int k = 2; k <= 2 * n; k++)
        sa += k * pxpy[k];
    for (int k = 2; k <= 2 * n; k++)
        v.sumvariance += (k - sa) * (k - sa) * pxpy[k];

    return v;
}

static void assert_3d_glcm_near (double actual, double expected, double rel, const std::string& what)
{
    EXPECT_NEAR (actual, expected, std::abs (expected) * rel) << what;
}

static std::string glcm_3d_asym_where (bool mirror, int k)
{
    return std::string (mirror ? "mirror" : "fixture") + " direction (" + std::to_string (glcm_3d_asym_shifts[k][0]) + ","
        + std::to_string (glcm_3d_asym_shifts[k][1]) + "," + std::to_string (glcm_3d_asym_shifts[k][2]) + ")";
}

// Cluster tendency, shade and prominence centre i + j on mu_x + mu_y, and sum variance is the variance
// of p_{x+y}, in all 13 directions of the fixture and of its mirror.
void test_3d_glcm_cluster_and_sum_variance_asymmetric_analytic()
{
    using F = Nyxus::Feature3D;
    for (bool mirror : {false, true})
    {
        const auto fv = run_3d_glcm_asym (mirror, 0.);
        ASSERT_EQ (fv[(int)F::GLCM_CLUSHADE].size(), 13u);
        for (int k = 0; k < 13; k++)
        {
            const auto ref = glcm_3d_ibsi_values (mirror, glcm_3d_asym_shifts[k][0], glcm_3d_asym_shifts[k][1], glcm_3d_asym_shifts[k][2]);
            const std::string at = glcm_3d_asym_where (mirror, k);
            assert_3d_glcm_near (fv[(int)F::GLCM_CLUTEND][k], ref.clutend, 1e-9, "3GLCM_CLUTEND " + at);
            assert_3d_glcm_near (fv[(int)F::GLCM_CLUSHADE][k], ref.clushade, 1e-9, "3GLCM_CLUSHADE " + at);
            assert_3d_glcm_near (fv[(int)F::GLCM_CLUPROM][k], ref.cluprom, 1e-9, "3GLCM_CLUPROM " + at);
            assert_3d_glcm_near (fv[(int)F::GLCM_SUMVARIANCE][k], ref.sumvariance, 1e-9, "3GLCM_SUMVARIANCE " + at);
        }
    }
}

// Joint variance is sum (i - mu_x)^2 p_ij in every direction, the value 3GLCM_VARIANCE reports.
void test_3d_glcm_jvar_asymmetric_analytic()
{
    using F = Nyxus::Feature3D;
    for (bool mirror : {false, true})
    {
        const auto fv = run_3d_glcm_asym (mirror, 0.);
        ASSERT_EQ (fv[(int)F::GLCM_JVAR].size(), 13u);
        for (int k = 0; k < 13; k++)
        {
            const auto ref = glcm_3d_ibsi_values (mirror, glcm_3d_asym_shifts[k][0], glcm_3d_asym_shifts[k][1], glcm_3d_asym_shifts[k][2]);
            const std::string at = glcm_3d_asym_where (mirror, k);
            assert_3d_glcm_near (fv[(int)F::GLCM_VARIANCE][k], ref.jvar, 1e-9, "3GLCM_VARIANCE " + at);
            assert_3d_glcm_near (fv[(int)F::GLCM_JVAR][k], ref.jvar, 1e-9, "3GLCM_JVAR " + at);
        }
    }
}

// Information correlation 1 divides by max(HX, HY). The fast_log10 entropies (helpers.h) put up to
// 1.4e-2 relative on this quotient in one of the 26 directions here, and the band is 2e-2, below the
// 4.2 % a divisor taken from one fixed marginal is off in the mirror's direction (1, 0, 1). The
// orientation check pairs each direction (1, dy, dz) of the mirror with (1, -dy, -dz) of the fixture
// and needs only rounding.
void test_3d_glcm_infomeas1_asymmetric_analytic()
{
    const int code = (int)Nyxus::Feature3D::GLCM_INFOMEAS1;
    const auto fixture = run_3d_glcm_asym (false, 0.),
        mirror = run_3d_glcm_asym (true, 0.);
    ASSERT_EQ (fixture[code].size(), 13u);
    for (bool m : {false, true})
        for (int k = 0; k < 13; k++)
        {
            const auto ref = glcm_3d_ibsi_values (m, glcm_3d_asym_shifts[k][0], glcm_3d_asym_shifts[k][1], glcm_3d_asym_shifts[k][2]);
            assert_3d_glcm_near ((m ? mirror : fixture)[code][k], ref.infomeas1, 2e-2, "3GLCM_INFOMEAS1 " + glcm_3d_asym_where (m, k));
        }

    for (int k = 0; k < 13; k++)
    {
        if (glcm_3d_asym_shifts[k][0] != 1)
            continue;
        int t = -1;
        for (int q = 0; q < 13; q++)
            if (glcm_3d_asym_shifts[q][0] == 1 && glcm_3d_asym_shifts[q][1] == -glcm_3d_asym_shifts[k][1] && glcm_3d_asym_shifts[q][2] == -glcm_3d_asym_shifts[k][2])
                t = q;
        ASSERT_GE (t, 0);
        assert_3d_glcm_near (mirror[code][k], fixture[code][t], 1e-12, "3GLCM_INFOMEAS1 mirror " + glcm_3d_asym_where (true, k) + " vs fixture " + glcm_3d_asym_where (false, t));
    }
}
