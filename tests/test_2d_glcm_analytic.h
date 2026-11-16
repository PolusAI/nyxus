#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "test_2d_glcm_common.h"   // GLCMFeature, Fsettings, LR, NyxusPixel, load_masked_test_roi_data

// 2D GLCM on an asymmetric co-occurrence matrix against the IBSI definitions evaluated in the test.
//
// MATLAB-style binning, Nyxus' default (a positive grey depth, 64 levels here), counts each ordered
// pair of pixels once, so the matrix P(i, j) -- i the grey level of the pixel, j that of its neighbour
// at the angle's offset -- is not symmetric. The features below read the two marginals p_x (the
// pixel's) and p_y (the neighbour's) separately, and on such a matrix they tell apart a definition that
// mixes the marginals from IBSI's:
//
//   cluster tendency, shade, prominence (DG8W, 7NFM, AE86)  sum_ij (i + j - mu_x - mu_y)^{2,3,4} p_ij
//   sum variance (OEEB)                                     sum_k (k - SA)^2 p_{x+y}(k), SA = sum_k k p_{x+y}(k)
//   joint variance (UR99)                                   sum_ij (i - mu_x)^2 p_ij
//   information correlation 1 (R8DG)                        (HXY - HXY1) / max(HX, HY)
//
// PyRadiomics run with symmetricalGLCM=False uses the same definitions (ClusterTendency, ClusterShade,
// ClusterProminence, SumSquares, Imc1); on the binned levels of the fixture below it agrees with this
// reference to 2e-15 for both orientations.
//
// The fixture is the 5 x 5 GLCM example image of the PyRadiomics documentation, and its left-right
// mirror. Mirroring turns the 0-degree matrix into its transpose, so a feature that depends on the
// matrix only through the IBSI expressions above, all symmetric in the two marginals except joint
// variance, has the same 0-degree value on both; and it turns the 45-degree matrix into the mirror's
// 135-degree one.
//
// Kind: *analytic* per tests/vetting/SPEC.md 4: the reference is the definition evaluated here, on a
// co-occurrence matrix counted from the fixture without Nyxus.

static const unsigned int glcm_2d_asym_side = 5;
static const unsigned int glcm_2d_asym_image[glcm_2d_asym_side][glcm_2d_asym_side] =
{
    {1, 2, 5, 2, 3},
    {3, 2, 1, 3, 1},
    {1, 3, 5, 5, 2},
    {1, 1, 1, 1, 2},
    {1, 2, 4, 3, 5}
};
static const int glcm_2d_asym_levels = 64;   // the default coarse grey depth

// The fixture's intensity at (x, y), mirrored left-right when 'mirror' is set.
static unsigned int glcm_2d_asym_intensity (unsigned int x, unsigned int y, bool mirror)
{
    return glcm_2d_asym_image[y][mirror ? glcm_2d_asym_side - 1 - x : x];
}

// The fixture inside a one-pixel background frame, so the ROI fills the 5 x 5 square.
static void make_2d_glcm_asym_roi (LR& roidata, bool mirror)
{
    const unsigned int side = glcm_2d_asym_side + 2;
    std::vector<NyxusPixel> intensity, mask;
    for (unsigned int y = 0; y < side; y++)
        for (unsigned int x = 0; x < side; x++)
        {
            const bool in_roi = x >= 1 && x <= glcm_2d_asym_side && y >= 1 && y <= glcm_2d_asym_side;
            intensity.push_back (NyxusPixel {x, y, in_roi ? glcm_2d_asym_intensity (x - 1, y - 1, mirror) : 0u});
            mask.push_back (NyxusPixel {x, y, in_roi ? 1u : 0u});
        }
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
}

// Default binning: MATLAB-style at 64 levels, asymmetric matrix, offset 1, the four 2D angles.
static Fsettings make_2d_glcm_asym_settings()
{
    Fsettings s = make_2d_glcm_dense_settings();
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = glcm_2d_asym_levels;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = glcm_2d_asym_levels;
    return s;
}

static std::vector<std::vector<double>> run_2d_glcm_asym (bool mirror)
{
    LR roidata;
    make_2d_glcm_asym_roi (roidata, mirror);
    GLCMFeature::symmetric_glcm = false;
    GLCMFeature::angles = {0, 45, 90, 135};
    GLCMFeature f;
    f.calculate (roidata, make_2d_glcm_asym_settings());
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);
    return roidata.fvals;
}

struct Glcm2dIbsiValues
{
    double clutend, clushade, cluprom, sumvariance, jvar, infomeas1;
};

// The IBSI values of the fixture at offset (dx, dy), each angle's offset in glcm.cpp. Levels are
// MATLAB-style, floor(n x / max + 1) clipped to 1..n over the ROI's maximum, which is how a positive
// grey depth bins; the whole 5 x 5 square is the ROI.
static Glcm2dIbsiValues glcm_2d_ibsi_values (bool mirror, int dx, int dy)
{
    const int n = glcm_2d_asym_levels, side = (int) glcm_2d_asym_side;
    unsigned int mx = 0;
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++)
            mx = std::max (mx, glcm_2d_asym_intensity (x, y, mirror));
    auto level = [&](int x, int y)
    {
        const int l = (int) std::floor (double(n) * glcm_2d_asym_intensity (x, y, mirror) / double(mx) + 1.);
        return std::min (std::max (l, 1), n);
    };

    // P(i, j), i the level of pixel (x, y) and j that of its neighbour (x + dx, y + dy), normalised
    std::vector<std::vector<double>> p (n + 1, std::vector<double> (n + 1, 0.));
    double total = 0;
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++)
        {
            const int xn = x + dx, yn = y + dy;
            if (xn < 0 || xn >= side || yn < 0 || yn >= side)
                continue;
            p[level (x, y)][level (xn, yn)] += 1;
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

    Glcm2dIbsiValues v {};
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

// (dx, dy) of GLCMFeature::angles {0, 45, 90, 135}, as Extract_Texture_Features2 shifts them
static const int glcm_2d_asym_offsets[4][2] = { {1, 0}, {1, 1}, {0, 1}, {-1, 1} };

static void assert_2d_glcm_near (double actual, double expected, double rel, const std::string& what)
{
    EXPECT_NEAR (actual, expected, std::abs (expected) * rel) << what;
}

// Cluster tendency, shade and prominence centre i + j on mu_x + mu_y, and sum variance is the variance
// of p_{x+y}; each angle of the fixture and of its mirror matches the IBSI expression. Without a log in
// any of them, the band is rounding only.
void test_2d_glcm_cluster_and_sum_variance_asymmetric_analytic()
{
    for (bool mirror : {false, true})
    {
        const auto fv = run_2d_glcm_asym (mirror);
        for (int a = 0; a < 4; a++)
        {
            const auto ref = glcm_2d_ibsi_values (mirror, glcm_2d_asym_offsets[a][0], glcm_2d_asym_offsets[a][1]);
            const std::string at = std::string (mirror ? "mirror" : "fixture") + " angle " + std::to_string (GLCMFeature::angles[a]);
            assert_2d_glcm_near (fv[(int)Nyxus::Feature2D::GLCM_CLUTEND][a], ref.clutend, 1e-9, "CLUTEND " + at);
            assert_2d_glcm_near (fv[(int)Nyxus::Feature2D::GLCM_CLUSHADE][a], ref.clushade, 1e-9, "CLUSHADE " + at);
            assert_2d_glcm_near (fv[(int)Nyxus::Feature2D::GLCM_CLUPROM][a], ref.cluprom, 1e-9, "CLUPROM " + at);
            assert_2d_glcm_near (fv[(int)Nyxus::Feature2D::GLCM_SUMVARIANCE][a], ref.sumvariance, 1e-9, "SUMVARIANCE " + at);
        }
    }
}

// Joint variance is the variance of the pixel's own level, sum (i - mu_x)^2 p_ij: the IBSI value at
// every angle, and the same number GLCM_VARIANCE reports.
void test_2d_glcm_jvar_asymmetric_analytic()
{
    for (bool mirror : {false, true})
    {
        const auto fv = run_2d_glcm_asym (mirror);
        for (int a = 0; a < 4; a++)
        {
            const auto ref = glcm_2d_ibsi_values (mirror, glcm_2d_asym_offsets[a][0], glcm_2d_asym_offsets[a][1]);
            const std::string at = std::string (mirror ? "mirror" : "fixture") + " angle " + std::to_string (GLCMFeature::angles[a]);
            assert_2d_glcm_near (fv[(int)Nyxus::Feature2D::GLCM_JVAR][a], ref.jvar, 1e-9, "JVAR " + at);
            assert_2d_glcm_near (fv[(int)Nyxus::Feature2D::GLCM_JVAR][a], fv[(int)Nyxus::Feature2D::GLCM_VARIANCE][a], 1e-12, "JVAR vs VARIANCE " + at);
        }
    }
}

// Information correlation 1 divides by the larger marginal entropy. Nyxus' entropies read the
// fast_log10 approximation (helpers.h), a convention that puts up to 7.0e-3 relative on this quotient
// of a difference of entropies; the band is 1.5e-2, below the 4.3 % a divisor taken from one fixed
// marginal is off on the mirror's 0-degree matrix. The orientation check needs no band beyond
// rounding: the mirror's 0-degree matrix is the transpose of the fixture's, and HXY, HXY1 and
// max(HX, HY) do not change under a transpose.
void test_2d_glcm_infomeas1_asymmetric_analytic()
{
    const auto fixture = run_2d_glcm_asym (false),
        mirror = run_2d_glcm_asym (true);
    for (bool m : {false, true})
    {
        const auto& fv = m ? mirror : fixture;
        for (int a = 0; a < 4; a++)
        {
            const auto ref = glcm_2d_ibsi_values (m, glcm_2d_asym_offsets[a][0], glcm_2d_asym_offsets[a][1]);
            assert_2d_glcm_near (fv[(int)Nyxus::Feature2D::GLCM_INFOMEAS1][a], ref.infomeas1, 1.5e-2,
                std::string ("INFOMEAS1 ") + (m ? "mirror" : "fixture") + " angle " + std::to_string (GLCMFeature::angles[a]));
        }
    }

    const int code = (int)Nyxus::Feature2D::GLCM_INFOMEAS1;
    assert_2d_glcm_near (mirror[code][0], fixture[code][0], 1e-12, "INFOMEAS1 at 0 degrees, mirror vs fixture");
    assert_2d_glcm_near (mirror[code][3], fixture[code][1], 1e-12, "INFOMEAS1 at 135 degrees on the mirror vs 45 on the fixture");
}
