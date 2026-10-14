#pragma once

// 2D GLRLM emphases against their closed form, on ROIs whose run lengths and grey levels are large
// enough that a squared term no longer fits in 32 bits (SPEC 4, oracle `analytic`).

// Nothing of its own: gtest, <vector>, GLRLMFeature, Feature2D, the IBSI settings recipe and the
// ROI loader all arrive with the common header.
#include "test_2d_glrlm_common.h"

// Featurises one fully masked ROI given as (x, y, intensity) pixels, at the IBSI recipe, which keeps
// the grey levels as they are.
static void run_2d_glrlm_on_pixels (LR& roidata, const std::vector<NyxusPixel>& intensity)
{
    std::vector<NyxusPixel> mask (intensity);
    for (auto& px : mask)
        px.intensity = 1u;
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());

    GLRLMFeature f;
    f.calculate (roidata, make_2d_glrlm_ibsi_settings());
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);
}

static void assert_2d_glrlm_closed_form (const LR& roidata, Feature2D fcode, int angle, double expected)
{
    const double actual = roidata.fvals[(int)fcode][angle];
    ASSERT_NEAR (actual, expected, 1e-12 * expected) << "feature " << (int)fcode << " angle " << angle;
}

// bench_glrlm_2d_two_long_runs -- one row of 2L pixels, L = 46,656, the left half at intensity 1 and
// the right half at intensity 2. At 0 degrees that is two runs, one of length L at each level, and
// L^2 = 2,176,782,336 is past INT_MAX while 2^2 * L^2 is past UINT_MAX, so every run-length term is
// right here only if it is taken in double. Over the Nr = 2 runs:
//
//   SRE    = 1/L^2                 LRE    = L^2
//   SRLGLE = (1 + 1/4) / 2 / L^2   SRHGLE = (1 + 4) / 2 / L^2
//   LRLGLE = (1 + 1/4) / 2 * L^2   LRHGLE = (1 + 4) / 2 * L^2
void test_2d_glrlm_long_run_emphasis_analytic()
{
    const size_t L = 46656;
    std::vector<NyxusPixel> intensity;
    for (size_t x = 0; x < 2 * L; x++)
        intensity.push_back ({ x, 0, x < L ? 1u : 2u });

    LR roidata;
    ASSERT_NO_FATAL_FAILURE (run_2d_glrlm_on_pixels (roidata, intensity));

    const double L2 = double(L) * double(L);
    ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_SRE, 0, 1.0 / L2));
    ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_LRE, 0, L2));
    ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_SRLGLE, 0, 0.625 / L2));
    ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_SRHGLE, 0, 2.5 / L2));
    ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_LRLGLE, 0, 0.625 * L2));
    ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_LRHGLE, 0, 2.5 * L2));
}

// bench_glrlm_2d_high_grey_pair -- two adjacent pixels at intensities 1 and G = 70,000. Every
// direction sees two runs of length 1, and G^2 = 4.9e9 is past UINT_MAX, so the grey-level terms are
// right here only if the level is squared in double. Over the Nr = 2 runs, in all four directions:
//
//   LGLRE = SRLGLE = LRLGLE = (1 + 1/G^2) / 2
//   HGLRE = SRHGLE = LRHGLE = (1 + G^2) / 2
void test_2d_glrlm_high_grey_level_emphasis_analytic()
{
    const double G = 70000.0;
    std::vector<NyxusPixel> intensity = { { 0, 0, 1u }, { 1, 0, 70000u } };

    LR roidata;
    ASSERT_NO_FATAL_FAILURE (run_2d_glrlm_on_pixels (roidata, intensity));

    const double low = (1.0 + 1.0 / (G * G)) / 2.0,
        high = (1.0 + G * G) / 2.0;
    for (int angle = 0; angle < 4; angle++)
    {
        ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_LGLRE, angle, low));
        ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_SRLGLE, angle, low));
        ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_LRLGLE, angle, low));
        ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_HGLRE, angle, high));
        ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_SRHGLE, angle, high));
        ASSERT_NO_FATAL_FAILURE (assert_2d_glrlm_closed_form (roidata, Feature2D::GLRLM_LRHGLE, angle, high));
    }
}
