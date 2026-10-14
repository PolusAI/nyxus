#pragma once

// 3D GLRLM emphases against their closed form, on volumes whose run lengths and grey levels are large
// enough that a squared term no longer fits in 32 bits (SPEC 4, oracle `analytic`).

// gtest, <vector>, Fsettings and the volume runner arrive with the common headers; the feature method
// is this file's own.
#include "../src/nyx/features/3d_glrlm.h"     // D3_GLRLM_feature
#include "test_3d_glrlm_common.h"             // make_glrlm3d_settings

// Featurises a volume held in memory, every non-zero voxel inside the ROI, into 'fvals' (indexed by
// Nyxus 3D feature code), with no binning (GLRLM_GREYDEPTH = 0), which keeps the levels of a fixture
// as they are.
static void run_3d_glrlm_on_volume (
    std::vector<std::vector<double>>& fvals,
    const std::vector<PixIntens>& voxels,
    int w, int h, int d)
{
    D3_GLRLM_feature f;
    run_3d_feature_on_volume (fvals, voxels, w, h, d, make_glrlm3d_settings (64/*greydepth, inert for GLRLM*/, 0/*no binning*/), f);
}

static void assert_3d_glrlm_closed_form (const std::vector<std::vector<double>>& fvals, Feature3D fcode, int angle, double expected)
{
    const double actual = fvals[(int)fcode][angle];
    ASSERT_NEAR (actual, expected, 1e-12 * expected) << "feature " << (int)fcode << " angle " << angle;
}

// bench_glrlm_3d_two_long_runs -- a 2L x 1 x 1 line of voxels, L = 46,656, the first half at
// intensity 1 and the second at intensity 2. Direction slot 12 is {dz=0, dy=0, dx=1}, along the
// line, and sees two runs, one of length L at each level; L^2 = 2,176,782,336 is past INT_MAX and
// 2^2 * L^2 is past UINT_MAX, so every run-length term is right here only if it is taken in double.
// Over its Nr = 2 runs:
//
//   SRE    = 1/L^2                 LRE    = L^2
//   SRLGLE = (1 + 1/4) / 2 / L^2   SRHGLE = (1 + 4) / 2 / L^2
//   LRLGLE = (1 + 1/4) / 2 * L^2   LRHGLE = (1 + 4) / 2 * L^2
void test_3d_glrlm_long_run_emphasis_analytic()
{
    const int L = 46656;
    std::vector<PixIntens> voxels (2 * size_t(L));
    for (size_t x = 0; x < voxels.size(); x++)
        voxels[x] = x < size_t(L) ? 1 : 2;

    std::vector<std::vector<double>> fvals;
    ASSERT_NO_FATAL_FAILURE (run_3d_glrlm_on_volume (fvals, voxels, 2 * L, 1, 1));

    const int along_x = 12;
    const double L2 = double(L) * double(L);
    ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRE, along_x, 1.0 / L2));
    ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRE, along_x, L2));
    ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRLGLE, along_x, 0.625 / L2));
    ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRHGLE, along_x, 2.5 / L2));
    ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRLGLE, along_x, 0.625 * L2));
    ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRHGLE, along_x, 2.5 * L2));
}

// bench_glrlm_3d_high_grey_pair -- two adjacent voxels at intensities 1 and G = 70,000. Every one of
// the 13 directions sees two runs of length 1, and G^2 = 4.9e9 is past UINT_MAX, so the grey-level
// terms are right here only if the level is squared in double. Over the Nr = 2 runs:
//
//   LGLRE = SRLGLE = LRLGLE = (1 + 1/G^2) / 2
//   HGLRE = SRHGLE = LRHGLE = (1 + G^2) / 2
void test_3d_glrlm_high_grey_level_emphasis_analytic()
{
    const double G = 70000.0;
    std::vector<PixIntens> voxels = { 1, 70000 };

    std::vector<std::vector<double>> fvals;
    ASSERT_NO_FATAL_FAILURE (run_3d_glrlm_on_volume (fvals, voxels, 2, 1, 1));

    const double low = (1.0 + 1.0 / (G * G)) / 2.0,
        high = (1.0 + G * G) / 2.0;
    for (int angle = 0; angle < 13; angle++)
    {
        ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LGLRE, angle, low));
        ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRLGLE, angle, low));
        ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRLGLE, angle, low));
        ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_HGLRE, angle, high));
        ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRHGLE, angle, high));
        ASSERT_NO_FATAL_FAILURE (assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRHGLE, angle, high));
    }
}
