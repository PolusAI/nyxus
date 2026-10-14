#pragma once

// 3D GLRLM emphases against their closed form, on volumes whose run lengths and grey levels are large
// enough that a squared term no longer fits in 32 bits (SPEC 4, oracle `analytic`).

#include <gtest/gtest.h>
#include <vector>
#include "../src/nyx/features/3d_glrlm.h"     // D3_GLRLM_feature, and SimpleCube / PixIntens with it
#include "test_main_nyxus.h"                  // the Environment / roi_cache graph, Fsettings with it

// No binning (GLRLM_GREYDEPTH = 0), which keeps the levels of a fixture as they are.
static Fsettings make_glrlm3d_nobinning_settings()
{
    Fsettings s;
    s.resize ((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 64;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = 0;
    return s;
}

// Featurises a w x h x d volume held in memory, every non-zero voxel inside the ROI, into 'fvals'
// (indexed by Nyxus 3D feature code). The LR is built the way the prescan hands one to calculate():
// the voxel cube, the voxel cloud and the running intensity extrema.
static void run_3d_glrlm_on_volume (
    std::vector<std::vector<double>>& fvals,
    const std::vector<PixIntens>& voxels,
    int w, int h, int d)
{
    ASSERT_EQ (voxels.size(), size_t(w) * size_t(h) * size_t(d));

    LR r;
    r.aux_image_cube = SimpleCube<PixIntens> (voxels, w, h, d);
    for (int z = 0; z < d; z++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                PixIntens v = r.aux_image_cube.zyx (z, y, x);
                if (!v)
                    continue;
                r.raw_pixels_3D.push_back (Pixel3 (x, y, z, v));
                if (v < r.aux_min)
                    r.aux_min = v;
                if (v > r.aux_max)
                    r.aux_max = v;
            }

    D3_GLRLM_feature f;
    ASSERT_NO_THROW (r.initialize_fvals());
    ASSERT_NO_THROW (f.calculate (r, make_glrlm3d_nobinning_settings()));
    f.save_value (r.fvals);
    fvals = r.fvals;
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
    assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRE, along_x, 1.0 / L2);
    assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRE, along_x, L2);
    assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRLGLE, along_x, 0.625 / L2);
    assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRHGLE, along_x, 2.5 / L2);
    assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRLGLE, along_x, 0.625 * L2);
    assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRHGLE, along_x, 2.5 * L2);
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
        assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LGLRE, angle, low);
        assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRLGLE, angle, low);
        assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRLGLE, angle, low);
        assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_HGLRE, angle, high);
        assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_SRHGLE, angle, high);
        assert_3d_glrlm_closed_form (fvals, Feature3D::GLRLM_LRHGLE, angle, high);
    }
}
