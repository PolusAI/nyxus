#pragma once

#include <gtest/gtest.h>
#include <typeinfo>

#include "../src/nyx/environment.h"
#include "../src/nyx/feature_settings.h"
#include "../src/nyx/features/glcm.h"
#include "test_2d_glcm_common.h"	// the dense phantom and its settings, load_masked_test_roi_data

// The production path (CLI and Python featurize) builds GLCM settings via
// Environment::compile_feature_settings(), which must initialise the GLCM-specific settings
// (GLCM_OFFSET / GLCM_GREYDEPTH / GLCM_NUMANG) and not just the COMMON ones. A zero GLCM_OFFSET
// makes the co-occurrence shift dx = dy = 0, so every pixel co-occurs with itself -> a purely
// diagonal matrix -> CONTRAST = 0 and CORRELATION = 1 for any image. The oracle tests cannot catch
// that because they set GLCM_OFFSET = 1 explicitly; this one exercises the real default.
inline void test_2d_glcm_bug_offset_default_is_one_mechanics()
{
    Environment e;
    e.set_coarse_gray_depth(64);
    e.compile_feature_settings();

    const Fsettings& s = e.get_feature_settings(typeid(GLCMFeature));

    // co-occurrence distance must default to 1 (IBSI delta = 1), NOT 0
    ASSERT_EQ(STNGS_GLCM_OFFSET(s), 1);
    // the GLCM-specific grey depth and angle count must be initialised, not left at 0
    ASSERT_GT(STNGS_GLCM_GREYDEPTH(s), 0);
    ASSERT_GT(STNGS_GLCM_NUMANG(s), 0);
}

// 2D GLCM bins at its own grey depth, GLCM_GREYDEPTH -- what glcm/greydepth sets -- and not at the
// common GREYDEPTH the other 2D families read. Measured on the dense phantom's eight levels with
// MATLAB binning, where 4 levels merge them and 64 keep them apart: a run at GLCM_GREYDEPTH=4 under a
// common depth of 64 has to equal the run with both at 4, value for value on every feature and
// angle, and differ from the run with both at 64.
static std::vector<std::vector<double>> run_2d_glcm_dense (int greydepth, int glcm_greydepth)
{
    std::vector<NyxusPixel> intensity, mask;
    make_2d_glcm_dense_phantom (intensity, mask);

    Fsettings s = make_2d_glcm_dense_settings();
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = greydepth;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = glcm_greydepth;

    LR roidata;
    GLCMFeature f;
    GLCMFeature::angles = {0, 45, 90, 135};
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
    f.calculate (roidata, s);
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);
    return roidata.fvals;
}

inline void test_2d_glcm_own_greydepth_bins_the_matrix_mechanics()
{
    const auto own = run_2d_glcm_dense (64, 4),
        both4 = run_2d_glcm_dense (4, 4),
        both64 = run_2d_glcm_dense (64, 64);

    bool any_differs = false;
    for (auto fc : GLCMFeature::featureset)
    {
        SCOPED_TRACE ("2D feature code " + std::to_string ((int)fc));
        const auto& a = own[(int)fc];
        ASSERT_EQ (a, both4[(int)fc]);
        any_differs = any_differs || a != both64[(int)fc];
    }
    ASSERT_TRUE (any_differs);
}