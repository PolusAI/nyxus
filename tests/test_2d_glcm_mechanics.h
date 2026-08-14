#pragma once

#include <gtest/gtest.h>
#include <typeinfo>

#include "../src/nyx/environment.h"
#include "../src/nyx/feature_settings.h"
#include "../src/nyx/features/glcm.h"

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
