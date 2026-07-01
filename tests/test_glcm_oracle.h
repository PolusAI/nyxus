#pragma once

#include <gtest/gtest.h>
#include <typeinfo>

#include "../src/nyx/environment.h"
#include "../src/nyx/feature_settings.h"
#include "../src/nyx/features/glcm.h"

// Regression guard for the GLCM "offset = 0 default" defect (found 2026-06).
//
// The production path (CLI and Python featurize) builds GLCM settings via
// Environment::compile_feature_settings(). That function used to initialise only the COMMON
// settings and left the GLCM-specific ones (GLCM_OFFSET / GLCM_GREYDEPTH / GLCM_NUMANG)
// zero-initialised. With GLCM_OFFSET == 0 the co-occurrence shift is dx = dy = 0, so every
// pixel co-occurs with itself -> a purely diagonal matrix -> CONTRAST = 0, CORRELATION = 1 for
// any image. The existing test_glcm.h cases never caught this because they hard-code
// GLCM_OFFSET = 1 (and run on a fully-masked phantom). This test exercises the real default.
inline void test_glcm_bug_offset_default_is_one()
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
