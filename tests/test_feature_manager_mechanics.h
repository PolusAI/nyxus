#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/feature_mgr.h"

// FeatureManager::compile() runs check_11_correspondence() (every registered Feature2D/Feature3D
// code has exactly one provider) then gather_dependencies() (no cyclic dependency). Registration
// itself happens in the FeatureManager constructor (feature_mgr_init.cpp), so a bare instance is
// already fully populated -- no phantom, no image I/O, no ROI setup needed to exercise this.
//
// This is the dedicated, cheap check that closes the gap check_11_correspondence() used to leave
// for 3D (feature_mgr.cpp, "//xxxx what about 3D and IMQ?"): before this, the only thing standing
// in for it was the Wave-9 coverage sweep's per-family assert_3d_feature_is_registered_and_computable,
// which paid for a full phantom pipeline run per feature to check something that needs none of that.
void test_feature_manager_mechanics()
{
    FeatureManager fm;
    ASSERT_TRUE(fm.compile());
}
