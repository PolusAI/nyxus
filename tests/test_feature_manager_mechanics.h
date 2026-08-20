#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/feature_mgr.h"

// FeatureManager::compile() runs check_11_correspondence() (every registered Feature2D/Feature3D
// code has exactly one provider) then gather_dependencies() (no cyclic dependency). Registration
// itself happens in the FeatureManager constructor (feature_mgr_init.cpp), so a bare instance is
// already fully populated -- no phantom, no image I/O, no ROI setup needed to exercise this.
//
// This is the dedicated, cheap check for 3D registration correspondence (feature_mgr.cpp,
// "//xxxx what about 3D and IMQ?"), run with no phantom or full pipeline needed.
// History: tests/vetting/audit/glcm_3d_golden_regen.md, "grey64 table and the retired Wave-9 sweep".
void test_feature_manager_mechanics()
{
    FeatureManager fm;
    ASSERT_TRUE(fm.compile());
}
