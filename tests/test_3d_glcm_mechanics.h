#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/feature_settings.h"   // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"         // Feature3D
#include "../src/nyx/features/3d_glcm.h"   // D3_GLCM_feature
#include "../src/nyx/roi_cache.h"          // LR

// Information correlation 1 is undefined where both marginals of a direction hold one grey level
// (HX = HY = 0), and reports the soft-NaN substitute there rather than NaN. The ROI is two voxels of
// level 7 side by side in x and one voxel of level 3 two slices away, so the direction (1, 0, 0),
// the fifth of the 13, pairs only 7 with 7 while the ROI itself has two levels. The substitute is -999
// so that it cannot be mistaken for a computed value.
void test_3d_glcm_infomeas1_single_level_reports_noval_mechanics()
{
    LR r (1);
    r.slide_idx = -1;
    r.aabb.init_x (0); r.aabb.update_x (1);
    r.aabb.init_y (0); r.aabb.update_y (0);
    r.aabb.init_z (0); r.aabb.update_z (2);
    r.raw_pixels_3D.push_back (Pixel3 (0, 0, 0, 7));
    r.raw_pixels_3D.push_back (Pixel3 (1, 0, 0, 7));
    r.raw_pixels_3D.push_back (Pixel3 (0, 0, 2, 3));
    r.aux_min = 3;
    r.aux_max = 7;
    r.aux_area = 3;

    const double noval = -999.;
    Fsettings s;
    s.resize ((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = noval;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 64;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 64;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;

    D3_GLCM_feature::symmetric_glcm = false;
    D3_GLCM_feature f;
    f.calculate (r, s);
    r.initialize_fvals();
    f.save_value (r.fvals);

    // the direction has pairs, so the value comes from the quotient's own guard
    const int x_dir = 4;
    ASSERT_GT (r.fvals[(int)Nyxus::Feature3D::GLCM_ASM][x_dir], 0.);
    ASSERT_EQ (r.fvals[(int)Nyxus::Feature3D::GLCM_INFOMEAS1][x_dir], noval);
}
