#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/gldm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// Regression guard for the GLDM "background pollutes the dependence matrix" defect (bug #14b,
// fixed 2026-06). C++ counterpart of tests/python/test_gldm_bugs.py.
//
// The MATLAB grey-binning path (the production default) maps off-ROI background (original
// intensity 0) to binned level 1, so the GLDM zone loop's old `pi == 0` guard - which tested
// the BINNED value - never rejected background pixels sitting inside the ROI's bounding box.
// Those background pixels were counted both as their own zones and as dependent neighbours,
// inflating Nz and every count/dependence feature. The fix skips a pixel by its ORIGINAL
// intensity (imR == 0) and only counts a neighbour that is itself an ROI pixel.
//
// The existing test_gldm.h phantom cases can't catch this: they run on a fully-masked ROI
// (mask == intensity), so no off-ROI background ever lands inside the bounding box. This test
// uses a concave ROI - a 3x3 ring with a background hole in the centre - so the bounding box
// genuinely contains a background pixel, exactly the production condition.
inline void test_gldm_bug_background_excluded()
{
    // ---- feature settings: MATLAB grey binning (128 levels), non-IBSI - the production default
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 128;   // matlab binning (> 0)
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    // ---- a 3x3 ring ROI with a background hole in the centre.
    // Intensities 127 and 128 both bin to MATLAB level 128 (level = floor(128*i/128 + 1),
    // clipped to 128), so every ROI pixel shares one grey level - which makes the expected
    // GLDM matrix exactly hand-computable. Two distinct values are used so aux_min != aux_max
    // and the "blank ROI" degenerate guard is not triggered. The centre (1,1) intensity is
    // irrelevant: it is masked out, so it enters the image matrix as background (0).
    static const NyxusPixel ring_int[] = {
        {0,0,128}, {1,0,128}, {2,0,128},
        {0,1,128}, {1,1,  0}, {2,1,127},
        {0,2,128}, {1,2,128}, {2,2,128}
    };
    static const NyxusPixel ring_seg[] = {
        {0,0,1}, {1,0,1}, {2,0,1},
        {0,1,1}, {1,1,0}, {2,1,1},   // centre masked OUT -> background inside the bounding box
        {0,2,1}, {1,2,1}, {2,2,1}
    };

    LR roidata;
    Nyxus::load_masked_test_roi_data(roidata, ring_int, ring_seg, sizeof(ring_seg) / sizeof(NyxusPixel));

    GLDMFeature f;
    ASSERT_NO_THROW(f.calculate(roidata, s));

    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // ---- expected values with the background correctly EXCLUDED.
    // Only the 8 ROI pixels contribute -> single grey level, Nz = 8. Each pixel's dependence
    // count (self + equal-level ROI neighbours) is 3 for the 4 corners and 5 for the 4 edge
    // pixels (the centre neighbour is background, so it is never counted). The dependence
    // matrix therefore has one row with 4 entries at dependence-size 3 and 4 at size 5:
    //   Nz  = 8
    //   GLN = 8^2 / 8            = 8      (single grey level -> GLN == Nz == the 8 ROI pixels)
    //   DN  = (4^2 + 4^2) / 8    = 4
    //   DNN = (4^2 + 4^2) / 8^2  = 0.5
    //   LDE = (4*3^2 + 4*5^2)/8  = 17
    //   SDE = (4/3^2 + 4/5^2)/8  = 0.0755556
    // Pre-fix the background pixel added a spurious level-1 zone (Nz = 9, a 2nd grey level),
    // giving GLN ~= 7.22, DN ~= 3.67, LDE ~= 15.22, SDE ~= 0.178 - all clearly distinguishable.
    const double tol = 1e-4;
    EXPECT_NEAR(roidata.fvals[(int)Nyxus::Feature2D::GLDM_GLN][0], 8.0,       tol)
        << "GLDM_GLN must equal the ROI pixel count (8) - background inflated Nz (bug regressed)";
    EXPECT_NEAR(roidata.fvals[(int)Nyxus::Feature2D::GLDM_DN][0],  4.0,       tol);
    EXPECT_NEAR(roidata.fvals[(int)Nyxus::Feature2D::GLDM_DNN][0], 0.5,       tol);
    EXPECT_NEAR(roidata.fvals[(int)Nyxus::Feature2D::GLDM_LDE][0], 17.0,      tol);
    EXPECT_NEAR(roidata.fvals[(int)Nyxus::Feature2D::GLDM_SDE][0], 0.0755556, tol);
}
