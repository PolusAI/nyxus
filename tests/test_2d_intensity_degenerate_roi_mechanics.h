#pragma once

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "../src/nyx/dataset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// ---------------------------------------------------------------------------
// Degenerate-ROI mechanics for the 2D intensity family (SPEC.md 2: plumbing, no
// correctness claim).
//
// QCOD and UNIFORMITY_PIU are quotients whose denominators are sums -- p75 + p25 and
// aux_max + aux_min -- that a populated ROI sitting entirely at grey level 0 leaves at
// 0. Such an ROI is ordinary once the load-time offset map is in play, since that map
// puts the slide's own minimum on grey level 0.
//
// This has to be asserted HERE, on the feature object, rather than through the Python
// API: every value leaving the pipeline goes through Nyxus::force_finite_number() in
// output_2_buffer.cpp, which replaces a NaN with ResultOptions::noval(). At the default
// noval of 0.0 that substitution is indistinguishable from the computed 0, so no
// end-to-end test can tell a guarded quotient from an unguarded one. It becomes visible
// only under a non-default --noval, which the Python API does not expose. Reading
// save_value()'s output directly is what makes the distinction observable.
// ---------------------------------------------------------------------------

// A populated ROI whose pixels are all grey level 0 -- what an all-air CT ROI becomes
// once the offset map has shifted the slide's minimum onto 0.
static const NyxusPixel intensityZeroValuedRoiTestData[] =
{
    {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0},
    {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0},
    {0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {3, 2, 0}
};

// Both quotients come back as numbers rather than NaN, and specifically as the 0 the
// pre-guard code reported by returning before it ever computed them.
void test_2d_intensity_zero_valued_roi_ratios_mechanics()
{
    Dataset ds;
    ds.dataset_props.push_back (SlideProps("", ""));

    LR roidata (100);
    roidata.slide_idx = -1;
    load_test_roi_data (roidata, intensityZeroValuedRoiTestData,
        sizeof(intensityZeroValuedRoiTestData) / sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();

    // the premise: a populated ROI that is nonetheless all zeros
    ASSERT_FALSE (roidata.raw_pixels.empty());
    ASSERT_EQ (roidata.aux_min, 0u);
    ASSERT_EQ (roidata.aux_max, 0u);

    Fsettings s;
    PixelIntensityFeatures f;
    ASSERT_NO_THROW (f.calculate (roidata, s, ds));

    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    double qcod = roidata.fvals[(int)Feature2D::QCOD][0],
        piu = roidata.fvals[(int)Feature2D::UNIFORMITY_PIU][0];

    ASSERT_FALSE (std::isnan(qcod)) << "QCOD is NaN before the output sanitizer";
    ASSERT_FALSE (std::isnan(piu)) << "UNIFORMITY_PIU is NaN before the output sanitizer";
    ASSERT_DOUBLE_EQ (qcod, 0.0);
    ASSERT_DOUBLE_EQ (piu, 0.0);

    // the rest of the distribution is still described, which is the point of letting a
    // populated zero-valued ROI through in the first place
    ASSERT_DOUBLE_EQ (roidata.fvals[(int)Feature2D::UNIFORMITY][0], 1.0);
    ASSERT_DOUBLE_EQ (roidata.fvals[(int)Feature2D::MIN][0], 0.0);
    ASSERT_DOUBLE_EQ (roidata.fvals[(int)Feature2D::MAX][0], 0.0);
}

// The other half of the blank-ROI guard, which the guard's own rewrite left untested: an
// ROI with no pixels at all. It must return without computing a distribution and without
// reading past an empty cloud, leaving the histogram-derived features at their defaults.
void test_2d_intensity_empty_roi_mechanics()
{
    Dataset ds;
    ds.dataset_props.push_back (SlideProps("", ""));

    LR roidata (100);
    roidata.slide_idx = -1;
    // deliberately no load_test_roi_data() call -- the cloud stays empty
    ASSERT_TRUE (roidata.raw_pixels.empty());

    Fsettings s;
    PixelIntensityFeatures f;
    ASSERT_NO_THROW (f.calculate (roidata, s, ds));

    roidata.initialize_fvals();
    ASSERT_NO_THROW (f.save_value (roidata.fvals));

    // nothing was measured, so nothing may be claimed: the distribution features keep the
    // defaults reset() gave them
    ASSERT_DOUBLE_EQ (roidata.fvals[(int)Feature2D::UNIFORMITY][0], 0.0);
    ASSERT_DOUBLE_EQ (roidata.fvals[(int)Feature2D::ENTROPY][0], 0.0);
    ASSERT_TRUE (roidata.fvals[(int)Feature2D::HISTOGRAM].empty());
}

// A non-degenerate ROI is unaffected: both quotients keep their ordinary definitions.
void test_2d_intensity_nonzero_roi_ratios_unaffected_mechanics()
{
    Dataset ds;
    ds.dataset_props.push_back (SlideProps("", ""));

    LR roidata (100);
    roidata.slide_idx = -1;
    load_test_roi_data (roidata, pixelIntensityFeaturesTestData,
        sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();

    Fsettings s;
    PixelIntensityFeatures f;
    ASSERT_NO_THROW (f.calculate (roidata, s, ds));

    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    double qcod = roidata.fvals[(int)Feature2D::QCOD][0],
        piu = roidata.fvals[(int)Feature2D::UNIFORMITY_PIU][0];

    ASSERT_FALSE (std::isnan(qcod));
    ASSERT_FALSE (std::isnan(piu));
    // the guard must not have collapsed a live denominator to the degenerate answer
    ASSERT_NE (piu, 0.0);
}
