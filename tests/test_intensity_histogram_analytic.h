#pragma once

// TAXONOMY: oracle=analytic (SPEC §2/§6).
// Hand-computable intensity-histogram goldens on a tiny ROI — closed-form bin math,
// not a Nyxus self-snapshot. Split from test_intensity_histogram_regression.h.

#include "test_intensity_histogram_regression.h"

void test_ih_integer_domain_values()
{
    std::vector<std::vector<double>> fv;
    ih_run(fv, ih_make_settings(/*nbins*/ 3, /*ibsi*/ true));

    // bookkeeping
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_NUM_BINS), 3.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 2.0));

    // value family
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), 1.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), 7.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 6.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MEAN_VAL), 4.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MEDIAN_VAL), 4.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MODE_VAL), 2.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_VARIANCE_VAL), 3.2));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_EXCESS_KURTOSIS_VAL), -1.75, 1e3));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_UNIFORMITY_VAL), 0.36));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_ENTROPY_VAL), 1.521928));
    // symmetric distribution -> skewness ~ 0 (use absolute tolerance)
    ASSERT_NEAR(ih_get(fv, Feature2D::IH_SKEWNESS_VAL), 0.0, 1e-9);

    // index family (1-based)
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_IDX), 1.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_IDX), 3.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MEAN_IDX), 2.0));

    // gradients
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAX_GRADIENT), 1.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAX_GRADIENT_IDX), 3.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MIN_GRADIENT), -1.0, 1e3));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MIN_GRADIENT_IDX), 1.0));
}

void test_ih_index_and_percentile_bounds()
{
    const int N = 3;
    std::vector<std::vector<double>> fv;
    ih_run(fv, ih_make_settings(N, true));

    for (auto fc : { Feature2D::IH_MINIMUM_IDX, Feature2D::IH_MAXIMUM_IDX, Feature2D::IH_MEDIAN_IDX,
                     Feature2D::IH_P10_IDX, Feature2D::IH_P90_IDX, Feature2D::IH_MODE_IDX })
    {
        double v = ih_get(fv, fc);
        ASSERT_GE(v, 1.0);
        ASSERT_LE(v, (double)N);
    }
    for (auto fc : { Feature2D::IH_P10_VAL, Feature2D::IH_P90_VAL, Feature2D::IH_MEDIAN_VAL })
    {
        double v = ih_get(fv, fc);
        ASSERT_GE(v, 1.0);
        ASSERT_LE(v, 7.0);
    }
}

void test_ih_float_domain_reconstruction()
{
    Fsettings s = ih_make_settings(3, true);
    s[(int)NyxSetting::FPIMG_ACTIVE].bval = true;
    s[(int)NyxSetting::FPIMG_MIN].rval = 0.0;
    s[(int)NyxSetting::FPIMG_MAX].rval = 1.0;
    s[(int)NyxSetting::FPIMG_TARGET_DR].rval = 10.0;

    std::vector<std::vector<double>> fv;
    ih_run(fv, s, /*slide_idx*/ 0, /*fp_image*/ true, /*slide_min*/ 0.0, /*slide_max*/ 1.0);

    // integer-domain pixels {1,7} -> float {0.1,0.7}; domain features scale by 1/10
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), 0.1));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), 0.7));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 0.6));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 0.2));   // (0.7-0.1)/3
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_NUM_BINS), 3.0));
}

void test_ih_float_domain_reconstruction_negative_min()
{
    Fsettings s = ih_make_settings(3, true);
    s[(int)NyxSetting::FPIMG_ACTIVE].bval = true;
    s[(int)NyxSetting::FPIMG_MIN].rval = -1000.0;
    s[(int)NyxSetting::FPIMG_MAX].rval = 1000.0;
    s[(int)NyxSetting::FPIMG_TARGET_DR].rval = 10.0;

    std::vector<std::vector<double>> fv;
    ih_run(fv, s, /*slide_idx*/ 0, /*fp_image*/ true, /*slide_min*/ -1000.0, /*slide_max*/ 1000.0);

    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), -800.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), 400.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 1200.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 400.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_NUM_BINS), 3.0));
}

void test_ih_float_domain_reconstruction_preserve_hu()
{
    Fsettings s = ih_make_settings(3, true);   // FPIMG knobs irrelevant in HU mode
    std::vector<std::vector<double>> fv;
    ih_run(fv, s, /*slide_idx*/ 0, /*fp_image*/ false,
           /*slide_min*/ -1024.0, /*slide_max*/ 3071.0, /*preserve_hu*/ true);

    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), -1023.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), -1017.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 6.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 2.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_NUM_BINS), 3.0));
}

void test_ih_float_domain_reconstruction_preserve_hu_fpactive()
{
    Fsettings s = ih_make_settings(3, true);
    s[(int)NyxSetting::FPIMG_ACTIVE].bval = true;    // fp options supplied alongside --preserve-hu
    s[(int)NyxSetting::FPIMG_MIN].rval = 0.0;        // default --fpimgmin; must be ignored in HU mode
    s[(int)NyxSetting::FPIMG_MAX].rval = 1.0;
    s[(int)NyxSetting::FPIMG_TARGET_DR].rval = 10.0;
    std::vector<std::vector<double>> fv;
    ih_run(fv, s, /*slide_idx*/ 0, /*fp_image*/ false,
           /*slide_min*/ -1024.0, /*slide_max*/ 3071.0, /*preserve_hu*/ true);

    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), -1023.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), -1017.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 6.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 2.0));
}
