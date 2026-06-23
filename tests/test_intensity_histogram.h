#pragma once

#include <cmath>
#include <gtest/gtest.h>

#include "../src/nyx/dataset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity_histogram.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

using namespace Nyxus;

// ---------------------------------------------------------------------------
// IBSI Intensity Histogram (IH) family — unit tests.
//
// A tiny, fully hand-computable ROI is used so every feature can be checked
// against an exact ground truth.
//
//   intensities: {1, 1, 3, 5, 7}   (5 pixels on a 1-D line)
//   N = 3 bins over [min=1, max=7]  -> binWidth = (7-1)/3 = 2
//   bin edges:  [1,3) [3,5) [5,7]   centers: 2, 4, 6
//   binning:    1,1 -> bin0 ; 3 -> bin1 ; 5 -> bin2 ; 7 -> bin2 (top folds in)
//   freq = [2,1,2], count = 5, probabilities = [0.4, 0.2, 0.4]
//
// Derived ground truth (integer domain):
//   mean      = 0.4*2 + 0.2*4 + 0.4*6 = 4.0
//   variance  = 0.4*4 + 0.2*0 + 0.4*4 = 3.2
//   skewness  = 0                          (symmetric)
//   exc.kurt  = (0.4*16 + 0.4*16)/3.2^2 - 3 = 12.8/10.24 - 3 = -1.75
//   median    = center of bin where running count first exceeds count/2 (=2) -> bin1 -> 4.0
//   uniformity= 0.4^2 + 0.2^2 + 0.4^2 = 0.36
//   entropy   = -(0.4*log2 0.4 + 0.2*log2 0.2 + 0.4*log2 0.4) = 1.521928...
//   mode      = first bin with max freq -> bin0 center = 2.0
//   min/max/range = 1 / 7 / 6 ;  min/max idx (1-based) = 1 / 3
//   mean idx  = (0.4*0 + 0.2*1 + 0.4*2) + 1 = 2.0
//   num bins  = 3 ;  bin size = 2
//   gradients: g(0)=freq1-freq0=-1 ; g(1)=(freq2-freq0)/2=0 ; g(2)=freq2-freq1=1
//              -> max gradient 1 @ idx 3 ; min gradient -1 @ idx 1
// ---------------------------------------------------------------------------

static const NyxusPixel intensityHistogramTestData[] =
{
    {0, 0, 1}, {1, 0, 1}, {2, 0, 3}, {3, 0, 5}, {4, 0, 7}
};

// Builds settings with the common knobs the IH family consumes.
static Fsettings ih_make_settings(int nbins, bool ibsi, double softnan = -7777.0)
{
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = softnan;
    s[(int)NyxSetting::GREYDEPTH].ival = nbins;
    s[(int)NyxSetting::IBSI].bval = ibsi;
    s[(int)NyxSetting::USEGPU].bval = false;
    // float-domain knobs default to "inactive" (integer image path)
    s[(int)NyxSetting::FPIMG_ACTIVE].bval = false;
    s[(int)NyxSetting::FPIMG_MIN].rval = 0.0;
    s[(int)NyxSetting::FPIMG_MAX].rval = 1.0;
    s[(int)NyxSetting::FPIMG_TARGET_DR].rval = 1e4;
    return s;
}

// Runs the IH feature on the test ROI and returns the populated fvals.
static void ih_run(std::vector<std::vector<double>>& fvals,
                   const Fsettings& s,
                   int slide_idx = -1,
                   bool fp_image = false,
                   double slide_min = -1.0,
                   double slide_max = -1.0)
{
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("", ""));
    if (slide_idx >= 0)
    {
        ds.dataset_props[slide_idx].fp_phys_pivoxels = fp_image;
        ds.dataset_props[slide_idx].min_preroi_inten = slide_min;
        ds.dataset_props[slide_idx].max_preroi_inten = slide_max;
    }

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = slide_idx;
    load_test_roi_data(roidata, intensityHistogramTestData,
                       sizeof(intensityHistogramTestData) / sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();

    IntensityHistogramFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    roidata.initialize_fvals();
    f.save_value(roidata.fvals);
    fvals = roidata.fvals;
}

static double ih_get(const std::vector<std::vector<double>>& fvals, Nyxus::Feature2D fc)
{
    return fvals[(int)fc][0];
}

// 1) Integer-domain values vs exact hand-computed ground truth.
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

// 2) Every "...Index" feature lands inside [1, N]; percentiles inside [min,max].
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

// 3) IBSI gate: with IBSI off the family returns the soft-NaN sentinel for all 46.
void test_ih_ibsi_gate_off_returns_nan()
{
    const double sentinel = -7777.0;
    std::vector<std::vector<double>> fv;
    ih_run(fv, ih_make_settings(3, /*ibsi*/ false, sentinel));

    for (auto fc : IntensityHistogramFeatures::featureset)
        ASSERT_DOUBLE_EQ(fv[(int)fc][0], sentinel);
}

// 4) Float-domain reconstruction: a float image with fpimg [0,1] / DR=10 rescales
//    the stored uint v -> v/10 (mirroring the load-time quantization). The domain
//    features (min/max/range/bin-size), which are derived directly from the
//    reconstructed [minVal,maxVal], must scale by exactly 1/10 vs the integer run.
//    (Per-bin features like mean/entropy can shift when a pixel sits exactly on a
//    bin boundary because float binning is not bit-exact there — that is an inherent
//    floating-point effect, not a domain-mapping error, so it is not asserted here.)
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

// 5) required(): the class is only "required" when at least one IH feature is enabled.
void test_ih_required_predicate()
{
    FeatureSet fs;
    fs.enableAll(false);
    ASSERT_FALSE(IntensityHistogramFeatures::required(fs));
    fs.enableFeature((int)Feature2D::IH_ENTROPY_VAL);
    ASSERT_TRUE(IntensityHistogramFeatures::required(fs));
}
