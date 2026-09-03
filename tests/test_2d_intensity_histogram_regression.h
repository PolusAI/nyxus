#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "../src/nyx/dataset.h"                       // Dataset, SlideProps
#include "../src/nyx/feature_settings.h"              // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"                    // Feature2D
#include "../src/nyx/features/intensity_histogram.h"  // IntensityHistogramFeatures
#include "../src/nyx/features/pixel.h"                // NyxusPixel
#include "../src/nyx/roi_cache.h"                     // LR
#include "test_2d_intensity_histogram_common.h"       // the IBSI phantom fixture at recipe ih.ibsi_fbn
#include "test_main_nyxus.h"                          // agrees_gt, load_test_roi_data
#include "test_ref_vals.h"                            // ref_vals_map

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
// Builds the fp-image options a --fpimgmin/max/dr invocation would leave behind.
static FpImageOptions ih_make_fp_options(double fpmin, double fpmax, int dr)
{
    FpImageOptions o;
    o.raw_min_intensity = std::to_string(fpmin);
    o.raw_max_intensity = std::to_string(fpmax);
    o.raw_target_dyn_range = std::to_string(dr);
    EXPECT_TRUE(o.parse_input());
    return o;
}

static void run_intensity_histogram_fixture(std::vector<std::vector<double>>& fvals,
                   const Fsettings& s,
                   int slide_idx = -1,
                   bool fp_image = false,
                   double slide_min = -1.0,
                   double slide_max = -1.0,
                   bool preserve_hu = false,
                   const FpImageOptions* fpo = nullptr)
{
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("", ""));
    if (slide_idx >= 0)
    {
        SlideProps& sp = ds.dataset_props[slide_idx];
        sp.fname_int = "slide.tif";        // the recorder picks the loader from the extension
        sp.fp_phys_pivoxels = fp_image;
        sp.preserve_hu = preserve_hu;
        sp.min_preroi_inten = slide_min;
        sp.max_preroi_inten = slide_max;
        sp.min_allpix_inten = slide_min;
        FpImageOptions dflt;
        // the load-time map the loader would have applied, recorded as the scan records it
        Nyxus::record_intensity_domain_map (sp, fpo ? *fpo : dflt);
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
void test_2d_intensity_histogram_integer_domain_values_regression()
{
    std::vector<std::vector<double>> fv;
    run_intensity_histogram_fixture(fv, ih_make_settings(/*nbins*/ 3, /*ibsi*/ true));

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
void test_2d_intensity_histogram_index_and_percentile_bounds_regression()
{
    const int N = 3;
    std::vector<std::vector<double>> fv;
    run_intensity_histogram_fixture(fv, ih_make_settings(N, true));

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

// 4) Float-domain reconstruction: a float image with fpimg [0,1] / DR=10 rescales
//    the stored uint v -> v/10 (mirroring the load-time quantization). The domain
//    features (min/max/range/bin-size), which are derived directly from the
//    reconstructed [minVal,maxVal], must scale by exactly 1/10 vs the integer run.
//    (Per-bin features like mean/entropy can shift when a pixel sits exactly on a
//    bin boundary because float binning is not bit-exact there — that is an inherent
//    floating-point effect, not a domain-mapping error, so it is not asserted here.)
void test_2d_intensity_histogram_float_domain_regression()
{
    Fsettings s = ih_make_settings(3, true);
    FpImageOptions fpo = ih_make_fp_options(0.0, 1.0, 10);

    std::vector<std::vector<double>> fv;
    run_intensity_histogram_fixture(fv, s, /*slide_idx*/ 0, /*fp_image*/ true, /*slide_min*/ 0.0, /*slide_max*/ 1.0,
           /*preserve_hu*/ false, &fpo);

    // integer-domain pixels {1,7} -> float {0.1,0.7}; domain features scale by 1/10
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), 0.1));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), 0.7));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 0.6));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 0.2));   // (0.7-0.1)/3
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_NUM_BINS), 3.0));
}

// 4b) Negative-domain (CT / Hounsfield-Unit-like) reconstruction. The float image
//     spans [fpmin=-1000, fpmax=1000] with DR=10, so the load-time map is
//       u = DR*(x-fpmin)/(fpmax-fpmin) = (x+1000)/200,
//     and float_domain_map must invert it with pscale=(fpmax-fpmin)/DR=200,
//     poffset=fpmin=-1000, i.e. reported = -1000 + 200*u. Stored ROI mn/mx = 1/7:
//       IH_MINIMUM = -1000 + 200*1 = -800 ; IH_MAXIMUM = -1000 + 200*7 = 400
//       IH_RANGE = 1200 ; IH_BIN_SIZE = 1200/3 = 400
//     This exercises the reconstruction with a NEGATIVE offset — the exact regime
//     Hounsfield-Unit preservation targets (water=0, air=-1000) and which the
//     original fpmin=0 test never covered.
void test_2d_intensity_histogram_float_domain_negative_min_regression()
{
    Fsettings s = ih_make_settings(3, true);
    FpImageOptions fpo = ih_make_fp_options(-1000.0, 1000.0, 10);

    std::vector<std::vector<double>> fv;
    run_intensity_histogram_fixture(fv, s, /*slide_idx*/ 0, /*fp_image*/ true, /*slide_min*/ -1000.0, /*slide_max*/ 1000.0,
           /*preserve_hu*/ false, &fpo);

    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), -800.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), 400.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 1200.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 400.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_NUM_BINS), 3.0));
}

// 4c) Offset-preserving reconstruction. With preserve_hu the load-time map is the
//     slope-1 offset u = x - floor(min), so float_domain_map must invert it as
//     reported = floor(min) + u (pscale=1). With min=-1024 and stored ROI mn/mx=1/7:
//       IH_MINIMUM = -1024 + 1 = -1023 ; IH_MAXIMUM = -1024 + 7 = -1017
//       IH_RANGE = 6 ; IH_BIN_SIZE = 6/3 = 2  (integer grey spacing preserved)
//     i.e. features are reported back in absolute Hounsfield units.
void test_2d_intensity_histogram_float_domain_preserve_hu_regression()
{
    Fsettings s = ih_make_settings(3, true);   // FPIMG knobs irrelevant in HU mode
    std::vector<std::vector<double>> fv;
    run_intensity_histogram_fixture(fv, s, /*slide_idx*/ 0, /*fp_image*/ false,
           /*slide_min*/ -1024.0, /*slide_max*/ 3071.0, /*preserve_hu*/ true);

    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), -1023.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), -1017.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 6.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 2.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_NUM_BINS), 3.0));
}

// 4d) Regression: preserve_hu combined with active fp-image options
//     (--fpimgmin/max/dr). On the offset map the base is ALWAYS the slide's own
//     minimum, so the recorder must IGNORE the fp min. Here fp options are supplied
//     with a misleading min of 0 (the value --fpimgmin defaults to), yet the
//     reconstruction must still recover absolute HU from slide_min=-1024:
//       IH_MINIMUM = -1024 + 1 = -1023.
//     Taking the offset from the fp min instead would give 0 + 1 = 1, shifting every
//     value up by 1024 and clamping negative HU to 0 at load time.
void test_2d_intensity_histogram_float_domain_preserve_hu_fpactive_regression()
{
    Fsettings s = ih_make_settings(3, true);
    FpImageOptions fpo = ih_make_fp_options(0.0, 1.0, 10);   // supplied alongside --preserve-hu
    std::vector<std::vector<double>> fv;
    run_intensity_histogram_fixture(fv, s, /*slide_idx*/ 0, /*fp_image*/ false,
           /*slide_min*/ -1024.0, /*slide_max*/ 3071.0, /*preserve_hu*/ true, &fpo);

    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MINIMUM_VAL), -1023.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_MAXIMUM_VAL), -1017.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_RANGE_VAL), 6.0));
    ASSERT_TRUE(agrees_gt(ih_get(fv, Feature2D::IH_BIN_SIZE), 2.0));
}

// ---------------------------------------------------------------------------
// 5) Drift guards for the percentile-domain _VAL features, which claim no oracle.
//
// IH_P10_VAL, IH_P90_VAL and the IQR/QCoD built from P25/P75 use the grouped-data percentile
// L + binWidth*(n*p - F)/f, over the bin the CDF crosses. No reference implementation reproduces
// it: on the IBSI phantom at 6 bins Nyxus reports P90_VAL = 4.3125, while all nine of numpy's and
// all nine of Octave's native percentile methods answer between 4.0 and 5.0 - and their default,
// 4.0, is what the _IDX half and MIRP report. The values below are therefore pinned to catch
// drift, not to claim agreement with anything. Measurements:
// tests/vetting/audit/intensity_histogram_2d_analytic_vetting_report.md.
static const ref_vals_map<double> intensity_histogram_2d_regression_ref_vals
{
    {"IH_P10_VAL", 1.1233333333333333},
    {"IH_P90_VAL", 4.3125},
    {"IH_INTERQUANTILE_RANGE_VAL", 2.4260416666666664},
    {"IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL", 0.48109894649865725},
};

void test_2d_intensity_histogram_phantom_percentile_regression()
{
    std::vector<std::vector<double>> fvals;
    calc_2d_intensity_histogram_phantom(fvals);

    for (const auto& [feature_name, golden] : intensity_histogram_2d_regression_ref_vals)
    {
        double value = 0;
        ASSERT_TRUE(read_2d_intensity_histogram_feature(fvals, feature_name, value)) << feature_name;
        ASSERT_TRUE(agrees_gt(value, golden, 1.e9)) << feature_name;
    }
}

// The tail-trimming fixture: 17 px over [0,40] at 5 bins, counts {1,5,6,4,1}, so binWidth=8 and
// the [P10,P90] window is bins 2..4 - both tail bins strictly trimmed, which the phantom does not
// do at the low end. Its robust-mean statistics are vetted analytically in
// test_2d_intensity_histogram_analytic.h; the two percentile values below are drift guards for the
// same reason as the phantom set above.
static const NyxusPixel intensityHistogramRobustData[] = {
    {0,0,0},
    {1,0,10},{2,0,10},{3,0,10},{4,0,10},{5,0,10},
    {6,0,20},{7,0,20},{8,0,20},{9,0,20},{10,0,20},{11,0,20},
    {12,0,30},{13,0,30},{14,0,30},{15,0,30},
    {16,0,40}
};

static const ref_vals_map<double> intensity_histogram_2d_regression_robust_ref_vals
{
    {"IH_INTERQUANTILE_RANGE_VAL", 12.3},
    {"IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL", 0.3178294574},
};

void test_2d_intensity_histogram_dispersion_percentile_regression()
{
    Fsettings s = ih_make_settings(5, /*ibsi*/ true);
    Dataset ds; ds.dataset_props.push_back(SlideProps("", ""));
    LR roidata(100); roidata.slide_idx = -1;
    load_test_roi_data(roidata, intensityHistogramRobustData,
                       sizeof(intensityHistogramRobustData) / sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));
    roidata.initialize_fvals(); f.save_value(roidata.fvals);

    for (const auto& [feature_name, golden] : intensity_histogram_2d_regression_robust_ref_vals)
    {
        double value = 0;
        ASSERT_TRUE(read_2d_intensity_histogram_feature(roidata.fvals, feature_name, value)) << feature_name;
        ASSERT_TRUE(agrees_gt(value, golden, 1e4)) << feature_name;
    }
}

