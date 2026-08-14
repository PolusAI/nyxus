#pragma once

// Closed-form (analytic) intensity-histogram assertions: oracle assertions in the SPEC 4 `analytic`
// sense, so no external tool and no tool version applies. Each expected value is either derived in
// the comment above its test or emitted by
// tests/vetting/oracles/gen_intensity_histogram_analytic.py.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../src/nyx/dataset.h"                       // Dataset, SlideProps
#include "../src/nyx/feature_settings.h"              // Fsettings
#include "../src/nyx/featureset.h"                    // Feature2D
#include "../src/nyx/features/intensity.h"            // PixelIntensityFeatures
#include "../src/nyx/features/intensity_histogram.h"  // IntensityHistogramFeatures
#include "../src/nyx/roi_cache.h"                     // LR
#include "test_2d_intensity_histogram_common.h"       // the IBSI phantom fixture at recipe ih.ibsi_fbn
#include "test_2d_intensity_histogram_regression.h"   // ih_make_settings, intensityHistogramTestData
#include "test_main_nyxus.h"                          // agrees_gt, load_test_roi_data
#include "test_ref_vals.h"                            // ref_vals_map

// The robust-window statistics on the tail-trimming fixture (defined in the regression header,
// which also drift-guards the percentile features it exercises). At 5 bins over [0,40] the bin
// counts are {1,5,6,4,1}, so binWidth=8 and the [P10,P90] window is bins 2..4 - 15 of the 17
// voxels, both tail bins strictly trimmed, which the phantom does not do at the low end.
// IDX is the mean of their 1-based bin indices, (2*5+3*6+4*4)/15 = 44/15; VAL is the mean of their
// bin centres, (12*5+20*6+28*4)/15 = 292/15 - the same number through the family's centre map,
// VAL = lo + (IDX - 0.5) * binWidth.
static const ref_vals_map<double> intensity_histogram_2d_analytic_robust_ref_vals
{
    {"IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX", 0.6222222222},
    {"IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL", 4.977777778},
    {"IH_ROBUST_MEAN_IDX", 2.933333333},
    {"IH_ROBUST_MEAN_VAL", 19.46666667},
};

void test_2d_intensity_histogram_dispersion_robust_analytic() {
    Fsettings s = ih_make_settings(5, /*ibsi*/ true);
    Dataset ds; ds.dataset_props.push_back(SlideProps("",""));
    LR roidata(100); roidata.slide_idx = -1;
    load_test_roi_data(roidata, intensityHistogramRobustData,
                       sizeof(intensityHistogramRobustData)/sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f; ASSERT_NO_THROW(f.calculate(roidata, s, ds));
    roidata.initialize_fvals(); f.save_value(roidata.fvals);

    for (const auto& [feature_name, golden] : intensity_histogram_2d_analytic_robust_ref_vals)
    {
        double value = 0;
        ASSERT_TRUE(read_2d_intensity_histogram_feature(roidata.fvals, feature_name, value)) << feature_name;
        ASSERT_TRUE(agrees_gt(value, golden, 1e4)) << feature_name;
    }
}

// ---------------------------------------------------------------------------------------------------
// Analytic vetting of Feature2D::HISTOGRAM -- the multi-valued (vector) per-ROI intensity
// histogram computed by PixelIntensityFeatures (src/nyx/features/intensity.cpp:
// val_HISTOGRAM = H.get_cust_frequencies(n_greybins), saved at fvals[(int)Feature2D::HISTOGRAM]
// in intensity.cpp:402). This is a distinct feature class/registry family from the scalar
// IntensityHistogramFeatures (IH_*) tested above; it is opt-in (in *ALL*, not *ALL_INTENSITY*,
// see PixelIntensityFeatures::PixelIntensityFeatures()) and NOT IBSI-gated.
//
// Binning contract, read from src/nyx/features/histogram.h (TrivialHistogram::initialize /
// get_cust_frequencies) and src/nyx/helpers/helpers.h (Nyxus::to_grayscale):
//   - N = n_cust_bins equal-width bins spanning [minVal_, maxVal_] of the ROI (bin width =
//     range / N).
//   - Bin index of intensity i = floor((i - min) / range * N)  [Nyxus::to_grayscale with
//     disable_binning=false: pi = (i-min)/range*N ; new_pi = (unsigned int)pi, i.e. truncation
//     toward 0 == floor for non-negative pi].
//   - The internal accumulator has N+1 slots; a value whose floored index lands exactly on N
//     (i.e. i == max, since (max-min)/range*N == N) falls into that extra slot, which is then
//     folded into bin N-1 ("Fix the special last bin": bins_cust_[N-1] += bins_cust_[N]).
//     So the histogram is top-inclusive: bin N-1 covers [min + (N-1)*range/N, max].
//   - get_cust_frequencies(N) returns the raw integer bin COUNTS (not normalized frequencies /
//     probabilities), trimmed to exactly N entries (the folded N-th slot is dropped after the
//     fold-in above).
//   - For a plain (non-IBSI) run, n_greybins = STNGS_NGREYS(settings) i.e. the GREYDEPTH
//     setting (falling back to DEFAULT_NUM_HISTO_BINS=24 only when settings are entirely
//     unpopulated) -- see intensity.cpp:157.
//
// Fixture: intensityHistogramTestData = {1,1,3,5,7} (5 px), N=3 bins over [min=1,max=7]
// (range=6, binWidth=2). Bin assignment via floor((i-1)/6*3):
//   i=1 -> floor(0/6*3)=floor(0.0) = 0 -> bin0
//   i=1 -> floor(0/6*3)=floor(0.0) = 0 -> bin0
//   i=3 -> floor(2/6*3)=floor(1.0) = 1 -> bin1
//   i=5 -> floor(4/6*3)=floor(2.0) = 2 -> bin2
//   i=7 -> floor(6/6*3)=floor(3.0) = 3 -> folds into bin (N-1)=2
// Expected per-bin COUNTS: bin0=2 (the two 1's), bin1=1 (the 3), bin2=2 (the 5 and the folded 7).
//   => expected = [2, 1, 2]   (sum = 5 = population; these are raw counts, not probabilities)
void test_2d_intensity_histogram_bin_counts_analytic()
{
    using F = Nyxus::Feature2D;
    const int N = 3;
    // HISTOGRAM is not IBSI-gated; IBSI is left off here to keep the test independent of that gate.
    Fsettings s = ih_make_settings(N, /*ibsi*/ false);

    Dataset ds;
    ds.dataset_props.push_back(SlideProps("", ""));

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1;
    load_test_roi_data(roidata, intensityHistogramTestData,
                       sizeof(intensityHistogramTestData) / sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();

    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    const auto& hist = roidata.fvals[(int)F::HISTOGRAM];
    ASSERT_EQ(hist.size(), (size_t)N);

    // a plain array rather than a ref_vals_map: HISTOGRAM is multi-valued, so there is no one
    // scalar to key by feature name - the golden is the per-bin vector itself
    static const double expected[N] = { 2.0, 1.0, 2.0 };
    for (int k = 0; k < N; k++)
        ASSERT_TRUE(agrees_gt(hist[k], expected[k], 1e4)) << "bin " << k;
}

// ---------------------------------------------------------------------------------------------
// The _VAL family plus IH_ROBUST_MEAN_IDX on the IBSI phantom, at recipe ih.ibsi_fbn (fixed bin
// number, 6 bins).
//
// No external tool reports these: MIRP and PyRadiomics both stay in the discretised domain, which
// is the _IDX family (vetted in test_2d_intensity_histogram_mirp.h), and the mean over the
// [P10,P90]-trimmed histogram is a Nyxus feature neither tool nor the IBSI reference set carries.
// _VAL takes the same statistics back into the intensity domain, by one of three maps:
//
//   bin centre       MEAN, MEDIAN, MODE, ROBUST_MEAN, VARIANCE, the deviation measures and COV -
//                    the statistic over the centre of the bin each voxel falls in,
//                    c = lo + (i - 0.5) * binsize.
//   domain-invariant SKEWNESS, EXCESS_KURTOSIS, ENTROPY, UNIFORMITY - unchanged by that map, or a
//                    function of the bin counts alone, so _VAL equals _IDX and MIRP vets both.
//   intensity        MINIMUM, MAXIMUM, RANGE - the untouched voxel values.
//   domain
//
// The percentile features (P10, P90, IQR, QCoD) are NOT here: they carry no oracle and are pinned
// as drift guards in test_2d_intensity_histogram_regression.h instead.
//
// Goldens generated offline by tests/vetting/oracles/gen_intensity_histogram_analytic.py, which
// evaluates each closed form from the published definitions rather than from
// intensity_histogram.cpp. Evidence:
// tests/vetting/audit/intensity_histogram_2d_analytic_vetting_report.md.
static ref_vals_map<double> intensity_histogram_2d_analytic_phantom_ref_vals
{
    {"IH_BIN_SIZE", 0.83333333333333337},
    {"IH_COEFFICIENT_OF_VARIATION_VAL", 0.61261603178456026},
    {"IH_ENTROPY_VAL", 1.2656115555865246},
    {"IH_EXCESS_KURTOSIS_VAL", -0.35462048068783236},
    {"IH_MAXIMUM_VAL", 6},
    {"IH_MEAN_ABSOLUTE_DEVIATION_VAL", 1.2935232529827125},
    {"IH_MEAN_VAL", 2.3738738738738738},
    {"IH_MEDIAN_ABSOLUTE_DEVIATION_VAL", 0.95720720720720731},
    {"IH_MEDIAN_VAL", 1.4166666666666667},
    {"IH_MINIMUM_VAL", 1},
    {"IH_MODE_VAL", 1.4166666666666667},
    {"IH_NUM_BINS", 6},
    {"IH_RANGE_VAL", 5},
    {"IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL", 0.92819484666221119},
    {"IH_ROBUST_MEAN_IDX", 1.7462686567164178},
    {"IH_ROBUST_MEAN_VAL", 2.0385572139303481},
    {"IH_SKEWNESS_VAL", 1.0838207225574572},
    {"IH_UNIFORMITY_VAL", 0.51241782322863394},
    {"IH_VARIANCE_VAL", 2.114910518626735},
};

void test_2d_intensity_histogram_phantom_analytic()
{
    std::vector<std::vector<double>> fvals;
    calc_2d_intensity_histogram_phantom (fvals);

    for (const auto& [feature_name, golden] : intensity_histogram_2d_analytic_phantom_ref_vals)
    {
        double value = 0;
        ASSERT_TRUE (read_2d_intensity_histogram_feature (fvals, feature_name, value)) << feature_name;
        // the closed forms reproduce Nyxus to 7.5e-15 at worst, so 1e-9 is a wide margin over the
        // floating-point noise and far tighter than any convention difference would be
        ASSERT_TRUE (agrees_gt (value, golden, 1.e9)) << feature_name;
    }
}
