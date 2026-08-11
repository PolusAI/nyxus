#pragma once

// Closed-form (analytic) intensity-histogram assertions: the fixtures are small enough that every
// expected value is hand-derivable, and the derivations are written out above each test. These are
// oracle assertions in the SPEC 4 `analytic` sense - no external tool is involved, so no tool version
// applies - and they were split out of test_2d_intensity_histogram_ibsi.h, whose goldens are IBSI
// consensus values on the IBSI digital phantom (SPEC 2: one kind per file).
//
// oracle_coverage.csv names this file in target_test for 26 of the family's rows; the 2 assertions
// here cover a subset of them, and the rest are still to be written.

#include "test_2d_intensity_histogram_regression.h"   // shared fixture: run_intensity_histogram_fixture and the IH settings

// 17-px discriminating fixture: N=5 grey levels, freq {1,5,6,4,1}. The robust
// window [p10Idx,p90Idx] strictly trims the tail bins (first + last), so the
// robust statistics differ from the full-histogram statistics on this fixture.
// Goldens below are hand-computed (carried from PR 367), derived independently
// of intensity_histogram.cpp -- covers the 4 oracle=analytic features that have
// no clean IBSI/VAL-vs-IDX anchor: the robust-mean pair (no IBSI feature exists
// for ROBUST_MEAN_IDX/VAL) and QCoD_VAL/IQR_VAL (not IBSI-anchorable; see notes
// in test_2d_intensity_histogram_dispersion_ibsi() above).
static const NyxusPixel intensityHistogramRobustData[] = {
    {0,0,0},
    {1,0,10},{2,0,10},{3,0,10},{4,0,10},{5,0,10},
    {6,0,20},{7,0,20},{8,0,20},{9,0,20},{10,0,20},{11,0,20},
    {12,0,30},{13,0,30},{14,0,30},{15,0,30},
    {16,0,40}
};

void test_2d_intensity_histogram_dispersion_robust_analytic() {
    using F = Nyxus::Feature2D;
    Fsettings s = ih_make_settings(5, /*ibsi*/ true);
    Dataset ds; ds.dataset_props.push_back(SlideProps("",""));
    LR roidata(100); roidata.slide_idx = -1;
    load_test_roi_data(roidata, intensityHistogramRobustData,
                       sizeof(intensityHistogramRobustData)/sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f; ASSERT_NO_THROW(f.calculate(roidata, s, ds));
    roidata.initialize_fvals(); f.save_value(roidata.fvals);
    auto& fv = roidata.fvals;
    // Hand-computed goldens (robust window strictly trims tails -> robust != full):
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL][0], 4.977777778, 1e4));
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_VAL][0], 19.46666667, 1e4));         // oracle=analytic
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX][0], 0.6222222222, 1e4));
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_IDX][0], 1.933333333, 1e4));         // oracle=analytic
    // QCoD_VAL: not IBSI-anchorable (needs unexposed P25/P75 sum) -> analytic golden here.
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL][0], 0.3178294574, 1e4)); // oracle=analytic
    // IQR_VAL: no IBSI anchor -- IDX is the IBSI discrete percentile, VAL interpolates, so they
    // differ by design (not a defect); vetted analytically here.
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_INTERQUANTILE_RANGE_VAL][0], 12.3, 1e4)); // oracle=analytic
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

    static const double expected[N] = { 2.0, 1.0, 2.0 };
    for (int k = 0; k < N; k++)
        ASSERT_TRUE(agrees_gt(hist[k], expected[k], 1e4)) << "bin " << k;
}
