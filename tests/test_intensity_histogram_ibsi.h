#pragma once
#include <gtest/gtest.h>
#include <cmath>
#include <unordered_map>
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity_histogram.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// Provenance: IBSI (Zwanenburg et al. 2020, arXiv:1612.07003) §3.4 digital-phantom
// intensity-histogram consensus values. Discretisation config: FBN (fixed bin number)
// with GREYDEPTH=6, IBSI mode on. Index base: Nyxus reports IDX features in the
// 1-based grey-level convention, matching IBSI directly (no offset). Recorded in
// design doc §6.4. Values sourced in Task 1.
static const int IH_PHANTOM_NBINS = 6;
static std::unordered_map<std::string,double> ibsi_ih_phantom_golden = {
    {"VARIANCE_IDX", 3.05},
    {"SKEWNESS_IDX", 1.08},
    {"EXCESS_KURTOSIS_IDX", -0.355},
    {"INTERQUANTILE_RANGE_IDX", 3},
    {"RANGE_IDX", 5},
    {"MEAN_ABSOLUTE_DEVIATION_IDX", 1.55},
    {"ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX", 1.11},
    {"MEDIAN_ABSOLUTE_DEVIATION_IDX", 1.15},
    {"COEFFICIENT_OF_VARIATION_IDX", 0.812},
    {"QUANTILE_COEFFICIENT_OF_DISPERSION_IDX", 0.6},
    {"ENTROPY_IDX", 1.27},
    {"UNIFORMITY_IDX", 0.512},
};

static void ih_ibsi_run(std::vector<std::vector<double>>& fvals, int nbins) {
    std::vector<NyxusPixel> img, msk;
    for (auto* z : {ibsi_phantom_z1_intensity, ibsi_phantom_z2_intensity,
                    ibsi_phantom_z3_intensity, ibsi_phantom_z4_intensity})
        for (size_t i = 0; i < 20; i++) img.push_back(z[i]);
    for (auto* z : {ibsi_phantom_z1_mask, ibsi_phantom_z2_mask,
                    ibsi_phantom_z3_mask, ibsi_phantom_z4_mask})
        for (size_t i = 0; i < 20; i++) msk.push_back(z[i]);
    Dataset ds; ds.dataset_props.push_back(SlideProps("",""));
    LR roidata(1);
    Fsettings s; s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = nbins;
    s[(int)NyxSetting::IBSI].bval = true;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::SOFTNAN].rval = -7777.0;
    load_masked_test_roi_data(roidata, img.data(), msk.data(), img.size());
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));
    roidata.initialize_fvals(); f.save_value(roidata.fvals);
    fvals = roidata.fvals;
}

static double ihg(const std::vector<std::vector<double>>& fv, Nyxus::Feature2D fc){ return fv[(int)fc][0]; }

// IDX dispersion/index features vs IBSI intensity-histogram consensus (12 with IBSI values).
void test_ih_dispersion_ibsi() {
    using F = Nyxus::Feature2D;
    std::vector<std::vector<double>> fv;
    ih_ibsi_run(fv, IH_PHANTOM_NBINS);
    auto chk = [&](const char* key, F fc){
        double gt = ibsi_ih_phantom_golden[key];
        if (std::abs(gt) < 1e-12) ASSERT_NEAR(ihg(fv,fc), gt, 1e-9) << key;
        else ASSERT_TRUE(agrees_gt(ihg(fv,fc), gt, 100.)) << key;  // rel 1e-2 (IBSI phantom tier)
    };
    chk("VARIANCE_IDX",                          F::IH_VARIANCE_IDX);
    chk("SKEWNESS_IDX",                          F::IH_SKEWNESS_IDX);
    chk("EXCESS_KURTOSIS_IDX",                   F::IH_EXCESS_KURTOSIS_IDX);
    chk("INTERQUANTILE_RANGE_IDX",               F::IH_INTERQUANTILE_RANGE_IDX);
    chk("RANGE_IDX",                             F::IH_RANGE_IDX);
    chk("MEAN_ABSOLUTE_DEVIATION_IDX",           F::IH_MEAN_ABSOLUTE_DEVIATION_IDX);
    chk("ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX",    F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX);
    chk("MEDIAN_ABSOLUTE_DEVIATION_IDX",         F::IH_MEDIAN_ABSOLUTE_DEVIATION_IDX);
    chk("COEFFICIENT_OF_VARIATION_IDX",          F::IH_COEFFICIENT_OF_VARIATION_IDX);
    chk("QUANTILE_COEFFICIENT_OF_DISPERSION_IDX",F::IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX);
    chk("ENTROPY_IDX",                           F::IH_ENTROPY_IDX);
    chk("UNIFORMITY_IDX",                        F::IH_UNIFORMITY_IDX);
    // ROBUST_MEAN_IDX has no IBSI feature -> covered analytically in Task 4.

    // ---- VAL anchored to the IBSI-vetted IDX values (design §5) ----
    double b = ihg(fv, F::IH_BIN_SIZE);                 // binWidth
    // NOTE: IH_INTERQUANTILE_RANGE_VAL is intentionally NOT anchored here. Its IDX
    // counterpart floors the interpolated quantile via getIndexOf() while the _VAL is
    // continuous, so VAL != b*IDX (a filed Nyxus flooring bug). It is vetted
    // analytically in Task 4 instead.
    // pure-scale spreads: VAL = b * IDX
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_MEAN_ABSOLUTE_DEVIATION_VAL),
                          b*ihg(fv,F::IH_MEAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL),
                          b*ihg(fv,F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_MEDIAN_ABSOLUTE_DEVIATION_VAL),
                          b*ihg(fv,F::IH_MEDIAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    // CoV_VAL = std_VAL / mean_VAL = b*sqrt(VARIANCE_IDX) / MEAN_VAL  (VARIANCE_IDX = IBSI anchor)
    double cov_val_expected = b*std::sqrt(ihg(fv,F::IH_VARIANCE_IDX)) / ihg(fv,F::IH_MEAN_VAL);
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_COEFFICIENT_OF_VARIATION_VAL), cov_val_expected, 1e4));
}

// 17-px discriminating fixture: N=5 grey levels, freq {1,5,6,4,1}. The robust
// window [p10Idx,p90Idx] strictly trims the tail bins (first + last), so the
// robust statistics differ from the full-histogram statistics on this fixture.
// Goldens below are hand-computed (carried from PR 367), derived independently
// of intensity_histogram.cpp -- covers the 4 oracle=analytic features that have
// no clean IBSI/VAL-vs-IDX anchor: the robust-mean pair (no IBSI feature exists
// for ROBUST_MEAN_IDX/VAL) and QCoD_VAL/IQR_VAL (not IBSI-anchorable; see notes
// in test_ih_dispersion_ibsi() above).
static const NyxusPixel intensityHistogramRobustData[] = {
    {0,0,0},
    {1,0,10},{2,0,10},{3,0,10},{4,0,10},{5,0,10},
    {6,0,20},{7,0,20},{8,0,20},{9,0,20},{10,0,20},{11,0,20},
    {12,0,30},{13,0,30},{14,0,30},{15,0,30},
    {16,0,40}
};

void test_ih_dispersion_robust_analytic() {
    using F = Nyxus::Feature2D;
    Fsettings s; s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = 5;
    s[(int)NyxSetting::IBSI].bval = true;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::SOFTNAN].rval = -7777.0;
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
    // IQR_VAL: not IBSI-anchorable (IQR_IDX bin-floored vs IQR_VAL interpolated) -> analytic golden here.
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_INTERQUANTILE_RANGE_VAL][0], 12.3, 1e4)); // oracle=analytic
}
