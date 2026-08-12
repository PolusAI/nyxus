#pragma once
#include <gtest/gtest.h>
#include <cmath>
#include <unordered_map>
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity_histogram.h"
#include "../src/nyx/features/intensity.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_2d_intensity_histogram_regression.h"
#include "test_ref_vals.h"

// Provenance: IBSI (Zwanenburg et al. 2020, arXiv:1612.07003) §3.4 digital-phantom
// intensity-histogram consensus values. Discretisation config: FBN (fixed bin number)
// with GREYDEPTH=6, IBSI mode on. Index base: Nyxus reports IDX features in the
// 1-based grey-level convention, matching IBSI directly (no offset). Recorded in
// design doc §6.4. Values sourced in Task 1.
static const int IH_PHANTOM_NBINS = 6;
static const ref_vals_map<double> intensity_histogram_2d_ibsi_ref_vals = {
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

static void run_intensity_histogram_ibsi_fixture(std::vector<std::vector<double>>& fvals, int nbins) {
    std::vector<NyxusPixel> img, msk;
    for (auto* z : {ibsi_phantom_z1_intensity, ibsi_phantom_z2_intensity,
                    ibsi_phantom_z3_intensity, ibsi_phantom_z4_intensity})
        for (size_t i = 0; i < 20; i++) img.push_back(z[i]);
    for (auto* z : {ibsi_phantom_z1_mask, ibsi_phantom_z2_mask,
                    ibsi_phantom_z3_mask, ibsi_phantom_z4_mask})
        for (size_t i = 0; i < 20; i++) msk.push_back(z[i]);
    Dataset ds; ds.dataset_props.push_back(SlideProps("",""));
    LR roidata(1);
    Fsettings s = ih_make_settings(nbins, /*ibsi*/ true);
    load_masked_test_roi_data(roidata, img.data(), msk.data(), img.size());
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));
    roidata.initialize_fvals(); f.save_value(roidata.fvals);
    fvals = roidata.fvals;
}

// IDX dispersion/index features vs IBSI intensity-histogram consensus (12 with IBSI values).
void test_2d_intensity_histogram_dispersion_ibsi() {
    using F = Nyxus::Feature2D;
    std::vector<std::vector<double>> fv;
    run_intensity_histogram_ibsi_fixture(fv, IH_PHANTOM_NBINS);
    auto chk = [&](const char* key, F fc){
        double gt = intensity_histogram_2d_ibsi_ref_vals.at(key);
        if (std::abs(gt) < 1e-12) ASSERT_NEAR(ih_get(fv,fc), gt, 1e-9) << key;
        else ASSERT_TRUE(agrees_gt(ih_get(fv,fc), gt, 100.)) << key;  // rel 1e-2 (IBSI phantom tier)
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
    double b = ih_get(fv, F::IH_BIN_SIZE);                 // binWidth
    // NOTE: IH_INTERQUANTILE_RANGE_VAL is intentionally NOT anchored here, and this is
    // BY DESIGN, not a bug. The IQR/QCoD _IDX variants use the IBSI *discrete* grey-level
    // percentile (getIndexOf picks the CDF-crossing bin: P25=1, P75=4 -> IQR_IDX=3, which
    // matches the IBSI reference; IBSI names the discrete P90=4 as reference and the
    // interpolated 4.2 as a non-reference variant). _VAL interpolates within the bin -- a
    // Nyxus continuous extension with no IBSI counterpart. A discrete percentile is a step
    // function of the CDF, so VAL != b*IDX inherently (forcing IDX continuous would give
    // ~2.91 and break the IBSI oracle of 3). IQR_VAL is vetted analytically instead.
    // pure-scale spreads: VAL = b * IDX
    ASSERT_TRUE(agrees_gt(ih_get(fv,F::IH_MEAN_ABSOLUTE_DEVIATION_VAL),
                          b*ih_get(fv,F::IH_MEAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    ASSERT_TRUE(agrees_gt(ih_get(fv,F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL),
                          b*ih_get(fv,F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    ASSERT_TRUE(agrees_gt(ih_get(fv,F::IH_MEDIAN_ABSOLUTE_DEVIATION_VAL),
                          b*ih_get(fv,F::IH_MEDIAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    // CoV_VAL = std_VAL / mean_VAL = b*sqrt(VARIANCE_IDX) / MEAN_VAL  (VARIANCE_IDX = IBSI anchor)
    double cov_val_expected = b*std::sqrt(ih_get(fv,F::IH_VARIANCE_IDX)) / ih_get(fv,F::IH_MEAN_VAL);
    ASSERT_TRUE(agrees_gt(ih_get(fv,F::IH_COEFFICIENT_OF_VARIATION_VAL), cov_val_expected, 1e4));
}

