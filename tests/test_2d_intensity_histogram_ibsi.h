#pragma once
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/nyx/featureset.h"                   // Feature2D
#include "test_2d_intensity_histogram_common.h"      // the phantom fixture at recipe ih.ibsi_fbn
#include "test_2d_intensity_histogram_regression.h"  // ih_get
#include "test_main_nyxus.h"                         // agrees_gt
#include "test_ref_vals.h"                           // ref_vals_map

// Provenance: IBSI (Zwanenburg et al. 2020, arXiv:1612.07003) §3.4 digital-phantom
// intensity-histogram consensus values, quoted to 3 significant figures. Discretisation config:
// FBN (fixed bin number) with GREYDEPTH=6, IBSI mode on. Index base: Nyxus reports IDX features in
// the 1-based grey-level convention, matching IBSI directly (no offset). That these are the
// published values and not Nyxus output relabelled is checked in
// tests/vetting/audit/intensity_histogram_2d_ibsi_vetting_report.md.
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

// The fixture lives in test_2d_intensity_histogram_common.h, so the IBSI consensus, the MIRP
// goldens and the analytic closed forms are all read off the same computation.
static void run_intensity_histogram_ibsi_fixture(std::vector<std::vector<double>>& fvals, int nbins) {
    calc_2d_intensity_histogram_phantom(fvals, nbins);
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
    // ROBUST_MEAN_IDX has no IBSI feature; it is covered in test_2d_intensity_histogram_analytic.h.

    // ---- the four _VAL features an IBSI-vetted _IDX can anchor ----
    // Only these four: the three deviation measures are pure-scale statistics, where the centre
    // map's offset cancels in the difference and only binWidth survives, and CoV rebuilds the ratio
    // from VARIANCE_IDX rather than from COEFFICIENT_OF_VARIATION_IDX. The rest of the _VAL family
    // is not an image of its _IDX partner at all - five conventions coexist, set out in
    // tests/vetting/audit/intensity_histogram_2d_analytic_vetting_report.md.
    double b = ih_get(fv, F::IH_BIN_SIZE);                 // binWidth
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

