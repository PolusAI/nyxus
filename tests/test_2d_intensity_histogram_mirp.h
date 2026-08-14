#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../src/nyx/featureset.h"                // UserFacingFeatureNames
#include "test_2d_intensity_histogram_common.h"   // the phantom fixture at recipe ih.ibsi_fbn
#include "test_main_nyxus.h"                      // agrees_gt
#include "test_ref_vals.h"                        // ref_vals_map

// mirp 2.6.0, recipe ih.ibsi_fbn (fixed bin number, 6 bins), IBSI digital phantom
// Generated offline by tests/vetting/oracles/gen_intensity_histogram_mirp.py (SPEC 6.4) on the
// IBSI digital phantom as tests/test_data.h stores it, at recipe ih.ibsi_fbn: fixed bin number,
// 6 bins, the whole 4-slice masked voxel set as one ROI.
//
// MIRP works entirely in the discretised domain, so these goldens vet the Nyxus _IDX family - the
// statistics over bin indices - plus the two gradient magnitudes, which carry no domain. The _VAL
// family has no MIRP counterpart and is vetted analytically in
// test_2d_intensity_histogram_analytic.h; the conventions are set out in
// tests/vetting/audit/intensity_histogram_2d_analytic_vetting_report.md.
static ref_vals_map<double> intensity_histogram_2d_mirp_ref_vals
{
    {"IH_COEFFICIENT_OF_VARIATION_IDX", 0.81219785849173143},   // ih_cov
    {"IH_ENTROPY_IDX", 1.2656115555865246},   // ih_entropy
    {"IH_INTERQUANTILE_RANGE_IDX", 3},   // ih_iqr
    {"IH_EXCESS_KURTOSIS_IDX", -0.35462048068783414},   // ih_kurt
    {"IH_MEAN_ABSOLUTE_DEVIATION_IDX", 1.552227903579255},   // ih_mad
    {"IH_MAXIMUM_IDX", 6},   // ih_max
    {"IH_MAX_GRADIENT", 8},   // ih_max_grad
    {"IH_MAX_GRADIENT_IDX", 3},   // ih_max_grad_g
    {"IH_MEAN_IDX", 2.1486486486486487},   // ih_mean
    {"IH_MEDIAN_ABSOLUTE_DEVIATION_IDX", 1.1486486486486487},   // ih_medad
    {"IH_MEDIAN_IDX", 1},   // ih_median
    {"IH_MINIMUM_IDX", 1},   // ih_min
    {"IH_MIN_GRADIENT", -50},   // ih_min_grad
    {"IH_MIN_GRADIENT_IDX", 1},   // ih_min_grad_g
    {"IH_MODE_IDX", 1},   // ih_mode
    {"IH_P10_IDX", 1},   // ih_p10
    {"IH_P90_IDX", 4},   // ih_p90
    {"IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX", 0.59999999999999998},   // ih_qcod
    {"IH_RANGE_IDX", 5},   // ih_range
    {"IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX", 1.1138338159946539},   // ih_rmad
    {"IH_SKEWNESS_IDX", 1.0838207225574563},   // ih_skew
    {"IH_UNIFORMITY_IDX", 0.51241782322863394},   // ih_uniformity
    {"IH_VARIANCE_IDX", 3.0454711468224982},   // ih_var
};

void assert_2d_intensity_histogram_feature_mirp (const std::vector<std::vector<double>>& fvals,
                                                 const std::string& feature_name, double golden)
{
    double value = 0;
    ASSERT_TRUE (read_2d_intensity_histogram_feature (fvals, feature_name, value)) << feature_name;
    // Nyxus reproduces MIRP to double precision on every one of these; 1e-9 is ~1e6 tighter than
    // the worst measured deviation and still leaves room for a different libm
    ASSERT_TRUE (Nyxus::agrees_gt (value, golden, 1.e9)) << feature_name;
}

void test_2d_intensity_histogram_family_mirp()
{
    // Every _IDX feature the build exposes has to be pinned here. Without this a feature added to
    // the family later is vetted by nothing while this test still passes over the table it has.
    // IH_ROBUST_MEAN_IDX is the one exemption: the mean over the [P10,P90]-trimmed histogram is a
    // Nyxus feature with no counterpart in MIRP or in the IBSI reference set, and it is vetted
    // analytically instead - on this same phantom in test_2d_intensity_histogram_phantom_analytic,
    // and on the tail-trimming fixture in test_2d_intensity_histogram_dispersion_robust_analytic.
    for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
        if (name.rfind ("IH_", 0) == 0 && name.size() > 4 &&
            name.compare (name.size() - 4, 4, "_IDX") == 0 && name != "IH_ROBUST_MEAN_IDX")
            ASSERT_TRUE (intensity_histogram_2d_mirp_ref_vals.count (name))
                << name << " has no mirp golden";

    std::vector<std::vector<double>> fvals;
    calc_2d_intensity_histogram_phantom (fvals);
    for (const auto& [feature_name, golden] : intensity_histogram_2d_mirp_ref_vals)
        assert_2d_intensity_histogram_feature_mirp (fvals, feature_name, golden);
}
