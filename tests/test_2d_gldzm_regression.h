#pragma once

#include "test_2d_gldzm_common.h"   // gtest, <string>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// Pinned Nyxus output on the four IBSI phantom slices in IBSI mode, averaged -- the same fixture and
// recipe the oracle files use. SPEC 2 regression tier: a drift guard, no oracle claim.
//
// These two are here because no reference in the tree covers them. GLDZM_GLM (grey level mean) and
// GLDZM_ZDM (zone distance mean) are not IBSI GLDZM features, so the published consensus has no
// entry for them, and mirp exposes no column for either. The rest of the family is vetted against
// mirp in test_2d_gldzm_mirp.h and against IBSI in test_2d_gldzm_ibsi.h.
static const ref_vals_map<double> gldzm_2d_regression_ref_vals
{
    {"GLDZM_GLM", 3.5261904761904761},
    {"GLDZM_ZDM", 1.0714285714285714}
};

/// @brief Pins one GLDZM feature against its recorded value on the IBSI phantom
void assert_gldzm_feature_regression (const Feature2D& feature_, const std::string& feature_name)
{
    // agrees_gt's default frac_tolerance is rel=1e-3, which is the drift band these guards want:
    // the quantities are means over four slices and nothing but a real change should move them
    assert_gldzm_feature_against_golden_values (feature_, feature_name, gldzm_2d_regression_ref_vals,
                                                "regression ", 1000.);
}

void test_2d_gldzm_glm_regression()
{
    assert_gldzm_feature_regression (Nyxus::Feature2D::GLDZM_GLM, "GLDZM_GLM");
}

void test_2d_gldzm_zdm_regression()
{
    assert_gldzm_feature_regression (Nyxus::Feature2D::GLDZM_ZDM, "GLDZM_ZDM");
}
