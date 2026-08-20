#pragma once

#include "test_2d_gldzm_common.h"   // gtest, <string>, <vector>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// mirp goldens for the 2D GLDZM family (SPEC 6.4 provenance).
// tool=mirp 2.6.0; env=nyxus_mirp (conda); recipe=gldzm.ibsi_phantom_2d, i.e. the four IBSI
// digital-phantom slices from test_data.h featurised one at a time, with by_slice=True and
// base_discretisation_method="none" (the phantom is already discrete 1..6), which is what Nyxus
// computes in IBSI mode; generator=tests/vetting/oracles/gen_gldzm_mirp.py.
//
// mirp and Nyxus build the same 8-connected zones over the same distance-to-border metric and agree
// to 8.9e-16 worst case, so these assert at the SPEC 7 "exact" tier (rel=1e-9). test_2d_gldzm_ibsi.h
// pins the same quantities as published IBSI consensus values, quoted to three significant figures,
// and therefore asserts at rel=1e-2: IBSI fixes the definition, mirp fixes the digits.
//
// TWO TABLES, because a mean cannot vet the four values behind it: errors in two slices that cancel
// leave the average unmoved, and a discrepancy confined to one slice reaches it quartered.
// gldzm_2d_mirp_slice_ref_vals holds mirp's value for each phantom slice on its own,
// gldzm_2d_mirp_ref_vals the four-slice mean, and every assertion below checks both. Planting a
// cancelling pair of per-slice errors fails the per-slice assertion and passes the mean one.
//
// GLDZM_GLM and GLDZM_ZDM are absent by design: mirp exposes no grey-level-mean or
// zone-distance-mean column because neither is an IBSI GLDZM feature. They are drift-pinned in
// test_2d_gldzm_regression.h.
static const ref_vals_map<double> gldzm_2d_mirp_ref_vals
{
    {"GLDZM_SDE", 0.9464285714285714},
    {"GLDZM_LDE", 1.2142857142857144},
    {"GLDZM_LGLZE", 0.3711970899470899},
    {"GLDZM_HGLZE", 16.44047619047619},
    {"GLDZM_SDLGLE", 0.3674768518518518},
    {"GLDZM_SDHGLE", 15.235119047619047},
    {"GLDZM_LDLGLE", 0.38607804232804227},
    {"GLDZM_LDHGLE", 21.261904761904763},
    {"GLDZM_GLNU", 1.4142857142857144},
    {"GLDZM_GLNUN", 0.3229931972789115},
    {"GLDZM_ZDNU", 3.7857142857142856},
    {"GLDZM_ZDNUN", 0.8979591836734694},
    {"GLDZM_ZP", 0.24038957688338494},
    {"GLDZM_GLV", 3.9694784580498865},
    {"GLDZM_ZDV", 0.05102040816326531},
    {"GLDZM_ZDE", 1.7319448617396767}
};

// mirp's value for each phantom slice on its own, keyed <feature>_z<slice>, from the same run and
// at the same config as the means above.
static const ref_vals_map<double> gldzm_2d_mirp_slice_ref_vals
{
    {"GLDZM_SDE_z1", 1.0}, {"GLDZM_SDE_z2", 0.7857142857142857},
    {"GLDZM_SDE_z3", 1.0}, {"GLDZM_SDE_z4", 1.0},

    {"GLDZM_LDE_z1", 1.0}, {"GLDZM_LDE_z2", 1.8571428571428572},
    {"GLDZM_LDE_z3", 1.0}, {"GLDZM_LDE_z4", 1.0},

    {"GLDZM_LGLZE_z1", 0.4305555555555555}, {"GLDZM_LGLZE_z2", 0.3273809523809524},
    {"GLDZM_LGLZE_z3", 0.3634259259259259}, {"GLDZM_LGLZE_z4", 0.3634259259259259},

    {"GLDZM_HGLZE_z1", 14.0}, {"GLDZM_HGLZE_z2", 16.428571428571427},
    {"GLDZM_HGLZE_z3", 17.666666666666668}, {"GLDZM_HGLZE_z4", 17.666666666666668},

    {"GLDZM_SDLGLE_z1", 0.4305555555555555}, {"GLDZM_SDLGLE_z2", 0.3125},
    {"GLDZM_SDLGLE_z3", 0.3634259259259259}, {"GLDZM_SDLGLE_z4", 0.3634259259259259},

    {"GLDZM_SDHGLE_z1", 14.0}, {"GLDZM_SDHGLE_z2", 11.607142857142858},
    {"GLDZM_SDHGLE_z3", 17.666666666666668}, {"GLDZM_SDHGLE_z4", 17.666666666666668},

    {"GLDZM_LDLGLE_z1", 0.4305555555555555}, {"GLDZM_LDLGLE_z2", 0.3869047619047619},
    {"GLDZM_LDLGLE_z3", 0.3634259259259259}, {"GLDZM_LDLGLE_z4", 0.3634259259259259},

    {"GLDZM_LDHGLE_z1", 14.0}, {"GLDZM_LDHGLE_z2", 35.714285714285715},
    {"GLDZM_LDHGLE_z3", 17.666666666666668}, {"GLDZM_LDHGLE_z4", 17.666666666666668},

    {"GLDZM_GLNU_z1", 1.8}, {"GLDZM_GLNU_z2", 1.8571428571428572},
    {"GLDZM_GLNU_z3", 1.0}, {"GLDZM_GLNU_z4", 1.0},

    {"GLDZM_GLNUN_z1", 0.36}, {"GLDZM_GLNUN_z2", 0.2653061224489796},
    {"GLDZM_GLNUN_z3", 0.3333333333333333}, {"GLDZM_GLNUN_z4", 0.3333333333333333},

    {"GLDZM_ZDNU_z1", 5.0}, {"GLDZM_ZDNU_z2", 4.142857142857143},
    {"GLDZM_ZDNU_z3", 3.0}, {"GLDZM_ZDNU_z4", 3.0},

    {"GLDZM_ZDNUN_z1", 1.0}, {"GLDZM_ZDNUN_z2", 0.5918367346938775},
    {"GLDZM_ZDNUN_z3", 1.0}, {"GLDZM_ZDNUN_z4", 1.0},

    {"GLDZM_ZP_z1", 0.25}, {"GLDZM_ZP_z2", 0.3684210526315789},
    {"GLDZM_ZP_z3", 0.17647058823529413}, {"GLDZM_ZP_z4", 0.16666666666666666},

    {"GLDZM_GLV_z1", 3.7600000000000002}, {"GLDZM_GLV_z2", 3.673469387755102},
    {"GLDZM_GLV_z3", 4.222222222222222}, {"GLDZM_GLV_z4", 4.222222222222222},

    {"GLDZM_ZDV_z1", 0.0}, {"GLDZM_ZDV_z2", 0.20408163265306123},
    {"GLDZM_ZDV_z3", 0.0}, {"GLDZM_ZDV_z4", 0.0},

    {"GLDZM_ZDE_z1", 1.5219280948873621}, {"GLDZM_ZDE_z2", 2.2359263506290326},
    {"GLDZM_ZDE_z3", 1.5849625007211563}, {"GLDZM_ZDE_z4", 1.5849625007211563}
};

// rel=1e-9: agrees_gt divides the golden by this, so a larger argument is a tighter band
static const double gldzm_2d_mirp_frac_tolerance = 1e9;

static void assert_gldzm_feature_mirp (const Feature2D& feature_, const std::string& feature_name)
{
    // per slice first: this is what a mean cannot check
    const std::vector<double> per_slice = gldzm_2d_phantom_slice_values (feature_);
    ASSERT_EQ (per_slice.size(), 4u) << feature_name;

    for (size_t z = 0; z < per_slice.size(); z++)
    {
        const std::string key = feature_name + "_z" + std::to_string (z + 1);
        SCOPED_TRACE ("mirp " + key);
        ASSERT_TRUE (gldzm_2d_mirp_slice_ref_vals.count(key) > 0) << key;
        ASSERT_TRUE (agrees_gt (per_slice[z], gldzm_2d_mirp_slice_ref_vals.at(key),
                                gldzm_2d_mirp_frac_tolerance)) << key;
    }

    // then the four-slice mean, the quantity IBSI publishes
    assert_gldzm_feature_against_golden_values (feature_, feature_name, gldzm_2d_mirp_ref_vals,
                                                "mirp ", gldzm_2d_mirp_frac_tolerance);
}

void test_2d_gldzm_sde_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_SDE, "GLDZM_SDE");
}

void test_2d_gldzm_lde_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_LDE, "GLDZM_LDE");
}

void test_2d_gldzm_lglze_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_LGLZE, "GLDZM_LGLZE");
}

void test_2d_gldzm_hglze_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_HGLZE, "GLDZM_HGLZE");
}

void test_2d_gldzm_sdlgle_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_SDLGLE, "GLDZM_SDLGLE");
}

void test_2d_gldzm_sdhgle_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_SDHGLE, "GLDZM_SDHGLE");
}

void test_2d_gldzm_ldlgle_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_LDLGLE, "GLDZM_LDLGLE");
}

void test_2d_gldzm_ldhgle_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_LDHGLE, "GLDZM_LDHGLE");
}

void test_2d_gldzm_glnu_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_GLNU, "GLDZM_GLNU");
}

void test_2d_gldzm_glnun_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_GLNUN, "GLDZM_GLNUN");
}

void test_2d_gldzm_zdnu_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_ZDNU, "GLDZM_ZDNU");
}

void test_2d_gldzm_zdnun_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_ZDNUN, "GLDZM_ZDNUN");
}

void test_2d_gldzm_zp_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_ZP, "GLDZM_ZP");
}

void test_2d_gldzm_glv_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_GLV, "GLDZM_GLV");
}

void test_2d_gldzm_zdv_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_ZDV, "GLDZM_ZDV");
}

void test_2d_gldzm_zde_mirp()
{
    assert_gldzm_feature_mirp (Nyxus::Feature2D::GLDZM_ZDE, "GLDZM_ZDE");
}
