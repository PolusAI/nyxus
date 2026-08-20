#pragma once

#include "test_2d_glszm_common.h"   // gtest, <string>, <vector>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// mirp goldens for the 2D GLSZM family (SPEC 6.4 provenance).
// tool=mirp 2.6.0; env=nyxus_mirp (conda); recipe=glszm.ibsi_phantom_2d, i.e. the four IBSI
// digital-phantom slices from test_data.h featurised one at a time, with by_slice=True and
// base_discretisation_method="none" (the phantom is already discrete 1..6), which is what Nyxus
// computes in IBSI mode; generator=tests/vetting/oracles/gen_glszm_mirp.py.
//
// Fifteen of the sixteen features are bit-identical to mirp (2.0e-16 worst case), so they assert at
// the SPEC 7 "exact" tier. test_2d_glszm_ibsi.h pins the same quantities as published IBSI consensus
// values, quoted to three significant figures, and therefore asserts at rel=1e-2: IBSI fixes the
// definition, mirp fixes the digits.
//
// GLSZM_ZE is the exception and carries its own band, see glszm_2d_mirp_ze_frac_tolerance below.
//
// TWO TABLES, because a mean cannot vet the four values behind it: errors in two slices that cancel
// leave the average unmoved, and a discrepancy confined to one slice reaches it quartered.
// glszm_2d_mirp_slice_ref_vals holds mirp's value for each phantom slice on its own,
// glszm_2d_mirp_ref_vals the four-slice mean, and every assertion below checks both. Planting a
// cancelling pair of per-slice errors fails the per-slice assertion and passes the mean one.
static const ref_vals_map<double> glszm_2d_mirp_ref_vals
{
    {"GLSZM_SAE", 0.36330794123204835},
    {"GLSZM_LAE", 43.86666666666667},
    {"GLSZM_LGLZE", 0.3711970899470899},
    {"GLSZM_HGLZE", 16.44047619047619},
    {"GLSZM_SALGLE", 0.025854788674729148},
    {"GLSZM_SAHGLE", 10.277990480914587},
    {"GLSZM_LALGLE", 40.398082010582016},
    {"GLSZM_LAHGLE", 112.52142857142857},
    {"GLSZM_GLN", 1.4142857142857144},
    {"GLSZM_GLNN", 0.3229931972789115},
    {"GLSZM_SZN", 1.4857142857142858},
    {"GLSZM_SZNN", 0.3331972789115646},
    {"GLSZM_ZP", 0.24038957688338494},
    {"GLSZM_GLV", 3.9694784580498865},
    {"GLSZM_ZV", 20.99705215419501},
    {"GLSZM_ZE", 1.9319448617396766}
};

// mirp's value for each phantom slice on its own, keyed <feature>_z<slice>, from the same run and
// at the same config as the means above.
static const ref_vals_map<double> glszm_2d_mirp_slice_ref_vals
{
    {"GLSZM_SAE_z1", 0.10555555555555556}, {"GLSZM_SAE_z2", 0.5111607142857143},
    {"GLSZM_SAE_z3", 0.4183673469387755}, {"GLSZM_SAE_z4", 0.41814814814814816},

    {"GLSZM_LAE_z1", 18.8}, {"GLSZM_LAE_z2", 13.0},
    {"GLSZM_LAE_z3", 67.0}, {"GLSZM_LAE_z4", 76.66666666666667},

    {"GLSZM_LGLZE_z1", 0.4305555555555555}, {"GLSZM_LGLZE_z2", 0.3273809523809524},
    {"GLSZM_LGLZE_z3", 0.3634259259259259}, {"GLSZM_LGLZE_z4", 0.3634259259259259},

    {"GLSZM_HGLZE_z1", 14.0}, {"GLSZM_HGLZE_z2", 16.428571428571427},
    {"GLSZM_HGLZE_z3", 17.666666666666668}, {"GLSZM_HGLZE_z4", 17.666666666666668},

    {"GLSZM_SALGLE_z1", 0.03186728395061729}, {"GLSZM_SALGLE_z2", 0.0394345238095238},
    {"GLSZM_SALGLE_z3", 0.016168272864701436}, {"GLSZM_SALGLE_z4", 0.015949074074074074},

    {"GLSZM_SAHGLE_z1", 1.7166666666666668}, {"GLSZM_SAHGLE_z2", 12.725446428571429},
    {"GLSZM_SAHGLE_z3", 13.335034013605442}, {"GLSZM_SAHGLE_z4", 13.334814814814814},

    {"GLSZM_LALGLE_z1", 9.55}, {"GLSZM_LALGLE_z2", 11.523809523809524},
    {"GLSZM_LALGLE_z3", 65.42592592592592}, {"GLSZM_LALGLE_z4", 75.0925925925926},

    {"GLSZM_LAHGLE_z1", 201.8}, {"GLSZM_LAHGLE_z2", 41.285714285714285},
    {"GLSZM_LAHGLE_z3", 98.66666666666667}, {"GLSZM_LAHGLE_z4", 108.33333333333333},

    {"GLSZM_GLN_z1", 1.8}, {"GLSZM_GLN_z2", 1.8571428571428572},
    {"GLSZM_GLN_z3", 1.0}, {"GLSZM_GLN_z4", 1.0},

    {"GLSZM_GLNN_z1", 0.36}, {"GLSZM_GLNN_z2", 0.2653061224489796},
    {"GLSZM_GLNN_z3", 0.3333333333333333}, {"GLSZM_GLNN_z4", 0.3333333333333333},

    {"GLSZM_SZN_z1", 1.8}, {"GLSZM_SZN_z2", 2.142857142857143},
    {"GLSZM_SZN_z3", 1.0}, {"GLSZM_SZN_z4", 1.0},

    {"GLSZM_SZNN_z1", 0.36}, {"GLSZM_SZNN_z2", 0.30612244897959184},
    {"GLSZM_SZNN_z3", 0.3333333333333333}, {"GLSZM_SZNN_z4", 0.3333333333333333},

    {"GLSZM_ZP_z1", 0.25}, {"GLSZM_ZP_z2", 0.3684210526315789},
    {"GLSZM_ZP_z3", 0.17647058823529413}, {"GLSZM_ZP_z4", 0.16666666666666666},

    {"GLSZM_GLV_z1", 3.7600000000000002}, {"GLSZM_GLV_z2", 3.673469387755102},
    {"GLSZM_GLV_z3", 4.222222222222222}, {"GLSZM_GLV_z4", 4.222222222222222},

    {"GLSZM_ZV_z1", 2.8}, {"GLSZM_ZV_z2", 5.63265306122449},
    {"GLSZM_ZV_z3", 34.888888888888886}, {"GLSZM_ZV_z4", 40.666666666666664},

    {"GLSZM_ZE_z1", 2.321928094887362}, {"GLSZM_ZE_z2", 2.2359263506290326},
    {"GLSZM_ZE_z3", 1.5849625007211563}, {"GLSZM_ZE_z4", 1.5849625007211563}
};

// rel=1e-9: agrees_gt divides the golden by this, so a larger argument is a tighter band
static const double glszm_2d_mirp_frac_tolerance = 1e9;

// rel=4e-3 for GLSZM_ZE alone. Nyxus computes zone entropy through Nyxus::fast_log10, a
// float-precision quadratic approximation of the logarithm, where mirp uses a double log2; that
// costs 2.5e-3 per slice and 2.0e-3 on the mean against this run, while the family's other fifteen
// features are bit-identical to mirp. The band is a statement about the approximation and nothing
// else, and it is the only reason this feature is not at the exact tier. Widening it to cover
// anything further, or applying it to another feature, would be a test bug.
static const double glszm_2d_mirp_ze_frac_tolerance = 250.;

static double glszm_2d_mirp_tolerance_for (const std::string& feature_name)
{
    return feature_name == "GLSZM_ZE" ? glszm_2d_mirp_ze_frac_tolerance : glszm_2d_mirp_frac_tolerance;
}

static void assert_glszm_feature_mirp (const Feature2D& feature_, const std::string& feature_name)
{
    const Fsettings s = make_glszm2d_settings (true, 128);
    const double frac_tolerance = glszm_2d_mirp_tolerance_for (feature_name);

    // per slice first: this is what a mean cannot check
    const std::vector<double> per_slice = glszm_2d_phantom_slice_values (feature_, s);
    ASSERT_EQ (per_slice.size(), 4u) << feature_name;

    for (size_t z = 0; z < per_slice.size(); z++)
    {
        const std::string key = feature_name + "_z" + std::to_string (z + 1);
        SCOPED_TRACE ("mirp " + key);
        ASSERT_TRUE (glszm_2d_mirp_slice_ref_vals.count(key) > 0) << key;
        ASSERT_TRUE (agrees_gt (per_slice[z], glszm_2d_mirp_slice_ref_vals.at(key), frac_tolerance))
            << key;
    }

    // then the four-slice mean, the quantity IBSI publishes
    assert_glszm_feature_against_golden_values (feature_, feature_name, glszm_2d_mirp_ref_vals,
                                                "mirp ", frac_tolerance, s);
}

void test_2d_glszm_sae_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_SAE, "GLSZM_SAE");
}

void test_2d_glszm_lae_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_LAE, "GLSZM_LAE");
}

void test_2d_glszm_lglze_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_LGLZE, "GLSZM_LGLZE");
}

void test_2d_glszm_hglze_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_HGLZE, "GLSZM_HGLZE");
}

void test_2d_glszm_salgle_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_SALGLE, "GLSZM_SALGLE");
}

void test_2d_glszm_sahgle_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_SAHGLE, "GLSZM_SAHGLE");
}

void test_2d_glszm_lalgle_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_LALGLE, "GLSZM_LALGLE");
}

void test_2d_glszm_lahgle_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_LAHGLE, "GLSZM_LAHGLE");
}

void test_2d_glszm_gln_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_GLN, "GLSZM_GLN");
}

void test_2d_glszm_glnn_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_GLNN, "GLSZM_GLNN");
}

void test_2d_glszm_szn_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_SZN, "GLSZM_SZN");
}

void test_2d_glszm_sznn_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_SZNN, "GLSZM_SZNN");
}

void test_2d_glszm_zp_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_ZP, "GLSZM_ZP");
}

void test_2d_glszm_glv_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_GLV, "GLSZM_GLV");
}

void test_2d_glszm_zv_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_ZV, "GLSZM_ZV");
}

void test_2d_glszm_ze_mirp()
{
    assert_glszm_feature_mirp (Nyxus::Feature2D::GLSZM_ZE, "GLSZM_ZE");
}
