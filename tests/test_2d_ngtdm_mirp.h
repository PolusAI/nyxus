#pragma once

#include "test_2d_ngtdm_common.h"   // gtest, <string>, <vector>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// mirp goldens for the 2D NGTDM family (SPEC 6.4 provenance).
// tool=mirp 2.6.0; env=nyxus_mirp (conda); recipe=ngtdm.ibsi_phantom_2d, i.e. the four IBSI
// digital-phantom slices from test_data.h featurised one at a time, with by_slice=True and
// base_discretisation_method="none" (the phantom is already discrete 1..6), which is what Nyxus
// computes in IBSI mode; generator=tests/vetting/oracles/gen_ngtdm_mirp.py.
//
// All five features are bit-identical to mirp on every slice (3.2e-16 worst case, floating-point
// summation order), so they assert at the SPEC 7 exact tier. PyRadiomics 3.0.1 was run on the same
// fixture and recipe and agrees with these goldens to 1.6e-16; it is not pinned separately because a
// second table identical to this one is redundancy rather than coverage. See
// tests/vetting/audit/ngtdm_2d_mirp_vetting_report.md.
//
// test_2d_ngtdm_ibsi.h pins the same quantities as published IBSI consensus values, quoted to three
// significant figures, and therefore asserts at rel=1e-2: IBSI fixes the definition, mirp fixes the
// digits.
//
// TWO TABLES, because a mean cannot vet the four values behind it: errors in two slices that cancel
// leave the average unmoved, and a discrepancy confined to one slice reaches it quartered.
// ngtdm_2d_mirp_slice_ref_vals holds mirp's value for each phantom slice on its own,
// ngtdm_2d_mirp_ref_vals the four-slice mean, and every assertion below checks both.
static const ref_vals_map<double> ngtdm_2d_mirp_ref_vals
{
    {"NGTDM_COARSENESS", 0.12051055470374192},
    {"NGTDM_CONTRAST", 0.9252630132885581},
    {"NGTDM_BUSYNESS", 2.9887939543849873},
    {"NGTDM_COMPLEXITY", 10.400131856837582},
    {"NGTDM_STRENGTH", 2.8763659173789415}
};

// mirp's value for each phantom slice on its own, keyed <feature>_z<slice>, from the same run and
// at the same config as the means above.
static const ref_vals_map<double> ngtdm_2d_mirp_slice_ref_vals
{
    {"NGTDM_COARSENESS_z1", 0.1003260596940055}, {"NGTDM_COARSENESS_z2", 0.1063128234847425},
    {"NGTDM_COARSENESS_z3", 0.13481363996827916}, {"NGTDM_COARSENESS_z4", 0.14058969566794052},

    {"NGTDM_CONTRAST_z1", 1.6362843750000002}, {"NGTDM_CONTRAST_z2", 0.691729844000583},
    {"NGTDM_CONTRAST_z3", 0.7298857259166641}, {"NGTDM_CONTRAST_z4", 0.6431521082369848},

    {"NGTDM_BUSYNESS_z1", 2.1668478260869564}, {"NGTDM_BUSYNESS_z2", 2.2912545787545793},
    {"NGTDM_BUSYNESS_z3", 3.940625}, {"NGTDM_BUSYNESS_z4", 3.556448412698413},

    {"NGTDM_COMPLEXITY_z1", 10.750942513368985}, {"NGTDM_COMPLEXITY_z2", 14.882002533807048},
    {"NGTDM_COMPLEXITY_z3", 8.37328431372549}, {"NGTDM_COMPLEXITY_z4", 7.594298066448804},

    {"NGTDM_STRENGTH_z1", 1.7958446251129176}, {"NGTDM_STRENGTH_z2", 2.5242791143458305},
    {"NGTDM_STRENGTH_z3", 3.5422771781859765}, {"NGTDM_STRENGTH_z4", 3.6430627518710423}
};

// rel=1e-9: agrees_gt divides the golden by this, so a larger argument is a tighter band
static const double ngtdm_2d_mirp_frac_tolerance = 1e9;

static void assert_ngtdm_feature_mirp (const Feature2D& feature_, const std::string& feature_name)
{
    const Fsettings s = make_ngtdm2d_settings (true);

    // per slice first: this is what a mean cannot check. n_levels is irrelevant in IBSI mode and is
    // passed as the default 0 rather than the regression file's 100, which the assertions confirm.
    const std::vector<double> per_slice = ngtdm_2d_phantom_slice_values (feature_, s, 0);
    ASSERT_EQ (per_slice.size(), 4u) << feature_name;

    for (size_t z = 0; z < per_slice.size(); z++)
    {
        const std::string key = feature_name + "_z" + std::to_string (z + 1);
        SCOPED_TRACE ("mirp " + key);
        ASSERT_TRUE (ngtdm_2d_mirp_slice_ref_vals.count(key) > 0) << key;
        ASSERT_TRUE (agrees_gt (per_slice[z], ngtdm_2d_mirp_slice_ref_vals.at(key),
                                ngtdm_2d_mirp_frac_tolerance)) << key;
    }

    // then the four-slice mean, the quantity IBSI publishes
    assert_ngtdm_feature_against_golden_values (feature_, feature_name, ngtdm_2d_mirp_ref_vals,
                                                "mirp ", ngtdm_2d_mirp_frac_tolerance, s, 0);
}

void test_2d_ngtdm_coarseness_mirp()
{
    assert_ngtdm_feature_mirp (Nyxus::Feature2D::NGTDM_COARSENESS, "NGTDM_COARSENESS");
}

void test_2d_ngtdm_contrast_mirp()
{
    assert_ngtdm_feature_mirp (Nyxus::Feature2D::NGTDM_CONTRAST, "NGTDM_CONTRAST");
}

void test_2d_ngtdm_busyness_mirp()
{
    assert_ngtdm_feature_mirp (Nyxus::Feature2D::NGTDM_BUSYNESS, "NGTDM_BUSYNESS");
}

void test_2d_ngtdm_complexity_mirp()
{
    assert_ngtdm_feature_mirp (Nyxus::Feature2D::NGTDM_COMPLEXITY, "NGTDM_COMPLEXITY");
}

void test_2d_ngtdm_strength_mirp()
{
    assert_ngtdm_feature_mirp (Nyxus::Feature2D::NGTDM_STRENGTH, "NGTDM_STRENGTH");
}
