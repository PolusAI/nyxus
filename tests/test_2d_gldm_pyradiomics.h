#pragma once

#include "test_2d_gldm_common.h"   // gtest, <string>, <vector>, the slice featuriser and the mean assertion
#include "test_ref_vals.h"         // ref_vals_map

// PyRadiomics goldens for the 2D GLDM family (SPEC 6.4 provenance).
// tool=pyradiomics v3.0.1; env=nyxus_oracle (conda, Python 3.9); recipe=gldm.ibsi_phantom_2d, i.e.
// the four IBSI digital-phantom slices from test_data.h featurised one at a time and averaged, with
// binWidth=1 (identity binning on this integer phantom, so neither tool discretises), gldm_a=0,
// distances=[1] and force2D -- the alpha=0, d=1 coarseness Nyxus computes in IBSI mode;
// generator=tests/vetting/oracles/gen_gldm_pyradiomics.py.
//
// PyRadiomics is the reference that defines GLDM and names all 14 of the family's features, so one
// run covers the family. Worst residual over 13 of the 14 features x 4 slices is 2.2e-16, so those
// assert at the SPEC 7 "exact" tier (rel=1e-9). GLDM_DE is the exception and carries its own band --
// see gldm_2d_pyradiomics_de_frac_tolerance below.
//
// TWO TABLES, because a mean cannot vet the four values behind it: errors in two slices that cancel
// leave the average unmoved. gldm_2d_pyradiomics_slice_ref_vals holds PyRadiomics' value for each
// phantom slice on its own, and gldm_2d_pyradiomics_ref_vals holds the four-slice mean that IBSI's
// "2D, averaged" aggregation publishes. Every assertion below checks both.
//
// test_2d_gldm_ibsi.h pins the same quantities as published IBSI consensus values, quoted to three
// significant figures -- that file therefore asserts at rel=1e-2 and this one at rel=1e-9, or at
// rel=2.5e-3 for GLDM_DE. The two are complementary: IBSI fixes the definition, PyRadiomics fixes
// the digits. The IBSI values are published under the NGLDM name because a GLDM dependence count is
// IBSI's j = k + 1 at alpha=0, d=1; tests/vetting/audit/gldm_2d_pyradiomics_vetting_report.md
// carries the mapping and the measurement that establishes it.
static const ref_vals_map<double> gldm_2d_pyradiomics_ref_vals
{
    {"GLDM_SDE", 0.15807024738501638},
    {"GLDM_LDE", 19.173821809425526},
    {"GLDM_GLN", 10.24637942896457},
    {"GLDM_DN", 3.9646456828345373},
    {"GLDM_DNN", 0.21177218060411693},
    {"GLDM_GLV", 2.7037332451477982},
    {"GLDM_DV", 2.729504577399913},
    {"GLDM_DE", 2.714292423281547},
    {"GLDM_LGLE", 0.7017531915300232},
    {"GLDM_HGLE", 7.486949604403165},
    {"GLDM_SDLGLE", 0.047290498640367454},
    {"GLDM_SDHGLE", 3.064914180133554},
    {"GLDM_LDLGLE", 17.59968920804189},
    {"GLDM_LDHGLE", 49.477721878224976}
};

// PyRadiomics' value for each phantom slice on its own, keyed <feature>_z<slice>, in the same run
// and at the same config as the means above.
static const ref_vals_map<double> gldm_2d_pyradiomics_slice_ref_vals
{
    {"GLDM_SDE_z1", 0.14045833333333332}, {"GLDM_SDE_z2", 0.24780701754385964},
    {"GLDM_SDE_z3", 0.127515339469121}, {"GLDM_SDE_z4", 0.11650029919375157},

    {"GLDM_LDE_z1", 10.6}, {"GLDM_LDE_z2", 14.578947368421053},
    {"GLDM_LDE_z3", 22.294117647058822}, {"GLDM_LDE_z4", 29.22222222222222},

    {"GLDM_GLN_z1", 7.7}, {"GLDM_GLN_z2", 8.68421052631579},
    {"GLDM_GLN_z3", 11.823529411764707}, {"GLDM_GLN_z4", 12.777777777777779},

    {"GLDM_DN_z1", 5.7}, {"GLDM_DN_z2", 4.2631578947368425},
    {"GLDM_DN_z3", 3.1176470588235294}, {"GLDM_DN_z4", 2.7777777777777777},

    {"GLDM_DNN_z1", 0.285}, {"GLDM_DNN_z2", 0.22437673130193905},
    {"GLDM_DNN_z3", 0.18339100346020762}, {"GLDM_DNN_z4", 0.15432098765432098},

    {"GLDM_GLV_z1", 3.5474999999999994}, {"GLDM_GLV_z2", 3.1412742382271466},
    {"GLDM_GLV_z3", 2.110726643598616}, {"GLDM_GLV_z4", 2.0154320987654324},

    {"GLDM_DV_z1", 0.99}, {"GLDM_DV_z2", 2.8753462603878113},
    {"GLDM_DV_z3", 2.8304498269896197}, {"GLDM_DV_z4", 4.222222222222222},

    {"GLDM_DE_z1", 3.0464393446710125}, {"GLDM_DE_z2", 2.3789919869000604},
    {"GLDM_DE_z3", 2.60985016602894}, {"GLDM_DE_z4", 2.821888195526176},

    {"GLDM_LGLE_z1", 0.4791666666666667}, {"GLDM_LGLE_z2", 0.6535087719298245},
    {"GLDM_LGLE_z3", 0.8325163398692811}, {"GLDM_LGLE_z4", 0.841820987654321},

    {"GLDM_HGLE_z1", 12.25}, {"GLDM_HGLE_z2", 8.263157894736842},
    {"GLDM_HGLE_z3", 4.823529411764706}, {"GLDM_HGLE_z4", 4.611111111111111},

    {"GLDM_SDLGLE_z1", 0.06062133487654321}, {"GLDM_SDLGLE_z2", 0.049342105263157895},
    {"GLDM_SDLGLE_z3", 0.04275226757369615}, {"GLDM_SDLGLE_z4", 0.03644628684807256},

    {"GLDM_SDHGLE_z1", 2.011986111111111}, {"GLDM_SDHGLE_z2", 5.142543859649123},
    {"GLDM_SDHGLE_z3", 2.627515339469121}, {"GLDM_SDHGLE_z4", 2.4776114103048625},

    {"GLDM_LDLGLE_z1", 6.211111111111111}, {"GLDM_LDLGLE_z2", 13.640350877192983},
    {"GLDM_LDLGLE_z3", 21.795751633986928}, {"GLDM_LDLGLE_z4", 28.751543209876544},

    {"GLDM_LDHGLE_z1", 97.35}, {"GLDM_LDHGLE_z2", 31.31578947368421},
    {"GLDM_LDHGLE_z3", 31.41176470588235}, {"GLDM_LDHGLE_z4", 37.833333333333336}
};

// rel=1e-9: agrees_gt divides the golden by this, so a larger argument is a tighter band
static const double gldm_2d_pyradiomics_frac_tolerance = 1e9;

// rel=2.5e-3, the one feature that cannot hold the exact tier. GLDMFeature::calc_DE() reads its
// logarithm through Nyxus::fast_log10 (src/nyx/helpers/helpers.h), the shared float polynomial every
// entropy in the texture set uses; its worst error against log2 is 8.9e-3 over the [0.75, 1.5)
// reduction range, and on this phantom that lands in DE as at most 1.3e-3 relative (worst slice z1:
// Nyxus 3.0425443887710575 vs PyRadiomics 3.0464393446710125; the four-slice mean is off by 7.9e-4).
// The band is twice that measured residual, per SPEC 7's "documented residual" tier. It is not a
// free pass: every other GLDM feature holds rel=1e-9 on the same dependence matrix, so a real error
// in the matrix still fails somewhere. This is the same accommodation 2D GLCM already makes for the
// same helper -- its log-based features land 1e-3..3e-3 off both tools and assert at rel=5e-3 --
// so closing the band is the whole texture set's business, not this family's.
// tests/vetting/audit/gldm_2d_pyradiomics_vetting_report.md carries the measurement.
static const double gldm_2d_pyradiomics_de_frac_tolerance = 400.0;

static double gldm_2d_pyradiomics_tolerance_for (const std::string& feature_name)
{
    return feature_name == "GLDM_DE" ? gldm_2d_pyradiomics_de_frac_tolerance
                                     : gldm_2d_pyradiomics_frac_tolerance;
}

static void assert_gldm_feature_pyradiomics (const Feature2D& feature_, const std::string& feature_name)
{
    const double frac_tolerance = gldm_2d_pyradiomics_tolerance_for (feature_name);

    // per slice first: this is what a mean cannot check
    const std::vector<double> per_slice = gldm_2d_phantom_slice_values (feature_);
    ASSERT_EQ (per_slice.size(), 4u) << feature_name;

    for (size_t z = 0; z < per_slice.size(); z++)
    {
        const std::string key = feature_name + "_z" + std::to_string (z + 1);
        SCOPED_TRACE ("pyradiomics " + key);
        ASSERT_TRUE (gldm_2d_pyradiomics_slice_ref_vals.count(key) > 0) << key;
        ASSERT_TRUE (agrees_gt (per_slice[z], gldm_2d_pyradiomics_slice_ref_vals.at(key),
                                frac_tolerance)) << key;
    }

    // then the four-slice mean, the quantity IBSI publishes -- averaged from the same per_slice the
    // loop above just checked, so the two assertions cannot be reading different featurisations
    assert_gldm_mean_against_golden_values (per_slice, feature_name, gldm_2d_pyradiomics_ref_vals,
                                            "pyradiomics ", frac_tolerance);
}

void test_2d_gldm_sde_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_SDE, "GLDM_SDE");
}

void test_2d_gldm_lde_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_LDE, "GLDM_LDE");
}

void test_2d_gldm_gln_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_GLN, "GLDM_GLN");
}

void test_2d_gldm_dn_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_DN, "GLDM_DN");
}

void test_2d_gldm_dnn_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_DNN, "GLDM_DNN");
}

void test_2d_gldm_glv_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_GLV, "GLDM_GLV");
}

void test_2d_gldm_dv_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_DV, "GLDM_DV");
}

void test_2d_gldm_de_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_DE, "GLDM_DE");
}

void test_2d_gldm_lgle_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_LGLE, "GLDM_LGLE");
}

void test_2d_gldm_hgle_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_HGLE, "GLDM_HGLE");
}

void test_2d_gldm_sdlgle_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_SDLGLE, "GLDM_SDLGLE");
}

void test_2d_gldm_sdhgle_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_SDHGLE, "GLDM_SDHGLE");
}

void test_2d_gldm_ldlgle_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_LDLGLE, "GLDM_LDLGLE");
}

void test_2d_gldm_ldhgle_pyradiomics()
{
    assert_gldm_feature_pyradiomics (Nyxus::Feature2D::GLDM_LDHGLE, "GLDM_LDHGLE");
}
