#pragma once
// ORACLE TEST — Nyxus 2D firstorder vs PyRadiomics.
//
// Provenance (goldens pinned below):
//   tool         = pyradiomics
//   version      = v3.0.1.post2+g9ccbec1
//   image_digest = radiomics/pyradiomics@sha256:eea20621c9e77afd049871e1a4e7308844a57d399343b087f6a4e86c3dab1923
//   recipe       = firstorder.pyradiomics_default (binCount=64, single 2D slice, spacing 1, label 1)
//   fixture      = pixelIntensityFeaturesTestData (tests/test_data.h)
//   generated    = 2026-07-14, offline, by a scratch generator (gen_firstorder_pyradiomics.py; not in-repo per design)
//
// Each golden is the PyRadiomics value; the test recomputes Nyxus live under the matching recipe and
// asserts agreement within the tier tolerance (agrees_gt frac_tolerance = 1/rel_bar):
//   exact       (rel <= 1e-6)  -> frac_tolerance 1e6
//   definitional(rel <= 1e-2)  -> frac_tolerance 100  [SPEC 7 cross-tool tier; the delta is derived]
//   approx      (rel <= 5e-2)  -> frac_tolerance 20   [convention deltas without a closed form]
// The regression file (test_2d_firstorder_regression.h) is untouched; it remains the exact change-detector.

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/nyx/dataset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_2d_firstorder_common.h"

using namespace Nyxus;

// Pinned PyRadiomics firstorder goldens (all 18 mapped features agreed; 0 flags).
static const std::unordered_map<std::string, double> firstorder_2d_pyradiomics_ref_vals = {
    {"MEAN", 32566.38961038961},
    {"MEDIAN", 29803.5},
    {"MIN", 11079.0},
    {"MAX", 64090.0},
    {"RANGE", 53011.0},
    // PyRadiomics Variance is the population variance (/N). Nyxus computes both denominators, so the
    // one PyRadiomics number is the reference for two features: VARIANCE_BIASED is var/N, the same
    // definition, and matches exactly; VARIANCE is the sample variance var/(N-1) and stands off by
    // the Bessel factor 1/(N-1), which on this 154-pixel fixture is 6.54e-03 -- the divergence the
    // registry already records for the row.
    {"VARIANCE_BIASED", 215592327.38067126},
    {"VARIANCE", 215592327.38067126},
    {"SKEWNESS", 0.45025675970449414},
    {"KURTOSIS", 1.9278887207100905},                 // pyradiomics Kurtosis is non-excess (includes +3)
    {"ENERGY", 196528957184.0},
    {"ROOT_MEAN_SQUARED", 35723.41052638121},
    {"MEAN_ABSOLUTE_DEVIATION", 12833.084499915672},
    {"ROBUST_MEAN_ABSOLUTE_DEVIATION", 10440.618496000001},
    {"INTERQUARTILE_RANGE", 26116.25},                 // approx: percentile interpolation convention
    {"P10", 16329.0},                                  // approx: percentile interpolation convention
    {"P90", 53295.0},                                  // approx: percentile interpolation convention
    {"ENTROPY", 5.54700500819408},
    {"UNIFORMITY", 0.0252993759487266},
};

// Per-feature agrees_gt frac_tolerance: exact=1e6 (rel<=1e-6), definitional=100 (rel<=1e-2),
// approx=20 (rel<=5%). A band wider than the divergence it exists to absorb hides drift (SPEC 7),
// so VARIANCE sits at the tier its known Bessel gap of 6.54e-03 needs, not at the generic approx one.
static const std::unordered_map<std::string, double> firstorder_2d_pyradiomics_ref_tols = {
    {"MEAN", 1e6}, {"MEDIAN", 1e6}, {"MIN", 1e6}, {"MAX", 1e6}, {"RANGE", 1e6},
    {"VARIANCE_BIASED", 1e6}, {"VARIANCE", 100.0}, {"SKEWNESS", 1e6}, {"KURTOSIS", 1e6}, {"ENERGY", 1e6},
    {"ROOT_MEAN_SQUARED", 1e6}, {"MEAN_ABSOLUTE_DEVIATION", 1e6},
    {"ROBUST_MEAN_ABSOLUTE_DEVIATION", 1e6}, {"INTERQUARTILE_RANGE", 20.0},
    {"P10", 20.0}, {"P90", 20.0}, {"ENTROPY", 1e6}, {"UNIFORMITY", 1e6},
};

static void compute_firstorder_pyradiomics_recipe(std::vector<std::vector<double>>& fvals)
{
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("", ""));
    LR roidata(100);
    roidata.slide_idx = -1;
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData,
        sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = 64;   // recipe: match pyradiomics binCount=64
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::IBSI].bval = false;

    PixelIntensityFeatures f;
    f.calculate(roidata, s, ds);
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);
    fvals = roidata.fvals;
}

static void assert_fo_pyradiomics(const std::vector<std::vector<double>>& fvals,
                                  Nyxus::Feature2D code, const std::string& name)
{
    SCOPED_TRACE("PYRADIOMICS_ORACLE__" + name);
    ASSERT_TRUE(firstorder_2d_pyradiomics_ref_vals.count(name)) << name;
    ASSERT_TRUE(Nyxus::agrees_gt(fvals[(int)code][0],
        firstorder_2d_pyradiomics_ref_vals.at(name),
        firstorder_2d_pyradiomics_ref_tols.at(name))) << name;
}

void test_2d_firstorder_pyradiomics()
{
    std::vector<std::vector<double>> fvals;
    compute_firstorder_pyradiomics_recipe(fvals);

    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::MEAN, "MEAN");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::MEDIAN, "MEDIAN");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::MIN, "MIN");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::MAX, "MAX");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::RANGE, "RANGE");
    // Both denominators against the one PyRadiomics number: var/N exactly, var/(N-1) within Bessel.
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::VARIANCE_BIASED, "VARIANCE_BIASED");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::VARIANCE, "VARIANCE");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::SKEWNESS, "SKEWNESS");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::KURTOSIS, "KURTOSIS");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::ENERGY, "ENERGY");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::ROOT_MEAN_SQUARED, "ROOT_MEAN_SQUARED");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION, "MEAN_ABSOLUTE_DEVIATION");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION, "ROBUST_MEAN_ABSOLUTE_DEVIATION");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::INTERQUARTILE_RANGE, "INTERQUARTILE_RANGE");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::P10, "P10");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::P90, "P90");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::ENTROPY, "ENTROPY");
    assert_fo_pyradiomics(fvals, Nyxus::Feature2D::UNIFORMITY, "UNIFORMITY");
}


void test_2d_firstorder_robust_mean_absolute_deviation_pyradiomics()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 1.044061849600000e+04));
}
