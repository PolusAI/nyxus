#pragma once

#include "test_2d_glszm_common.h"   // gtest, <string>, <vector>, GLSZMFeature, the settings recipe and the ROI loader
#include "test_ref_vals.h"          // ref_vals_map

// PyRadiomics goldens for the 2D GLSZM zone connectivity (SPEC 6.4 provenance).
// tool=pyradiomics 3.0.1 (SimpleITK 2.3.1); env=nyxus_oracle (conda);
// recipe=glszm.pyradiomics_connectivity_2d, i.e. each fixture below read as one 2D ROI at binWidth=1
// with force2D=True. Every fixture's smallest ROI intensity is 1, so PyRadiomics' grey levels are the
// intensities themselves, which is what Nyxus computes in IBSI mode;
// generator=tests/vetting/oracles/gen_glszm_pyradiomics.py, which also re-verifies every table here.
//
// A GLSZM zone is an 8-connected set of same-level pixels. The first three fixtures hold zones whose
// pixels can be reached from the zone's first pixel in raster order only through a step to the west,
// north or north-east, so a walk that steps only east, south-east, south and south-west splits them
// into more, smaller zones and every feature moves. The fourth is the negative case: same-level pixels
// that touch only across a gap or through a pixel outside the mask, which must stay separate zones.
//
// Fifteen features assert at SPEC 7's exact tier, an ABSOLUTE 1e-9 band. GLSZM_ZE goes through
// Nyxus::fast_log10 where PyRadiomics takes a double log2, and asserts at the same rel=4e-3 band
// test_2d_glszm_mirp.h gives it for the same reason.

struct Glszm2dFixture
{
    std::vector<std::vector<unsigned>> intensity;
    std::vector<std::vector<unsigned>> mask;   // empty: every pixel is in the ROI
};

// glszm_2d_docs_2x3 -- the image of the GLSZM page. The level-1 zone starts at the top-right pixel and
// needs two steps west to reach the bottom-left one: one zone of 4 and one of 2, not 3 + 1 + 2.
static const Glszm2dFixture glszm_2d_docs_2x3
{
    {{2, 2, 1},
     {1, 1, 1}},
    {}
};
static const ref_vals_map<double> glszm_2d_pyradiomics_docs_2x3_ref_vals
{
    {"GLSZM_SAE", 0.15625},
    {"GLSZM_LAE", 10.0},
    {"GLSZM_GLN", 1.0},
    {"GLSZM_GLNN", 0.5},
    {"GLSZM_SZN", 1.0},
    {"GLSZM_SZNN", 0.5},
    {"GLSZM_ZP", 0.3333333333333333},
    {"GLSZM_GLV", 0.25},
    {"GLSZM_ZV", 1.0},
    {"GLSZM_ZE", 0.9999999999999993},
    {"GLSZM_LGLZE", 0.625},
    {"GLSZM_HGLZE", 2.5},
    {"GLSZM_SALGLE", 0.0625},
    {"GLSZM_SAHGLE", 0.53125},
    {"GLSZM_LALGLE", 8.5},
    {"GLSZM_LAHGLE", 16.0}
};

// glszm_2d_pyrad_5x5 -- the example of the PyRadiomics GLSZM documentation: 11 zones, where an
// east/south-east/south/south-west walk finds 13.
static const Glszm2dFixture glszm_2d_pyrad_5x5
{
    {{5, 2, 5, 4, 4},
     {3, 3, 3, 1, 3},
     {2, 1, 1, 1, 3},
     {4, 2, 2, 2, 3},
     {3, 5, 3, 3, 2}},
    {}
};
static const ref_vals_map<double> glszm_2d_pyradiomics_pyrad_5x5_ref_vals
{
    {"GLSZM_SAE", 0.5912373737373737},
    {"GLSZM_LAE", 7.7272727272727275},
    {"GLSZM_GLN", 2.4545454545454546},
    {"GLSZM_GLNN", 0.2231404958677686},
    {"GLSZM_SZN", 3.909090909090909},
    {"GLSZM_SZNN", 0.35537190082644626},
    {"GLSZM_ZP", 0.44},
    {"GLSZM_GLV", 1.6859504132231407},
    {"GLSZM_ZV", 2.56198347107438},
    {"GLSZM_ZE", 3.0271691184406153},
    {"GLSZM_LGLZE", 0.18893939393939393},
    {"GLSZM_HGLZE", 13.0},
    {"GLSZM_SALGLE", 0.05895693041526375},
    {"GLSZM_SAHGLE", 9.962045454545454},
    {"GLSZM_LALGLE", 2.438308080808081},
    {"GLSZM_LAHGLE", 53.63636363636363}
};

// glszm_2d_hook_6x6 -- the outer level-1 zone runs down the left edge and along the bottom row, then
// turns north-east into the right edge and climbs it. 5 zones, where the forward walk finds 6.
static const Glszm2dFixture glszm_2d_hook_6x6
{
    {{1, 2, 2, 2, 2, 1},
     {1, 2, 1, 1, 2, 1},
     {1, 2, 1, 3, 2, 1},
     {1, 2, 3, 3, 2, 1},
     {1, 2, 2, 2, 2, 1},
     {1, 1, 1, 1, 1, 3}},
    {}
};
static const ref_vals_map<double> glszm_2d_pyradiomics_hook_6x6_ref_vals
{
    {"GLSZM_SAE", 0.24635374149659867},
    {"GLSZM_LAE", 88.0},
    {"GLSZM_GLN", 1.8},
    {"GLSZM_GLNN", 0.36},
    {"GLSZM_SZN", 1.4},
    {"GLSZM_SZNN", 0.28},
    {"GLSZM_ZP", 0.1388888888888889},
    {"GLSZM_GLV", 0.8},
    {"GLSZM_ZV", 36.160000000000004},
    {"GLSZM_ZE", 2.321928094887361},
    {"GLSZM_LGLZE", 0.49444444444444446},
    {"GLSZM_HGLZE", 4.8},
    {"GLSZM_SALGLE", 0.048057571176618796},
    {"GLSZM_SAHGLE", 2.0271927437641724},
    {"GLSZM_LALGLE", 56.822222222222216},
    {"GLSZM_LAHGLE", 221.6}
};

// glszm_2d_apart_3x7 -- the negative case. The four level-1 corners of the left 3 x 3 block are two
// steps apart and stay four zones. The right 3 x 3 block is all 4, but its middle column is outside
// the mask, so the two level-4 columns stay two zones. 8 zones over 18 pixels: a walk that reached
// further than the 8-neighbourhood, or through pixels outside the ROI, would find fewer.
static const Glszm2dFixture glszm_2d_apart_3x7
{
    {{1, 3, 1, 2, 4, 4, 4},
     {3, 3, 3, 2, 4, 4, 4},
     {1, 3, 1, 2, 4, 4, 4}},
    {{1, 1, 1, 1, 1, 0, 1},
     {1, 1, 1, 1, 1, 0, 1},
     {1, 1, 1, 1, 1, 0, 1}}
};
static const ref_vals_map<double> glszm_2d_pyradiomics_apart_3x7_ref_vals
{
    {"GLSZM_SAE", 0.5466666666666666},
    {"GLSZM_LAE", 7.0},
    {"GLSZM_GLN", 2.75},
    {"GLSZM_GLNN", 0.34375},
    {"GLSZM_SZN", 3.25},
    {"GLSZM_SZNN", 0.40625},
    {"GLSZM_ZP", 0.4444444444444444},
    {"GLSZM_GLV", 1.609375},
    {"GLSZM_ZV", 1.9375},
    {"GLSZM_ZE", 1.7499999999999987},
    {"GLSZM_LGLZE", 0.5607638888888888},
    {"GLSZM_HGLZE", 6.125},
    {"GLSZM_SALGLE", 0.5057638888888889},
    {"GLSZM_SAHGLE", 1.045},
    {"GLSZM_LALGLE", 1.2690972222222223},
    {"GLSZM_LAHGLE", 69.125}
};

static const double glszm_2d_pyradiomics_abs_tolerance = 1e-9;
static const double glszm_2d_pyradiomics_ze_frac_tolerance = 250.;

/// @brief Featurises a fixture in IBSI mode and checks all sixteen features against PyRadiomics
static void assert_glszm_2d_pyradiomics (const Glszm2dFixture& fx, const ref_vals_map<double>& gt)
{
    std::vector<NyxusPixel> intensity, mask;
    for (size_t y = 0; y < fx.intensity.size(); y++)
        for (size_t x = 0; x < fx.intensity[y].size(); x++)
        {
            intensity.push_back ({ x, y, fx.intensity[y][x] });
            mask.push_back ({ x, y, fx.mask.empty() ? 1u : fx.mask[y][x] });
        }

    LR roidata;
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());

    GLSZMFeature f;
    f.calculate (roidata, make_glszm2d_settings (true/*ibsi*/, 0));
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    const std::pair<Feature2D, const char*> features[] =
    {
        {Feature2D::GLSZM_SAE, "GLSZM_SAE"}, {Feature2D::GLSZM_LAE, "GLSZM_LAE"},
        {Feature2D::GLSZM_GLN, "GLSZM_GLN"}, {Feature2D::GLSZM_GLNN, "GLSZM_GLNN"},
        {Feature2D::GLSZM_SZN, "GLSZM_SZN"}, {Feature2D::GLSZM_SZNN, "GLSZM_SZNN"},
        {Feature2D::GLSZM_ZP, "GLSZM_ZP"}, {Feature2D::GLSZM_GLV, "GLSZM_GLV"},
        {Feature2D::GLSZM_ZV, "GLSZM_ZV"}, {Feature2D::GLSZM_ZE, "GLSZM_ZE"},
        {Feature2D::GLSZM_LGLZE, "GLSZM_LGLZE"}, {Feature2D::GLSZM_HGLZE, "GLSZM_HGLZE"},
        {Feature2D::GLSZM_SALGLE, "GLSZM_SALGLE"}, {Feature2D::GLSZM_SAHGLE, "GLSZM_SAHGLE"},
        {Feature2D::GLSZM_LALGLE, "GLSZM_LALGLE"}, {Feature2D::GLSZM_LAHGLE, "GLSZM_LAHGLE"}
    };
    for (const auto& [id, name] : features)
    {
        const double got = roidata.fvals[(int)id][0];
        if (std::string(name) == "GLSZM_ZE")
            ASSERT_TRUE (agrees_gt (got, gt.at(name), glszm_2d_pyradiomics_ze_frac_tolerance)) << name << " = " << got;
        else
            ASSERT_NEAR (got, gt.at(name), glszm_2d_pyradiomics_abs_tolerance) << name;
    }
}

void test_2d_glszm_docs_2x3_pyradiomics()
{
    assert_glszm_2d_pyradiomics (glszm_2d_docs_2x3, glszm_2d_pyradiomics_docs_2x3_ref_vals);
}

void test_2d_glszm_pyrad_5x5_pyradiomics()
{
    assert_glszm_2d_pyradiomics (glszm_2d_pyrad_5x5, glszm_2d_pyradiomics_pyrad_5x5_ref_vals);
}

void test_2d_glszm_hook_6x6_pyradiomics()
{
    assert_glszm_2d_pyradiomics (glszm_2d_hook_6x6, glszm_2d_pyradiomics_hook_6x6_ref_vals);
}

void test_2d_glszm_apart_3x7_pyradiomics()
{
    assert_glszm_2d_pyradiomics (glszm_2d_apart_3x7, glszm_2d_pyradiomics_apart_3x7_ref_vals);
}
