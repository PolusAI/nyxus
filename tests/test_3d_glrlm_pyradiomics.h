#pragma once

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"

// PyRadiomics 3.0.1 on the compat phantom (compat_int/compat_int_mri.nii +
// compat_seg/compat_seg_liver.nii, label 1) at binCount 20, recipe glrlm3d.pyradiomics_bincount20.
// Nyxus side: GREYDEPTH=100, IBSI=false, GLRLM_GREYDEPTH=-20 (negative activates radiomics
// binCount-based binning, so the magnitude is the bin count).
//
// PyRadiomics reports one value per feature over its whole direction set, i.e. the Nyxus *_AVE
// aggregation over the 13 3D angles -- so each golden below is the reference for both the per-angle
// base feature (through calc_ave) and the stored *_AVE feature.
//
// Regenerate with tests/vetting/oracles/gen_glrlm3d_pyradiomics.py, which also re-verifies every
// pin. See tests/vetting/audit/glrlm_3d_golden_regen.md.

static ref_vals_map<double> glrlm_3d_pyradiomics_ref_vals
{
    {"3GLRLM_GLN", 406.68709120394277},     // Case-1_original_glrlm_GrayLevelNonUniformity
    {"3GLRLM_GLNN", 0.09722976558135092},   // Case-1_original_glrlm_GrayLevelNonUniformityNormalized
    {"3GLRLM_GLV", 9.100102904831404},      // Case-1_original_glrlm_GrayLevelVariance
    {"3GLRLM_HGLRE", 130.25347348795043},   // Case-1_original_glrlm_HighGrayLevelRunEmphasis
    {"3GLRLM_LRE", 1.5538285862328314},     // Case-1_original_glrlm_LongRunEmphasis
    {"3GLRLM_LRHGLE", 200.98033929654184},  // Case-1_original_glrlm_LongRunHighGrayLevelEmphasis
    {"3GLRLM_LRLGLE", 0.01863138831176311}, // Case-1_original_glrlm_LongRunLowGrayLevelEmphasis
    {"3GLRLM_LGLRE", 0.012578735424633676}, // Case-1_original_glrlm_LowGrayLevelRunEmphasis
    {"3GLRLM_RE", 4.228290966541947},       // Case-1_original_glrlm_RunEntropy
    {"3GLRLM_RLN", 3309.7814564084974},     // Case-1_original_glrlm_RunLengthNonUniformity
    {"3GLRLM_RLNN", 0.7807974007564221},    // Case-1_original_glrlm_RunLengthNonUniformityNormalized
    {"3GLRLM_RP", 0.8714583333333334},      // Case-1_original_glrlm_RunPercentage
    {"3GLRLM_RV", 0.19950155996777244},     // Case-1_original_glrlm_RunVariance
    {"3GLRLM_SRE", 0.9003824440228139},     // Case-1_original_glrlm_ShortRunEmphasis
    {"3GLRLM_SRHGLE", 117.56903884692184},  // Case-1_original_glrlm_ShortRunHighGrayLevelEmphasis
    {"3GLRLM_SRLGLE", 0.011465297979291003} // Case-1_original_glrlm_ShortRunLowGrayLevelEmphasis
};

// Nyxus reproduces this tool to double precision on 15 of the 16 quantities. The exception is run
// entropy, the family's only sum over logarithms, which Nyxus evaluates through fast_log10 with an
// EPSILON guard: measured 3.9e-4 away, so it is held to 5e-3 and everything else to 1e-9. Same shape
// and same cause as the 2D family. See tests/vetting/audit/glrlm_3d_pyradiomics_vetting_report.md.
static double glrlm_3d_pyradiomics_frac_tolerance (const std::string& feature_name)
{
    static const std::unordered_set<std::string> log_based { "3GLRLM_RE" };
    return log_based.count (feature_name) ? 200. : 1.e9;
}

void test_3d_glrlm_matrix_correctness_pyradiomics()
{
    // data (data and gt source: pyradiomics web page)

    std::vector<PixIntens> rawVolume =
    {
        // z=0
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        // z=1
        5, 2, 5, 4, 4,
        3, 3, 3, 1, 3,
        2, 1, 1, 1, 3,
        4, 2, 2, 2, 3,
        3, 5, 3, 3, 2,
        // z=2
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };

    SimpleCube <PixIntens> D(rawVolume, 5/*width*/, 5/*height*/, 3/*depth*/);
    PixIntens zeroI = 0;
    // --- unique intensities
    std::unordered_set<PixIntens> U (rawVolume.begin(), rawVolume.end());
    U.erase (0);
    // --- sorted non-zero (i.e. non-mask) intensities
    std::vector<PixIntens> I (U.begin(), U.end());
    std::sort (I.begin(), I.end());

    // zones

    std::vector <std::pair<PixIntens, int>> zones;
    AngleShift ash = {0, 0, 1}; // layout: dz,dy,dx
    D3_GLRLM_feature::gather_rl_zones (zones, ash, D, zeroI);

    // zone stats

    int maxZoneArea = 0;    // matrix width 
    for (const std::pair<PixIntens, int>& zo : zones)
        maxZoneArea = (std::max)(maxZoneArea, zo.second);

    // GLRLM
    SimpleMatrix <int> P;
    P.allocate (maxZoneArea /*width*/, I.size() /*height*/);
    P.fill (0);

    // --iterate zones and fill the matrix
    for (const auto& zone : zones)
    {
        // row of P-matrix
        auto itr = std::find (I.begin(), I.end(), zone.first);
        int row = (int) (itr - I.begin());

        // column of P-matrix
        int col = zone.second - 1;	// need a 0-based index
        auto& k = P.xy (col, row);
        k++;
    }

    //
    // Expecting the following GLRLM as the GT:
    // 
    //               rl=1   rl=2   rl=3
    //
    // [inten=1]     1      0      1
    // [inten=1]     3      0      1
    // [inten=1]     4      1      1
    // [inten=1]     1      1      0
    // [inten=1]     3      0      0
    // 
    //
    ASSERT_TRUE(P.yx(0, 0) == 1);     ASSERT_TRUE(P.yx(0, 1) == 0);   ASSERT_TRUE(P.yx(0, 2) == 1);
    ASSERT_TRUE(P.yx(1, 0) == 3);     ASSERT_TRUE(P.yx(1, 1) == 0);   ASSERT_TRUE(P.yx(1, 2) == 1);
    ASSERT_TRUE(P.yx(2, 0) == 4);     ASSERT_TRUE(P.yx(2, 1) == 1);   ASSERT_TRUE(P.yx(2, 2) == 1);
    ASSERT_TRUE(P.yx(3, 0) == 1);     ASSERT_TRUE(P.yx(3, 1) == 1);   ASSERT_TRUE(P.yx(3, 2) == 0);
    ASSERT_TRUE(P.yx(4, 0) == 3);     ASSERT_TRUE(P.yx(4, 1) == 0);   ASSERT_TRUE(P.yx(4, 2) == 0);
}

void assert_3d_glrlm_feature_pyradiomics(const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = glrlm_3d_pyradiomics_ref_vals.find(fname);
    ASSERT_TRUE(iter != glrlm_3d_pyradiomics_ref_vals.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // (2) mock the 3D workflow

    Environment e;

    // slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    // properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

    // voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

    // buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (3) common feature extraction settings

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    // (4) GLRLM-specific feature settings mocking the pyRadiomics recipe above

    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // (7) aggregate angled subfeatures
    double atot = f.calc_ave (r.fvals[fcode]);

    // (8) verdict
    ASSERT_TRUE(agrees_gt(atot, glrlm_3d_pyradiomics_ref_vals[fname],
                          glrlm_3d_pyradiomics_frac_tolerance(fname)));
}

// Vet the direction-averaged (_AVE) 3D GLRLM features vs PyRadiomics. save_value stores
// fvals[X_AVE][0] = calc_ave(angled_X) -- exactly the quantity the base test asserts == PyRadiomics
// (assert_3d_glrlm_feature_pyradiomics: atot = calc_ave(fvals[X])). So reading the _AVE slot directly and
// comparing to the same GT table is a direct PyRadiomics assertion on the _AVE feature. One workflow
// run covers all 16.
void test_3d_glrlm_ave_pyradiomics()
{
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = -20;  // radiomics binCount-based grey binning

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    using F = Nyxus::Feature3D;
    struct AvePair { F ave; const char* gt; };
    std::vector<AvePair> aves = {
        {F::GLRLM_GLN_AVE, "3GLRLM_GLN"},     {F::GLRLM_GLNN_AVE, "3GLRLM_GLNN"},
        {F::GLRLM_GLV_AVE, "3GLRLM_GLV"},     {F::GLRLM_HGLRE_AVE, "3GLRLM_HGLRE"},
        {F::GLRLM_LGLRE_AVE, "3GLRLM_LGLRE"}, {F::GLRLM_LRE_AVE, "3GLRLM_LRE"},
        {F::GLRLM_LRHGLE_AVE, "3GLRLM_LRHGLE"}, {F::GLRLM_LRLGLE_AVE, "3GLRLM_LRLGLE"},
        {F::GLRLM_RLN_AVE, "3GLRLM_RLN"},     {F::GLRLM_RLNN_AVE, "3GLRLM_RLNN"},
        {F::GLRLM_RV_AVE, "3GLRLM_RV"},       {F::GLRLM_SRE_AVE, "3GLRLM_SRE"},
        {F::GLRLM_SRHGLE_AVE, "3GLRLM_SRHGLE"}, {F::GLRLM_SRLGLE_AVE, "3GLRLM_SRLGLE"},
        // RP is only in its mathematical bound [0,1] at this recipe's binCount binning; it exceeds 1
        // at positive GLRLM_GREYDEPTH values (see the audit report), so this row vets it here and
        // says nothing about other configs.
        {F::GLRLM_RP_AVE, "3GLRLM_RP"},       {F::GLRLM_RE_AVE, "3GLRLM_RE"},
    };

    // Every 3D GLRLM feature the build exposes has to be asserted here -- the per-angle ones through
    // the golden table, the aggregated ones through the list above. Without this, a feature added to
    // the family later is vetted by nothing while this test still passes over the entries it has.
    std::unordered_set<int> covered_aves;
    for (const auto& a : aves)
        covered_aves.insert((int)a.ave);
    for (const auto& [name, code] : Nyxus::UserFacing_3D_featureNames)
    {
        if (name.rfind("3GLRLM_", 0) != 0)
            continue;
        const std::string suffix = "_AVE";
        bool is_ave = name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        if (is_ave)
            ASSERT_TRUE(covered_aves.count((int)code)) << name << " is not asserted by this test";
        else
            ASSERT_TRUE(glrlm_3d_pyradiomics_ref_vals.count(name)) << name << " has no pyradiomics golden";
    }

    for (auto& a : aves)
    {
        double v = r.fvals[(int)a.ave][0];
        ASSERT_TRUE(agrees_gt(v, glrlm_3d_pyradiomics_ref_vals[a.gt],
                              glrlm_3d_pyradiomics_frac_tolerance(a.gt))) << a.gt << "_AVE = " << v
            << " vs pyradiomics " << glrlm_3d_pyradiomics_ref_vals[a.gt];
    }
}

void test_3d_glrlm_gln_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_GLN, "3GLRLM_GLN");
}

void test_3d_glrlm_glnn_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_GLNN, "3GLRLM_GLNN");
}

void test_3d_glrlm_glv_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_GLV, "3GLRLM_GLV");
}

void test_3d_glrlm_hglre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_HGLRE, "3GLRLM_HGLRE");
}

void test_3d_glrlm_lre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LRE, "3GLRLM_LRE");
}

void test_3d_glrlm_lrhgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LRHGLE, "3GLRLM_LRHGLE");
}

void test_3d_glrlm_lrlgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LRLGLE, "3GLRLM_LRLGLE");
}

void test_3d_glrlm_lglre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LGLRE, "3GLRLM_LGLRE");
}

void test_3d_glrlm_re_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RE, "3GLRLM_RE");
}

void test_3d_glrlm_rln_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RLN, "3GLRLM_RLN");
}

void test_3d_glrlm_rlnn_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RLNN, "3GLRLM_RLNN");
}

void test_3d_glrlm_rp_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RP, "3GLRLM_RP");
}

void test_3d_glrlm_rv_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RV, "3GLRLM_RV");
}

void test_3d_glrlm_sre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_SRE, "3GLRLM_SRE");
}

void test_3d_glrlm_srhgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_SRHGLE, "3GLRLM_SRHGLE");
}

void test_3d_glrlm_srlgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_SRLGLE, "3GLRLM_SRLGLE");
}








