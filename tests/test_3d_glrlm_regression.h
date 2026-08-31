#pragma once

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <tuple>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"   // ref_vals_map, and the <string> / <vector> it already includes

// Drift guards on the segmented phantom (ut_inten.nii + ut_mask57.nii, label 57), at binCount-20
// grey binning, averaged over the 13 3D angles. Nyxus' own output, so these claim no oracle
// (SPEC 1); the implementation is vetted against PyRadiomics on the compat phantom in
// test_3d_glrlm_pyradiomics.h.
//
// Regenerate with test_3d_glrlm_dump_regression() below.
//
// Why these values and not the ones this file used to carry:
// tests/vetting/audit/glrlm_3d_pyradiomics_vetting_report.md.
static const ref_vals_map<double> glrlm_3d_regression_ref_vals{
    {"3GLRLM_SRE", 0.84064583359383949},
    {"3GLRLM_LRE", 4.7269894613234555},
    {"3GLRLM_LGLRE", 0.045266965539734721},
    {"3GLRLM_HGLRE", 156.55736675409963},
    {"3GLRLM_SRLGLE", 0.024058168696417533},
    {"3GLRLM_SRHGLE", 141.01413107252367},
    {"3GLRLM_LRLGLE", 1.0699954232575346},
    {"3GLRLM_LRHGLE", 478.13693595593821},
    {"3GLRLM_GLN", 9934.4008683562715},
    {"3GLRLM_GLNN", 0.051578758146456932},
    {"3GLRLM_RLN", 132077.24363689235},
    {"3GLRLM_RLNN", 0.67562186885577269},
    {"3GLRLM_RP", 0.70306894015499444},
    {"3GLRLM_GLV", 30.204626150216143},
    {"3GLRLM_RV", 2.5998881222894781},
    {"3GLRLM_RE", 5.2580435451573697}
};

// The same phantom at the "grey64" profile: GLRLM_GREYDEPTH=+64, a positive value, which routes
// bin_intensities_3d (texture_feature.h) through matlab_grey_binning -- a FIXED 64-level count --
// rather than the -20 binCount profile the table above uses, where the bin count comes from the
// ROI's own min/max. The two therefore bin the same voxels differently and are not comparable to
// each other.
//
// Every slot of every feature in the family, not just the direction-averaged one: a base feature
// holds one value per 3D angle and its *_AVE twin holds their mean, so two per-angle errors of
// opposite sign leave the mean where it was. The 16 mean pins each equal the arithmetic mean of
// their own 13 per-angle pins to rel<=2.9e-16, which is checkable without running anything.
//
// The whole table is Nyxus' own output, so it claims no vetting (SPEC 1): it is a drift guard on
// the grey64 configuration, which nothing else in the tree exercises. The family is vetted against
// PyRadiomics under the -20 profile in test_3d_glrlm_pyradiomics.h.
//
// Regenerate with test_3d_glrlm_dump_regression() below, which prints this table too.
// Where these goldens came from: tests/vetting/audit/glrlm_3d_golden_regen.md, "grey64 table and
// the retired sweep".
static const ref_vals_map<std::vector<double>> glrlm_3d_regression_grey64_ref_vals{
    {"3GLRLM_GLN", { 9658.5508820166888, 7864.294342649604, 9658.5508820166888, 7275.7038435192935, 5294.9954101273315, 7275.7038435192935, 9658.5508820166888, 7864.294342649604, 9658.5508820166888, 7924.8644339692491, 6390.2164957135401, 7924.8644339692491, 5811.0057485452298 }},
    {"3GLRLM_GLNN", { 0.034399366337045732, 0.029205112718638744, 0.034399366337045732, 0.027687433760253038, 0.024499349506437538, 0.027687433760253038, 0.034399366337045732, 0.029205112718638744, 0.034399366337045732, 0.032597318270314542, 0.026907081062577012, 0.032597318270314542, 0.025617200443242946 }},
    {"3GLRLM_GLV", { 327.7802163448697, 292.56886329307753, 327.7802163448697, 278.31013715040581, 216.56464294608497, 278.31013715040581, 327.7802163448697, 292.56886329307753, 327.7802163448697, 320.19290160789063, 272.27397969014555, 320.19290160789063, 254.88131322305716 }},
    {"3GLRLM_HGLRE", { 1775.7218005748334, 1831.3071324059151, 1775.7218005748334, 1859.6880280082198, 2035.5211541308854, 1859.6880280082198, 1775.7218005748334, 1831.3071324059151, 1775.7218005748334, 1821.6395271354179, 1885.0107287824433, 1821.6395271354179, 1922.2664256744843 }},
    {"3GLRLM_LGLRE", { 0.12853636684847319, 0.10051498661830537, 0.12853636684847319, 0.089851644520867982, 0.049513382538602824, 0.089851644520867982, 0.12853636684847319, 0.10051498661830537, 0.12853636684847319, 0.11989559650661127, 0.084295059820187923, 0.11989559650661127, 0.072161258942153098 }},
    {"3GLRLM_LRE", { 14.308515298617765, 22.86616804937648, 14.308515298617765, 25.293035999695562, 76.245336097127634, 25.293035999695562, 14.308515298617765, 22.86616804937648, 14.308515298617765, 21.584738024136826, 36.843514728917185, 21.584738024136826, 40.795238934932108 }},
    {"3GLRLM_LRHGLE", { 2406.8495353964177, 2556.7349727790611, 2406.8495353964177, 2753.8799908668848, 4756.7726532425231, 2753.8799908668848, 2406.8495353964177, 2556.7349727790611, 2406.8495353964177, 5925.944733746308, 4461.5637410944373, 5925.944733746308, 5678.3747839887146 }},
    {"3GLRLM_LRLGLE", { 12.943326782170793, 21.435979552767879, 12.943326782170793, 23.741684325170944, 73.40752988640304, 23.741684325170944, 12.943326782170793, 21.435979552767879, 12.943326782170793, 17.946921938258999, 34.234109125406505, 17.946921938258999, 37.410444484700058 }},
    {"3GLRLM_RE", { 6.1895191308639665, 6.2191739841548701, 6.1895191308639665, 6.2427463984415068, 6.3170415653715084, 6.2427463984415068, 6.1895191308639665, 6.2191739841548701, 6.1895191308639665, 6.4130842950857003, 6.4123119828378545, 6.4130842950857003, 6.442818000694567 }},
    {"3GLRLM_RLN", { 192919.13010324919, 193629.31793908155, 192919.13010324919, 188199.01541974276, 139897.28560852827, 188199.01541974276, 192919.13010324919, 193629.31793908155, 192919.13010324919, 158721.69890668575, 162887.15052296498, 158721.69890668575, 154513.36159407513 }},
    {"3GLRLM_RLNN", { 0.68709021787129709, 0.71906846433455962, 0.68709021787129709, 0.71618469982396971, 0.64728903986770936, 0.71618469982396971, 0.68709021787129709, 0.71906846433455962, 0.68709021787129709, 0.65286943124084074, 0.68586373655939981, 0.65286943124084074, 0.68115571148860488 }},
    // 3GLRLM_RP is runs/voxels and cannot exceed 1. Four of these thirteen angles do, at
    // 1.0231; the mean below stays at 0.9401 and hides it. The pins record what Nyxus
    // computes today so the eventual fix shows up as a diff -- the defect itself is filed,
    // see tests/vetting/audit/glrlm_3d_pyradiomics_vetting_report.md.
    {"3GLRLM_RP", { 1.0231204815764925, 0.98121939132462688, 1.0231204815764925, 0.95754139458955223, 0.78754664179104472, 0.95754139458955223, 1.0231204815764925, 0.98121939132462688, 1.0231204815764925, 0.8858806553171642, 0.86539470615671643, 0.8858806553171642, 0.82657999067164178 }},
    {"3GLRLM_RV", { 10.456897437611142, 18.678574666900062, 10.456897437611142, 20.895781173091144, 69.744874729144186, 20.895781173091144, 10.456897437611142, 18.678574666900062, 10.456897437611142, 16.447303344414838, 31.459970552104785, 16.447303344414838, 34.894220469305615 }},
    {"3GLRLM_SRE", { 0.84579294864193277, 0.86319730506307568, 0.84579294864193277, 0.86144018935068922, 0.82810378265611795, 0.86144018935068922, 0.84579294864193277, 0.86319730506307568, 0.84579294864193277, 0.82602159233427142, 0.84585814804218318, 0.82602159233427142, 0.84275775431181899 }},
    {"3GLRLM_SRHGLE", { 1704.2599012769638, 1751.4379431521975, 1704.2599012769638, 1773.2955735300955, 1848.1386035867843, 1773.2955735300955, 1704.2599012769638, 1751.4379431521975, 1704.2599012769638, 1698.037480359684, 1744.4559054513027, 1698.037480359684, 1771.7804216516631 }},
    {"3GLRLM_SRLGLE", { 0.02688849995052962, 0.020762832669439898, 0.02688849995052962, 0.013282277299082696, 0.0080087704834321174, 0.013282277299082696, 0.02688849995052962, 0.020762832669439898, 0.02688849995052962, 0.021347857736332493, 0.015116231043675595, 0.021347857736332493, 0.0067679262956016001 }},
    {"3GLRLM_GLNN_AVE", { 0.03027698660452716 }},
    {"3GLRLM_GLN_AVE", { 7866.1651094407034 }},
    {"3GLRLM_GLV_AVE", { 295.15266194934725 }},
    {"3GLRLM_HGLRE_AVE", { 1843.919606614327 }},
    {"3GLRLM_LGLRE_AVE", { 0.10312612492203121 }},
    {"3GLRLM_LRE_AVE", { 26.969695007835828 }},
    {"3GLRLM_LRHGLE_AVE", { 3615.1714395919889 }},
    {"3GLRLM_LRLGLE_AVE", { 24.851889404429883 }},
    {"3GLRLM_RE_AVE", { 6.2830967252095355 }},
    {"3GLRLM_RLNN_AVE", { 0.6876088115538187 }},
    {"3GLRLM_RLN_AVE", { 177698.02943612199 }},
    {"3GLRLM_RP_AVE", { 0.94009893441446601 }},
    {"3GLRLM_RV_AVE", { 22.305382605370099 }},
    {"3GLRLM_SRE_AVE", { 0.84624689639030193 }},
    {"3GLRLM_SRHGLE_AVE", { 1740.5351176831971 }},
    {"3GLRLM_SRLGLE_AVE", { 0.019094835618041383 }},
};

// Defined once, in test_3d_glcm_pyradiomics.h. Every 3D header that needs the segmented phantom
// forward-declares it instead of carrying its own copy -- two definitions in the single
// test_all.cc translation unit is a redefinition error, which is what keeps the remaining 3D
// snapshot headers out of the build.
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// Everything up to and including D3_GLRLM_feature::calculate() + save_value(). The two assert
// helpers and the regenerator below differ only in GLRLM_GREYDEPTH and in what they read out of
// the ROI afterwards, so the workflow they share is written once.
//
// D3_GLRLM_feature::calculate() reads exactly three settings -- SOFTNAN, IBSI and GLRLM_GREYDEPTH
// (3d_glrlm.cpp). GREYDEPTH is inert for this family; it is passed only so a caller can state the
// profile it is reproducing. GLRLM_GREYDEPTH's sign selects the binning: negative means radiomics
// binCount-based (bin count derived from the ROI's own min/max), positive means matlab's fixed
// level count, and 0 means no binning at all, i.e. raw intensities.
static void run_3d_glrlm_pipeline (Environment& e, int grey_depth, int glrlm_grey_depth, int& label_out)
{
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    // (1) slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.fpimageOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    // (2) properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
    // (3) voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
    // (4) buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (5) feature settings
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = grey_depth;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = glrlm_grey_depth;

    // (6) feature extraction
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (7) saving values
    f.save_value(r.fvals);

    label_out = label;
}

void assert_3d_glrlm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // check that the requested feature has a golden -- without this a feature absent from the table
    // would be compared against nothing
    auto iter = glrlm_3d_regression_ref_vals.find(fname);
    ASSERT_TRUE(iter != glrlm_3d_regression_ref_vals.end()) << fname;

    Environment e;
    int label = -1;
    run_3d_glrlm_pipeline (e, 100, -20, label);
    if (::testing::Test::HasFatalFailure()) return;

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // aggregate angled subfeatures (13 angles for 3D). fvals[fcode] holds the per-angle vector, so
    // reading element 0 would pin one direction and let the other twelve drift unguarded.
    D3_GLRLM_feature f;
    double atot = f.calc_ave (e.roiData[label].fvals[fcode]);

    // (8) verdict, at the band this feature's arithmetic can hold across platforms.
    //
    // rel=1e-9 for the fifteen features computed in double: Windows/MSVC and Linux/gcc reproduce
    // these values to <= 1.8e-16 of each other, so the guard has eight orders of magnitude of
    // headroom and still fails on any real change.
    //
    // 3GLRLM_RE is the exception, and not by measurement here -- it agrees exactly on those two
    // platforms too -- but by construction: it is the family's only sum over logarithms, and Nyxus
    // evaluates it through Nyxus::fast_log10 (helpers.h), which computes in FLOAT. A float carries
    // ~7 significant digits, so the last bits of RE depend on how each platform's compiler
    // evaluates that approximation, which is what a CI run on another architecture reported while
    // x86 stayed green. Its band is therefore set from that precision (rel=1e-6), still four
    // orders of magnitude tighter than the 5e-3 the oracle assertion needs and far tighter than
    // any change of definition would survive. The permanent fix is item 11 on the follow-up list:
    // compute the information measures in double, after which this can go back to 1e-9.
    const double frac = (fname == "3GLRLM_RE") ? 1e6 : 1e9;
    ASSERT_TRUE(agrees_gt(atot, iter->second, frac))
        << fname << " actual=" << atot << " pinned=" << iter->second;
}

// Regenerates every golden this file asserts -- both tables -- at full precision, in the exact
// shape each one wants. Run it with
//     runAllTests --gtest_filter=*3D_GLRLM_DUMP_REGRESSION*
// and paste each block over the table above it. These are Nyxus' own values on the ut_ phantom, so
// the only honest way to refresh them is to read them out of the same code path the assertions
// use -- which is what this does, through the same helper, at each table's own profile.
void test_3d_glrlm_dump_regression()
{
    {
        Environment e;
        int label = -1;
        run_3d_glrlm_pipeline (e, 100, -20, label);
        if (::testing::Test::HasFatalFailure()) return;

        D3_GLRLM_feature f;
        std::cout << "[3DGLRLM-REGEN]\n";
        for (const auto& nv : glrlm_3d_regression_ref_vals)
        {
            int fcode = -1;
            ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
            std::cout << "[3DGLRLM-REGEN]    {\"" << nv.first << "\", "
                      << std::setprecision(17) << f.calc_ave (e.roiData[label].fvals[fcode]) << "},\n";
        }
    }

    {
        Environment e;
        int label = -1;
        run_3d_glrlm_pipeline (e, 64, 64, label);
        if (::testing::Test::HasFatalFailure()) return;

        std::cout << "[3DGLRLM-REGEN-GREY64]\n";
        for (const auto& [name, code] : Nyxus::UserFacing_3D_featureNames)
        {
            if (name.rfind("3GLRLM_", 0) != 0)
                continue;
            const std::vector<double>& v = e.roiData[label].fvals[(int)code];
            std::cout << "[3DGLRLM-REGEN-GREY64]    {\"" << name << "\", { ";
            for (size_t i = 0; i < v.size(); i++)
                std::cout << (i ? ", " : "") << std::setprecision(17) << v[i];
            std::cout << " }},\n";
        }
    }
}

void test_3d_glrlm_sre_regression()
{
    assert_3d_glrlm_feature_regression (Nyxus::Feature3D::GLRLM_SRE, "3GLRLM_SRE");
}

void test_3d_glrlm_lre_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_LRE, "3GLRLM_LRE");
}

void test_3d_glrlm_lglre_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_LGLRE, "3GLRLM_LGLRE");
}

void test_3d_glrlm_hglre_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_HGLRE, "3GLRLM_HGLRE");
}

void test_3d_glrlm_srlgle_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_SRLGLE, "3GLRLM_SRLGLE");
}

void test_3d_glrlm_srhgle_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_SRHGLE, "3GLRLM_SRHGLE");
}

void test_3d_glrlm_lrlgle_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_LRLGLE, "3GLRLM_LRLGLE");
}

void test_3d_glrlm_lrhgle_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_LRHGLE, "3GLRLM_LRHGLE");
}

void test_3d_glrlm_gln_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_GLN, "3GLRLM_GLN");
}

void test_3d_glrlm_glnn_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_GLNN, "3GLRLM_GLNN");
}

void test_3d_glrlm_rln_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_RLN, "3GLRLM_RLN");
}

void test_3d_glrlm_rlnn_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_RLNN, "3GLRLM_RLNN");
}

void test_3d_glrlm_rp_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_RP, "3GLRLM_RP");
}

void test_3d_glrlm_glv_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_GLV, "3GLRLM_GLV");
}

void test_3d_glrlm_rv_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_RV, "3GLRLM_RV");
}

void test_3d_glrlm_re_regression()
{
    assert_3d_glrlm_feature_regression(Nyxus::Feature3D::GLRLM_RE, "3GLRLM_RE");
}

// Same shared pipeline as above, at the grey64 profile. Compares the feature's whole slot -- 13
// angled values for a base feature, one mean for an *_AVE feature -- so a per-angle move is caught
// and named even when it cancels in the mean.
static void assert_3d_glrlm_feature_grey64_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    auto iter = glrlm_3d_regression_grey64_ref_vals.find(fname);
    ASSERT_TRUE(iter != glrlm_3d_regression_grey64_ref_vals.end()) << fname;

    Environment e;
    int label = -1;
    run_3d_glrlm_pipeline (e, 64, 64, label);
    if (::testing::Test::HasFatalFailure()) return;

    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    ASSERT_TRUE((int)expecting_fcode == fcode);

    const std::vector<double>& actual = e.roiData[label].fvals[fcode];
    const std::vector<double>& expected = iter->second;
    ASSERT_EQ(expected.size(), actual.size()) << fname;

    // Same two bands, and for the same reasons, as the table above: rel=1e-9 for the arithmetic
    // done in double, rel=1e-6 for RE, whose sum over logarithms goes through the float-precision
    // Nyxus::fast_log10.
    const double frac = (fname == "3GLRLM_RE" || fname == "3GLRLM_RE_AVE") ? 1e6 : 1e9;
    for (size_t i = 0; i < expected.size(); i++)
    {
        SCOPED_TRACE(fname + " element " + std::to_string(i));
        EXPECT_TRUE(agrees_gt(actual[i], expected[i], frac))
            << fname << "[" << i << "] actual=" << actual[i] << " pinned=" << expected[i];
    }
}

void test_3d_glrlm_gln_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_GLN, "3GLRLM_GLN"); }
void test_3d_glrlm_glnn_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_GLNN, "3GLRLM_GLNN"); }
void test_3d_glrlm_glv_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_GLV, "3GLRLM_GLV"); }
void test_3d_glrlm_hglre_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_HGLRE, "3GLRLM_HGLRE"); }
void test_3d_glrlm_lglre_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LGLRE, "3GLRLM_LGLRE"); }
void test_3d_glrlm_lre_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LRE, "3GLRLM_LRE"); }
void test_3d_glrlm_lrhgle_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LRHGLE, "3GLRLM_LRHGLE"); }
void test_3d_glrlm_lrlgle_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LRLGLE, "3GLRLM_LRLGLE"); }
void test_3d_glrlm_re_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RE, "3GLRLM_RE"); }
void test_3d_glrlm_rln_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RLN, "3GLRLM_RLN"); }
void test_3d_glrlm_rlnn_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RLNN, "3GLRLM_RLNN"); }
void test_3d_glrlm_rp_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RP, "3GLRLM_RP"); }
void test_3d_glrlm_rv_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RV, "3GLRLM_RV"); }
void test_3d_glrlm_sre_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_SRE, "3GLRLM_SRE"); }
void test_3d_glrlm_srhgle_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_SRHGLE, "3GLRLM_SRHGLE"); }
void test_3d_glrlm_srlgle_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_SRLGLE, "3GLRLM_SRLGLE"); }
void test_3d_glrlm_gln_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_GLN_AVE, "3GLRLM_GLN_AVE"); }
void test_3d_glrlm_glnn_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_GLNN_AVE, "3GLRLM_GLNN_AVE"); }
void test_3d_glrlm_glv_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_GLV_AVE, "3GLRLM_GLV_AVE"); }
void test_3d_glrlm_hglre_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_HGLRE_AVE, "3GLRLM_HGLRE_AVE"); }
void test_3d_glrlm_lglre_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LGLRE_AVE, "3GLRLM_LGLRE_AVE"); }
void test_3d_glrlm_lre_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LRE_AVE, "3GLRLM_LRE_AVE"); }
void test_3d_glrlm_lrhgle_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LRHGLE_AVE, "3GLRLM_LRHGLE_AVE"); }
void test_3d_glrlm_lrlgle_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_LRLGLE_AVE, "3GLRLM_LRLGLE_AVE"); }
void test_3d_glrlm_re_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RE_AVE, "3GLRLM_RE_AVE"); }
void test_3d_glrlm_rln_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RLN_AVE, "3GLRLM_RLN_AVE"); }
void test_3d_glrlm_rlnn_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RLNN_AVE, "3GLRLM_RLNN_AVE"); }
void test_3d_glrlm_rp_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RP_AVE, "3GLRLM_RP_AVE"); }
void test_3d_glrlm_rv_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_RV_AVE, "3GLRLM_RV_AVE"); }
void test_3d_glrlm_sre_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_SRE_AVE, "3GLRLM_SRE_AVE"); }
void test_3d_glrlm_srhgle_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_SRHGLE_AVE, "3GLRLM_SRHGLE_AVE"); }
void test_3d_glrlm_srlgle_ave_grey64_regression() { assert_3d_glrlm_feature_grey64_regression(Nyxus::Feature3D::GLRLM_SRLGLE_AVE, "3GLRLM_SRLGLE_AVE"); }
