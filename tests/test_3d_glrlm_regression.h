#pragma once

// gtest, <iostream>, fs and the roi_cache graph arrive with test_3d_glrlm_common.h.
#include <iomanip>
#include <tuple>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "test_ref_vals.h"   // ref_vals_map, and the <string> / <vector> it already includes
#include "test_3d_glrlm_common.h"   // make_glrlm3d_settings

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
// bin_roi_3d (3d_ooc_volume.h) through matlab_grey_binning -- a FIXED 64-level count --
// rather than the -20 binCount profile the table above uses, where the bin count comes from the
// ROI's own min/max. The two therefore bin the same voxels differently and are not comparable to
// each other.
//
// Every slot of every feature in the family, not just the direction-averaged one: a base feature
// holds one value per 3D angle and its *_AVE twin holds their mean, so two per-angle errors of
// opposite sign leave the mean where it was. The 16 mean pins each equal the arithmetic mean of
// their own 13 per-angle pins to rel<=1.5e-16, which is checkable without running anything.
//
// The whole table is Nyxus' own output, so it claims no vetting (SPEC 1): it is a drift guard on
// the grey64 configuration, which nothing else in the tree exercises. The family is vetted against
// PyRadiomics under the -20 profile in test_3d_glrlm_pyradiomics.h.
//
// Regenerate with test_3d_glrlm_dump_regression() below, which prints this table too.
// Where these goldens came from: tests/vetting/audit/glrlm_3d_golden_regen.md, "grey64 table and
// the retired sweep".
static const ref_vals_map<std::vector<double>> glrlm_3d_regression_grey64_ref_vals{
    {"3GLRLM_GLN", { 5805.5128831349166, 5751.6418651857839, 5805.5128831349166, 5691.2604749887187, 5023.5925197999568, 5691.2604749887187, 5805.5128831349166, 5751.6418651857839, 5805.5128831349166, 5069.4198122460421, 5159.1255950739824, 5069.4198122460421, 5008.3592577820828 }},
    {"3GLRLM_GLNN", { 0.023710003402564444, 0.023730049200776409, 0.023710003402564444, 0.023779772345481249, 0.024439045903792433, 0.023779772345481249, 0.023710003402564444, 0.023730049200776409, 0.023710003402564444, 0.023676707357181085, 0.023707037933434344, 0.023676707357181085, 0.023780028003067645 }},
    {"3GLRLM_GLV", { 144.91508073637226, 144.86129129475947, 144.91508073637226, 143.91390236292509, 133.89874619769103, 143.91390236292509, 144.91508073637226, 144.86129129475947, 144.91508073637226, 144.39097879874467, 144.32600985229575, 144.39097879874467, 142.84085038335559 }},
    {"3GLRLM_HGLRE", { 2036.0863286434828, 2034.4413354347341, 2036.0863286434828, 2041.7886952016445, 2140.1591001965403, 2041.7886952016445, 2036.0863286434828, 2034.4413354347341, 2036.0863286434828, 2068.2689645509317, 2057.0494256042643, 2068.2689645509317, 2070.3031546160714 }},
    {"3GLRLM_LGLRE", { 0.00068634691802812141, 0.00068683860170490427, 0.00068634691802812141, 0.00068196123875489511, 0.00062867705784865329, 0.00068196123875489511, 0.00068634691802812141, 0.00068683860170490427, 0.00068634691802812141, 0.00067394352953326791, 0.00067733823553023386, 0.00067394352953326791, 0.00066976230432264213 }},
    {"3GLRLM_LRE", { 1.5668293479814583, 1.5902763452128494, 1.5668293479814583, 1.7048117259706181, 2.9865146237521647, 1.7048117259706181, 1.5668293479814583, 1.5902763452128494, 1.5668293479814583, 4.134435570501144, 2.8502159727966179, 4.134435570501144, 3.6489088940801095 }},
    {"3GLRLM_LRHGLE", { 2745.1109268750893, 2816.6771406645817, 2745.1109268750893, 2997.6187722494274, 4924.2391562396624, 2997.6187722494274, 2745.1109268750893, 2816.6771406645817, 2745.1109268750893, 6708.3172201204989, 4831.6143736788899, 6708.3172201204989, 6075.6132034262055 }},
    {"3GLRLM_LRLGLE", { 0.0013577991773710892, 0.001360280265647809, 0.0013577991773710892, 0.0014699537396934757, 0.0027565203084798182, 0.0014699537396934757, 0.0013577991773710892, 0.001360280265647809, 0.0013577991773710892, 0.0038297141558305487, 0.0025321404790105465, 0.0038297141558305487, 0.0033104804539068507 }},
    {"3GLRLM_RE", { 5.8527823565279871, 5.8758725757888692, 5.8527823565279871, 5.8989806972895886, 6.1402291690487383, 5.8989806972895886, 5.8527823565279871, 5.8758725757888692, 5.8527823565279871, 6.0819282846556231, 6.1049751995701191, 6.0819282846556231, 6.1396531960727465 }},
    {"3GLRLM_RLN", { 210053.24439770475, 206825.59587091237, 210053.24439770475, 201786.65793124196, 144672.07106579229, 201786.65793124196, 210053.24439770475, 206825.59587091237, 210053.24439770475, 172773.36070244267, 172629.57323775388, 172773.36070244267, 164500.6917744478 }},
    {"3GLRLM_RLNN", { 0.8578678989512355, 0.85331835344343288, 0.8578678989512355, 0.84312443773186185, 0.70380855370698148, 0.84312443773186185, 0.8578678989512355, 0.85331835344343288, 0.8578678989512355, 0.80693737192304271, 0.79326152576855935, 0.80693737192304271, 0.78106039434812735 }},
    {"3GLRLM_RP", { 0.89222466767723885, 0.88319875233208955, 0.89222466767723885, 0.87209946361940294, 0.7490234375, 0.87209946361940294, 0.89222466767723885, 0.88319875233208955, 0.89222466767723885, 0.78019327192164178, 0.79298332555970152, 0.78019327192164178, 0.76744694496268662 }},
    {"3GLRLM_RV", { 0.31065029367668973, 0.30829086537405376, 0.31065029367668973, 0.38998670143266545, 1.2040981583754526, 0.38998670143266545, 0.31065029367668973, 0.30829086537405376, 0.31065029367668973, 2.4915943228009287, 1.2599422259983606, 2.4915943228009287, 1.9510434211208152 }},
    {"3GLRLM_SRE", { 0.93966351987082797, 0.93654898758192096, 0.93966351987082797, 0.93186200804813868, 0.86276257105390042, 0.93186200804813868, 0.93966351987082797, 0.93654898758192096, 0.93966351987082797, 0.91427342440869985, 0.9071933958275944, 0.91427342440869985, 0.90098267001742083 }},
    {"3GLRLM_SRHGLE", { 1954.2569455606754, 1945.7965049746131, 1954.2569455606754, 1947.0161364021153, 1943.1827323197676, 1947.0161364021153, 1954.2569455606754, 1945.7965049746131, 1954.2569455606754, 1928.0352233415795, 1903.7352301302546, 1928.0352233415795, 1908.292297877485 }},
    {"3GLRLM_SRLGLE", { 0.00061986386941450156, 0.00061804559373080856, 0.00061986386941450156, 0.00060778325195095389, 0.00048905536543503039, 0.00060778325195095389, 0.00061986386941450156, 0.00061804559373080856, 0.00061986386941450156, 0.00059610754800567688, 0.00059216730167935977, 0.00059610754800567688, 0.00057845474384322329 }},
    {"3GLRLM_GLNN_AVE", { 0.02377993717364844 }},
    {"3GLRLM_GLN_AVE", { 5495.2133238489823 }},
    {"3GLRLM_GLV_AVE", { 143.61986725320691 }},
    {"3GLRLM_HGLRE_AVE", { 2053.9119219511867 }},
    {"3GLRLM_LGLRE_AVE", { 0.00067743476998462693 }},
    {"3GLRLM_LRE_AVE", { 2.3547695512249192 }},
    {"3GLRLM_LRHGLE_AVE", { 3989.0105159164718 }},
    {"3GLRLM_LRLGLE_AVE", { 0.00210386417486348 }},
    {"3GLRLM_RE_AVE", { 5.9622730850978245 }},
    {"3GLRLM_RLNN_AVE", { 0.82433556890963733 }},
    {"3GLRLM_RLN_AVE", { 191137.42635984672 }},
    {"3GLRLM_RP_AVE", { 0.84225656572904717 }},
    {"3GLRLM_RV_AVE", { 0.92595605841666806 }},
    {"3GLRLM_SRE_AVE", { 0.92268935049690348 }},
    {"3GLRLM_SRHGLE_AVE", { 1939.533367077448 }},
    {"3GLRLM_SRLGLE_AVE", { 0.00059869274430696139 }},
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
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    // (2) properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/, 0/*channel*/));
    // (3) voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/, 0/*channel*/));
    // (4) buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (5) feature settings
    Fsettings s = make_glrlm3d_settings (grey_depth, glrlm_grey_depth);

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
