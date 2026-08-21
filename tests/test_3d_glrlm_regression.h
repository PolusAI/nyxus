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
static ref_vals_map<double> glrlm_3d_regression_ref_vals{
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

// Defined once, in test_3d_glcm_pyradiomics.h. Every 3D header that needs the segmented phantom
// forward-declares it instead of carrying its own copy -- two definitions in the single
// test_all.cc translation unit is a redefinition error, which is what keeps the remaining 3D
// snapshot headers out of the build.
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void assert_3d_glrlm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // check that requested feature exists -- operator[] below would otherwise default-insert a 0
    // golden and compare against it
    auto iter = glrlm_3d_regression_ref_vals.find(fname);
    ASSERT_TRUE(iter != glrlm_3d_regression_ref_vals.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    Environment e;
    // (1) slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
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
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    // D3_GLRLM_feature bins on GLRLM_GREYDEPTH (3d_glrlm.cpp), not the generic GREYDEPTH above.
    // Left unset it defaults to 0, i.e. no binning at all, and the features run on raw intensities.
    // Negative activates radiomics binCount-based binning, so this is the same grey binning the
    // PyRadiomics test uses; at positive grey depths RP leaves its [0,1] bound (audit report).
    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = -20;

    // (6) feature extraction

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

    // (7) saving values

    f.save_value(r.fvals);

    // aggregate angled subfeatures (13 angles for 3D). fvals[fcode] holds the per-angle vector, so
    // reading element 0 would pin one direction and let the other twelve drift unguarded.
    double atot = f.calc_ave(r.fvals[fcode]);

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
    ASSERT_TRUE(agrees_gt(atot, glrlm_3d_regression_ref_vals[fname], frac))
        << fname << " actual=" << atot << " pinned=" << glrlm_3d_regression_ref_vals[fname];
}

// Regenerates every golden in glrlm_3d_regression_ref_vals at full precision, in the exact shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_GLRLM_DUMP_REGRESSION*
// and paste the output over the table above. These are Nyxus' own values on the ut_ phantom, so the
// only honest way to refresh them is to read them out of the same code path the assertions use --
// which is what this does, with the same settings the shared assert helper sets.
void test_3d_glrlm_dump_regression()
{
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
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
    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = -20;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    std::cout << "[3DGLRLM-REGEN]\n";
    for (const auto& nv : glrlm_3d_regression_ref_vals)
    {
        int fcode = -1;
        ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
        std::cout << "[3DGLRLM-REGEN]    {\"" << nv.first << "\", "
                  << std::setprecision(17) << f.calc_ave(r.fvals[fcode]) << "},\n";
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