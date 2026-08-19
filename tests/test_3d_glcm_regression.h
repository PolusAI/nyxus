#pragma once

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <tuple>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"   // ref_vals_map, and the <string> / <vector> it already includes

// Drift guards on the segmented phantom (ut_inten.nii + ut_mask57.nii, label 57), at 100 grey
// levels, offset 1, asymmetric cooc matrix, averaged over the 13 3D angles. Nyxus' own output, so
// these claim no oracle (SPEC 1); the implementation is vetted against PyRadiomics on the compat
// phantom in test_3d_glcm_pyradiomics.h.
//
// Regenerate with test_3d_glcm_dump_regression() below.
//
// Two properties any regenerated set must keep, both checkable by eye: ID, IDM, IDN, IDMN and JMAX
// are bounded in [0,1] by construction, and SUMVARIANCE == CLUTEND / DIS == DIFAVE hold exactly (to
// ~1e-15). Why these values replaced the ones this file used to carry:
// tests/vetting/audit/glcm_3d_pyradiomics_vetting_report.md.
static ref_vals_map<double> glcm_3d_regression_ref_vals
{
    {"3GLCM_ACOR", 2864.4036927337511},
    {"3GLCM_ASM", 0.00075632811401173821},
    {"3GLCM_CLUPROM", 17176172.731148321},
    {"3GLCM_CLUSHADE", 16474.460141543663},
    {"3GLCM_CLUTEND", 3070.5732092569574},
    {"3GLCM_CONTRAST", 105.49729387931679},
    {"3GLCM_CORRELATION", 0.93322047637351058},
    {"3GLCM_DIFAVE", 6.8734913176894725},
    {"3GLCM_DIFENTRO", 4.1849916431761667},
    {"3GLCM_DIFVAR", 55.221738206074143},
    {"3GLCM_DIS", 6.8734913176894974},
    {"3GLCM_ID", 0.27705716404219871},
    {"3GLCM_IDN", 0.939846148626171},
    {"3GLCM_IDM", 0.1933066324730974},
    {"3GLCM_IDMN", 0.99018160227583252},
    {"3GLCM_INFOMEAS1", -0.2870083573469876},
    {"3GLCM_INFOMEAS2", 0.9835240244988358},
    {"3GLCM_IV", 0.17398242245108428},
    {"3GLCM_JAVE", 46.073699428746984},
    {"3GLCM_JE", 11.316979486142573},
    {"3GLCM_JMAX", 0.0082741381486977333},
    {"3GLCM_JVAR", 794.01762578407022},
    {"3GLCM_SUMAVERAGE", 92.147398857493954},
    {"3GLCM_SUMENTROPY", 7.578343299408516},
    {"3GLCM_SUMVARIANCE", 3070.5732092569574}
};

// Defined once, in test_3d_glcm_pyradiomics.h. Every 3D header that needs the segmented phantom
// forward-declares it instead of carrying its own copy -- two definitions in the single
// test_all.cc translation unit is a redefinition error, which is what kept this file out of the
// build (and with it, the 25 assertions below).
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// frac_tolerance = 1e9, i.e. rel=1e-9: these are Nyxus' own values pinned to full precision, so a
// drift guard should catch any change at all. The previous 10% band would have passed a value off
// by a factor of 1.1.
//
// The two information measures get rel=1e-6 instead. Both are a difference of two entropies of
// similar size (INFOMEAS1 is (HXY-HXY1)/HX), and the entropies come out of fast_log10, whose core
// is a float-precision polynomial -- its last bits depend on how the compiler rounds and contracts
// that float arithmetic, and the cancellation amplifies the difference into the result. The table
// was dumped on MSVC; Apple clang reproduces INFOMEAS1 to rel 1.9e-9, which the 1e-9 band failed.
// 1e-6 keeps ~500x of headroom over that and still catches any change to the math, which moves
// these values by percent or more.
static double glcm_3d_regression_frac_tolerance (const std::string& feature_name)
{
    return (feature_name == "3GLCM_INFOMEAS1" || feature_name == "3GLCM_INFOMEAS2") ? 1.e6 : 1.e9;
}

void assert_3d_glcm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = glcm_3d_regression_ref_vals.find(fname);
    ASSERT_TRUE(iter != glcm_3d_regression_ref_vals.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
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

    // (4) GLCM-specific feature settings mocking default pyRadiomics settings

    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -100;  // intentionally negative to activate radiomics binCount-based grey-binning
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // aggregate angled subfeatures (13 angles for 3D)
    double atot = f.calc_ave(r.fvals[fcode]);

    // (7) verdict, at the per-feature drift band
    ASSERT_TRUE(agrees_gt(atot, glcm_3d_regression_ref_vals[fname], glcm_3d_regression_frac_tolerance (fname)))
        << fname;
}

// Regenerates every golden in glcm_3d_regression_ref_vals at full precision, in the exact shape the
// table wants. Run it with
//     runAllTests --gtest_filter=*3D_GLCM_DUMP_REGRESSION*
// and paste the output over the table above. These are Nyxus' own values on the ut_ phantom, so the
// only honest way to refresh them is to read them out of the same code path the assertions use --
// which is what this does, via the shared assert helper's settings.
void test_3d_glcm_dump_regression()
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
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -100;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    std::cout << "[3DGLCM-REGEN]\n";
    for (const auto& nv : glcm_3d_regression_ref_vals)
    {
        int fcode = -1;
        ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
        std::cout << "[3DGLCM-REGEN]    {\"" << nv.first << "\", "
                  << std::setprecision(17) << f.calc_ave(r.fvals[fcode]) << "},\n";
    }
}

void test_3d_glcm_acor_regression()
{
    assert_3d_glcm_feature_regression (Nyxus::Feature3D::GLCM_ACOR, "3GLCM_ACOR");
}

void test_3d_glcm_asm_regression()
{
    assert_3d_glcm_feature_regression (Nyxus::Feature3D::GLCM_ASM, "3GLCM_ASM");
}

void test_3d_glcm_cluprom_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CLUPROM, "3GLCM_CLUPROM");
}

void test_3d_glcm_clushade_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CLUSHADE, "3GLCM_CLUSHADE");
}

void test_3d_glcm_clutend_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CLUTEND, "3GLCM_CLUTEND");
}

void test_3d_glcm_contrast_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CONTRAST, "3GLCM_CONTRAST");
}

void test_3d_glcm_correlation_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CORRELATION, "3GLCM_CORRELATION");
}

void test_3d_glcm_difference_average_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIFAVE, "3GLCM_DIFAVE");
}

void test_3d_glcm_difference_entropy_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIFENTRO, "3GLCM_DIFENTRO");
}

void test_3d_glcm_difference_variance_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIFVAR, "3GLCM_DIFVAR");
}

void test_3d_glcm_dis_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIS, "3GLCM_DIS");
}

void test_3d_glcm_id_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_ID, "3GLCM_ID");
}

void test_3d_glcm_idn_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IDN, "3GLCM_IDN");
}

void test_3d_glcm_idm_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IDM, "3GLCM_IDM");
}

void test_3d_glcm_idmn_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IDMN, "3GLCM_IDMN");
}

void test_3d_glcm_infomeas1_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_INFOMEAS1, "3GLCM_INFOMEAS1");
}

void test_3d_glcm_infomeas2_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_INFOMEAS2, "3GLCM_INFOMEAS2");
}

void test_3d_glcm_iv_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IV, "3GLCM_IV");
}

void test_3d_glcm_jave_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JAVE, "3GLCM_JAVE");
}

void test_3d_glcm_je_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JE, "3GLCM_JE");
}

void test_3d_glcm_jmax_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JMAX, "3GLCM_JMAX");
}

void test_3d_glcm_jvar_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JVAR, "3GLCM_JVAR");
}

void test_3d_glcm_sum_average_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_SUMAVERAGE, "3GLCM_SUMAVERAGE");
}

void test_3d_glcm_sum_entropy_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_SUMENTROPY, "3GLCM_SUMENTROPY");
}

void test_3d_glcm_sum_variance_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_SUMVARIANCE, "3GLCM_SUMVARIANCE");
}

