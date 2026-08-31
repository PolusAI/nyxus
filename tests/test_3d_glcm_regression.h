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
static const ref_vals_map<double> glcm_3d_regression_ref_vals
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

// Shared by both regression assert helpers below; only grey_depth/glcm_grey_depth differ between
// the -100/binCount and +64/matlab_grey_binning profiles. ASSERT_* needs void return, so callers
// must check ::testing::Test::HasFatalFailure() before touching fvals.
static void run_3d_glcm_feature_pipeline (Environment& e, const std::string& fname,
    const Nyxus::Feature3D& expecting_fcode, int grey_depth, int glcm_grey_depth,
    int& label_out, int& fcode_out)
{
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // slide -> dataset -> prescan
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.fpimageOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    // properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

    // voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

    // buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // common feature extraction settings
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

    // GLCM-specific feature settings
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = glcm_grey_depth;   // sign selects binCount (-) vs fixed-level (+) grey-binning
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

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
    f.save_value(r.fvals);

    label_out = label;
    fcode_out = fcode;
}

void assert_3d_glcm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // check that requested feature exists
    auto iter = glcm_3d_regression_ref_vals.find(fname);
    ASSERT_TRUE(iter != glcm_3d_regression_ref_vals.end());

    Environment e;
    int label = -1, fcode = -1;
    run_3d_glcm_feature_pipeline(e, fname, expecting_fcode, 100, -100, label, fcode);
    if (::testing::Test::HasFatalFailure()) return;

    // aggregate angled subfeatures (13 angles for 3D)
    D3_GLCM_feature f;
    double atot = f.calc_ave(e.roiData[label].fvals[fcode]);

    // verdict, at rel=1e-8: a full-precision drift guard on Nyxus' own values, with headroom over
    // the fast_log10-derived (3d_glcm.cpp) compiler-rounding residuals measured on INFOMEAS1/2 and
    // DIFENTRO (up to rel 3.6e-9).
    ASSERT_TRUE(agrees_gt(atot, glcm_3d_regression_ref_vals.at(fname), 1.e8)) << fname;
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
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.fpimageOptions, e.resultOptions.need_annotation()));
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

// -----------------------------------------------------------------------------------------------
// "grey64" profile: the same phantom as above (ut_inten.nii / ut_mask57.nii, label 57), but
// GREYDEPTH=64 with GLCM_GREYDEPTH=+64 -- a positive value, which bin_intensities_3d
// (texture_feature.h) routes through matlab_grey_binning (a FIXED 64-level count) rather than the
// -100 bincount profile the rest of this file uses (radiomics_grey_binning, bin count derived from
// the ROI's own min/max).
//
// Every one of these 36 features is independently oracle-vetted against PyRadiomics elsewhere in
// test_3d_glcm_pyradiomics.h, under the -100 bincount profile (some directly, some by the
// same-value identity trick documented there). This table claims no vetting of its own (SPEC 1):
// it is a drift guard on the grey64 configuration, which nothing else in the tree exercises.
// History: tests/vetting/audit/glcm_3d_golden_regen.md, "grey64 table and the retired Wave-9 sweep".
static const ref_vals_map<std::vector<double>> glcm_3d_regression_grey64_ref_vals
{
	{ "3GLCM_ACOR_AVE", { 896.29490954682888 } },
	{ "3GLCM_ASM_AVE", { 0.21714923037245615 } },
	{ "3GLCM_CLUPROM_AVE", { 5319702.1464416385 } },
	{ "3GLCM_CLUSHADE_AVE", { 25300.544725496744 } },
	{ "3GLCM_CLUTEND_AVE", { 1830.7699876860852 } },
	{ "3GLCM_CONTRAST_AVE", { 190.39105377069197 } },
	{ "3GLCM_CORRELATION_AVE", { 0.81160543314537192 } },
	{ "3GLCM_DIFAVE_AVE", { 4.4826978263468851 } },
	{ "3GLCM_DIFENTRO_AVE", { 2.5804645841819687 } },
	{ "3GLCM_DIFVAR_AVE", { 168.82038645655439 } },
	{ "3GLCM_DIS", { 5.5662607942292563, 5.2865125777406536, 5.5662607942292563, 3.5205463112065347, 3.2058935920047036, 3.5205463112065347, 5.5662607942292563, 5.2865125777406536, 5.5662607942292563, 4.6920801473460996, 4.4260407115349301, 4.6920801473460996, 1.379816189466244 } },
	{ "3GLCM_DIS_AVE", { 4.4826978263468824 } },
	{ "3GLCM_ENERGY", { 0.20619709037207257, 0.21210950671886142, 0.20619709037207257, 0.22347189327232553, 0.229768029906674, 0.22347189327232553, 0.20619709037207257, 0.21210950671886142, 0.20619709037207257, 0.21657166507454639, 0.22249608735556295, 0.21657166507454639, 0.2415813859599367 } },
	{ "3GLCM_ENERGY_AVE", { 0.21714923037245615 } },
	{ "3GLCM_ENTROPY", { 6.1375193094015223, 6.0251268596431471, 6.1375193094015223, 5.6356261936885357, 5.5229037686481623, 5.6356261936885357, 6.1375193094015223, 6.0251268596431471, 6.1375193094015223, 5.9202631504971848, 5.8126798565436344, 5.9202631504971848, 4.8177837873733909 } },
	{ "3GLCM_ENTROPY_AVE", { 5.8358059275253087 } },
	{ "3GLCM_HOM1", { 0.63159286085358357, 0.67511464571418467, 0.63159286085358357, 0.675209797165082, 0.7178512712427948, 0.675209797165082, 0.63159286085358357, 0.67511464571418467, 0.63159286085358357, 0.65283716836970751, 0.69041197263012255, 0.65283716836970751, 0.77822827662266392 } },
	{ "3GLCM_HOM1_AVE", { 0.67070662972368189 } },
	{ "3GLCM_HOM2", { 0.59196175760722125, 0.64232872280698583, 0.59196175760722125, 0.63506326531940249, 0.68470533035896952, 0.63506326531940249, 0.59196175760722125, 0.64232872280698583, 0.59196175760722125, 0.61310667832992871, 0.65700403974706201, 0.61310667832992871, 0.75964624124070679 } },
	{ "3GLCM_IDMN_AVE", { 0.97393106857854816 } },
	{ "3GLCM_IDM_AVE", { 0.63463076728371071 } },
	{ "3GLCM_IDN_AVE", { 0.95491658377679667 } },
	{ "3GLCM_ID_AVE", { 0.67070662972368211 } },
	{ "3GLCM_INFOMEAS1_AVE", { -0.44521529228752316 } },
	{ "3GLCM_INFOMEAS2_AVE", { 0.97909293489987548 } },
	{ "3GLCM_IV_AVE", { 0.15630830580149896 } },
	{ "3GLCM_JAVE_AVE", { 22.049501181077645 } },
	{ "3GLCM_JE_AVE", { 5.83580592757035 } },
	{ "3GLCM_JMAX_AVE", { 0.46494916987187934 } },
	{ "3GLCM_JVAR_AVE", { 505.29026036419413 } },
	{ "3GLCM_SUMAVERAGE_AVE", { 44.099002362155289 } },
	{ "3GLCM_SUMENTROPY_AVE", { 4.3772246057690438 } },
	{ "3GLCM_SUMVARIANCE", { 1783.3177119109412, 1796.9249632904919, 1783.3177119109412, 1872.6626392475214, 1887.8346987420691, 1872.6626392475214, 1783.3177119109412, 1796.9249632904919, 1783.3177119109412, 1820.9267907064923, 1833.7216725021183, 1820.9267907064923, 1964.1538345421473 } },
	{ "3GLCM_SUMVARIANCE_AVE", { 1830.7699876860852 } },
	{ "3GLCM_VARIANCE", { 504.75832684863678, 506.44120833399279, 504.75832684863678, 504.89933309869377, 506.49086530867982, 504.89933309869377, 504.75832684863678, 506.44120833399279, 504.75832684863678, 504.73432274406343, 506.30964057188811, 504.73432274406343, 504.78984310590351 } },
	{ "3GLCM_VARIANCE_AVE", { 505.29026036419378 } },
};

// Same shared pipeline as above; checks every angled value against the table instead of averaging.
static void assert_3d_glcm_feature_grey64_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    auto iter = glcm_3d_regression_grey64_ref_vals.find(fname);
    ASSERT_TRUE(iter != glcm_3d_regression_grey64_ref_vals.end()) << fname;

    Environment e;
    int label = -1, fcode = -1;
    run_3d_glcm_feature_pipeline(e, fname, expecting_fcode, 64, 64, label, fcode);
    if (::testing::Test::HasFatalFailure()) return;

    const auto& actual = e.roiData[label].fvals[fcode];
    const auto& expected = iter->second;
    ASSERT_EQ(expected.size(), actual.size()) << fname;
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(agrees_gt(actual[i], expected[i], 1.e8)) << fname << "[" << i << "]";
}

void test_3d_glcm_acor_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_ACOR_AVE, "3GLCM_ACOR_AVE"); }
void test_3d_glcm_asm_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_ASM_AVE, "3GLCM_ASM_AVE"); }
void test_3d_glcm_cluprom_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_CLUPROM_AVE, "3GLCM_CLUPROM_AVE"); }
void test_3d_glcm_clushade_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_CLUSHADE_AVE, "3GLCM_CLUSHADE_AVE"); }
void test_3d_glcm_clutend_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_CLUTEND_AVE, "3GLCM_CLUTEND_AVE"); }
void test_3d_glcm_contrast_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_CONTRAST_AVE, "3GLCM_CONTRAST_AVE"); }
void test_3d_glcm_correlation_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_CORRELATION_AVE, "3GLCM_CORRELATION_AVE"); }
void test_3d_glcm_difave_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_DIFAVE_AVE, "3GLCM_DIFAVE_AVE"); }
void test_3d_glcm_difentro_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_DIFENTRO_AVE, "3GLCM_DIFENTRO_AVE"); }
void test_3d_glcm_difvar_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_DIFVAR_AVE, "3GLCM_DIFVAR_AVE"); }
void test_3d_glcm_dis_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_DIS, "3GLCM_DIS"); }
void test_3d_glcm_dis_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_DIS_AVE, "3GLCM_DIS_AVE"); }
void test_3d_glcm_energy_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_ENERGY, "3GLCM_ENERGY"); }
void test_3d_glcm_energy_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_ENERGY_AVE, "3GLCM_ENERGY_AVE"); }
void test_3d_glcm_entropy_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_ENTROPY, "3GLCM_ENTROPY"); }
void test_3d_glcm_entropy_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_ENTROPY_AVE, "3GLCM_ENTROPY_AVE"); }
void test_3d_glcm_hom1_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_HOM1, "3GLCM_HOM1"); }
void test_3d_glcm_hom1_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_HOM1_AVE, "3GLCM_HOM1_AVE"); }
void test_3d_glcm_hom2_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_HOM2, "3GLCM_HOM2"); }
void test_3d_glcm_idmn_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_IDMN_AVE, "3GLCM_IDMN_AVE"); }
void test_3d_glcm_idm_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_IDM_AVE, "3GLCM_IDM_AVE"); }
void test_3d_glcm_idn_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_IDN_AVE, "3GLCM_IDN_AVE"); }
void test_3d_glcm_id_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_ID_AVE, "3GLCM_ID_AVE"); }
void test_3d_glcm_infomeas1_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_INFOMEAS1_AVE, "3GLCM_INFOMEAS1_AVE"); }
void test_3d_glcm_infomeas2_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_INFOMEAS2_AVE, "3GLCM_INFOMEAS2_AVE"); }
void test_3d_glcm_iv_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_IV_AVE, "3GLCM_IV_AVE"); }
void test_3d_glcm_jave_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_JAVE_AVE, "3GLCM_JAVE_AVE"); }
void test_3d_glcm_je_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_JE_AVE, "3GLCM_JE_AVE"); }
void test_3d_glcm_jmax_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_JMAX_AVE, "3GLCM_JMAX_AVE"); }
void test_3d_glcm_jvar_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_JVAR_AVE, "3GLCM_JVAR_AVE"); }
void test_3d_glcm_sumaverage_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_SUMAVERAGE_AVE, "3GLCM_SUMAVERAGE_AVE"); }
void test_3d_glcm_sumentropy_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_SUMENTROPY_AVE, "3GLCM_SUMENTROPY_AVE"); }
void test_3d_glcm_sumvariance_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_SUMVARIANCE, "3GLCM_SUMVARIANCE"); }
void test_3d_glcm_sumvariance_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_SUMVARIANCE_AVE, "3GLCM_SUMVARIANCE_AVE"); }
void test_3d_glcm_variance_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_VARIANCE, "3GLCM_VARIANCE"); }
void test_3d_glcm_variance_ave_grey64_regression() { assert_3d_glcm_feature_grey64_regression(Nyxus::Feature3D::GLCM_VARIANCE_AVE, "3GLCM_VARIANCE_AVE"); }

