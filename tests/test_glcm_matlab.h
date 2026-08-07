#pragma once

// GLCM features whose registry rows read status=vetted, oracle=matlab
// (oracle_coverage.csv, target_test=test_glcm_matlab.h): ASM, CONTRAST, CORRELATION, ENERGY, HOM1
// and their angle-averaged _AVE twins. They were asserted in test_glcm_regression.h; SPEC 2 keeps
// one kind per file, so they are split out here. The config is the family's default path
// (GREYDEPTH=100, offset 1, asymmetric matrix) - the one MATLAB graycomatrix/graycoprops is matched
// on - which is why these ten and not the IBSI-path features live here.
//
// PROVENANCE RECORD MISSING (SPEC 6.4): the values are read from the shared golden maps in
// test_glcm_regression.h, which carry no MATLAB version, config or generator. The registry is the
// authority on what vets these features; the map names and the missing provenance are tracked in
// not_covered.md section C, and golden-table renaming is a separate tree-wide pass.

#include "test_glcm_regression.h"   // shared fixture: assert_glcm_feature + the golden maps

void test_glcm_asm_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ASM, "GLCM_ASM");
}
void test_glcm_contrast_matlab()
{
   assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CONTRAST, "GLCM_CONTRAST");
}
void test_glcm_correlation_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CORRELATION, "GLCM_CORRELATION");
}
void test_glcm_energy_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ENERGY, "GLCM_ENERGY");
}
void test_glcm_hom1_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_HOM1, "GLCM_HOM1");
}
void test_glcm_asm_ave_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ASM_AVE, "GLCM_ASM_AVE");
}
void test_glcm_contrast_ave_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CONTRAST_AVE, "GLCM_CONTRAST_AVE");
}
void test_glcm_correlation_ave_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CORRELATION_AVE, "GLCM_CORRELATION_AVE");
}
void test_glcm_energy_ave_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ENERGY_AVE, "GLCM_ENERGY_AVE");
}
void test_glcm_hom1_ave_matlab()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_HOM1_AVE, "GLCM_HOM1_AVE");
}
