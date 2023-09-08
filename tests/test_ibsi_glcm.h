#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map>

// Digital phantom values for intensity based features
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021. Dataset: dig phantom. Aggr. method: 2D, averaged)
static std::unordered_map<std::string, float> IBSI_glcm_values {
    {"GLCM_ACOR", 5.09},    // p. 76, consensus: very strong
    {"GLCM_ASM", 0.368},    // p. 68, consensus: very strong
    {"GLCM_CLUPROM", 79.1}, // p. 79, consensus: very strong
    {"GLCM_CLUSHADE", 7},   // p. 78, consensus: very strong
    {"GLCM_CLUTEND", 5.47}, // p. 78, consensus: very strong
    {"GLCM_CONTRAST", 5.28},    // p. 69, consensus: very strong
    {"GLCM_CORRELATION", -0.0121},  // p. 76, consensus: very strong
    {"GLCM_DIFAVE", 1.42},  // p. 64, consensus: very strong
    {"GLCM_DIFENTRO", 1.4}, // p. 65, consensus: very strong
    {"GLCM_DIFVAR", 2.9},   // p. 65, consensus: very strong
    {"GLCM_DIS", 1.42},     // p. 70, consensus: very strong
    {"GLCM_ID", 0.678},     // p. 71, consensus: very strong
    {"GLCM_IDN", 0.851},    // p. 72, consensus: very strong
    {"GLCM_IDM", 0.619},    // p. 73, consensus: very strong
    {"GLCM_IDMN", 0.899},   // p. 74, consensus: very strong
    {"GLCM_INFOMEAS1", -0.155}, // p. 80, consensus: very strong
    {"GLCM_INFOMEAS2", 0.487},  // p. 81, consensus: very strong
    {"GLCM_IV", 0.0567},    // p. 75, consensus: very strong
    {"GLCM_JAVE", 2.14},    // p. 62, consensus: very strong
    {"GLCM_JE", 2.05},      // p. 63, consensus: very strong
    {"GLCM_JMAX", 0.519},   // p. 61, consensus: very strong
    {"GLCM_JVAR", 2.69},    // p. 63, consensus: very strong
    {"GLCM_SUMAVERAGE", 4.28},  // p. 66, consensus: very strong
    {"GLCM_SUMENTROPY", 1.6},   // p. 67, consensus: very strong
    {"GLCM_SUMVARIANCE", 5.47}  // p. 67, consensus: very strong
};


void test_ibsi_glcm_feature(const AvailableFeatures& feature, const std::string& feature_name) {
    double total = 0;

    LR roidata;
    // Calculate features
    GLCMFeature f;
    Environment::ibsi_compliance = true;
    GLCMFeature::angles = {0, 45, 90, 135};

    // image 1
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    total += roidata.fvals[feature][1];
    total += roidata.fvals[feature][2];
    total += roidata.fvals[feature][3];

    // image 2
    // Calculate features
    LR roidata1;
    // Calculate features
    GLCMFeature f1;
    //GLCMFeature::n_levels = 6;
    Environment::ibsi_compliance = true; //<<< New!
    GLCMFeature::angles = {0, 45, 90, 135};

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];
    total += roidata1.fvals[feature][1];
    total += roidata1.fvals[feature][2];
    total += roidata1.fvals[feature][3];

    // image 3
    // Calculate features

    LR roidata2;
    // Calculate features
    GLCMFeature f2;
    //GLCMFeature::n_levels = 6;
    Environment::ibsi_compliance = true; //<<< New!
    GLCMFeature::angles = {0, 45, 90, 135};

    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];
    total += roidata2.fvals[feature][1];
    total += roidata2.fvals[feature][2];
    total += roidata2.fvals[feature][3];

    // image 4
    // Calculate features

    LR roidata3;
    // Calculate features
    GLCMFeature f3;
    Environment::ibsi_compliance = true;
    GLCMFeature::angles = {0, 45, 90, 135};

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += roidata3.fvals[feature][0];
    total += roidata3.fvals[feature][1];
    total += roidata3.fvals[feature][2];
    total += roidata3.fvals[feature][3];

    ASSERT_TRUE(agrees_gt(total / 16, IBSI_glcm_values[feature_name], 100.));
}

void test_ibsi_glcm_ACOR()
{
    test_ibsi_glcm_feature(GLCM_ACOR, "GLCM_ACOR");
}

void test_ibsi_glcm_CLUPROM()
{
    test_ibsi_glcm_feature(GLCM_CLUPROM, "GLCM_CLUPROM");
}

void test_ibsi_glcm_CLUSHADE()
{
    test_ibsi_glcm_feature(GLCM_CLUSHADE, "GLCM_CLUSHADE");
}

void test_ibsi_glcm_CLUTEND()
{
    test_ibsi_glcm_feature(GLCM_CLUTEND, "GLCM_CLUTEND");
}

void test_ibsi_glcm_difference_average()
{
    test_ibsi_glcm_feature(GLCM_DIFAVE, "GLCM_DIFAVE");
}

void test_ibsi_glcm_difference_entropy()
{
    test_ibsi_glcm_feature(GLCM_DIFENTRO, "GLCM_DIFENTRO");
}

void test_ibsi_glcm_difference_variance()
{
    test_ibsi_glcm_feature(GLCM_DIFVAR, "GLCM_DIFVAR");
}

void test_ibsi_glcm_DIS()
{
    test_ibsi_glcm_feature(GLCM_DIS, "GLCM_DIS");
}

void test_ibsi_glcm_ID()
{
    test_ibsi_glcm_feature(GLCM_ID, "GLCM_ID");
}

void test_ibsi_glcm_IDN()
{
    test_ibsi_glcm_feature(GLCM_IDN, "GLCM_IDN");
}

void test_ibsi_glcm_IDM()
{
    test_ibsi_glcm_feature(GLCM_IDM, "GLCM_IDM");
}

void test_ibsi_glcm_IDMN()
{
    test_ibsi_glcm_feature(GLCM_IDMN, "GLCM_IDMN");
}

void test_ibsi_glcm_angular_2d_moment()
{
    test_ibsi_glcm_feature(GLCM_ASM, "GLCM_ASM");
}

void test_ibsi_glcm_contrast()
{
   test_ibsi_glcm_feature(GLCM_CONTRAST, "GLCM_CONTRAST");
}

void test_ibsi_glcm_correlation()
{
    test_ibsi_glcm_feature(GLCM_CORRELATION, "GLCM_CORRELATION");
}

void test_ibsi_glcm_infomeas1()
{
   test_ibsi_glcm_feature(GLCM_INFOMEAS1, "GLCM_INFOMEAS1");
}

void test_ibsi_glcm_infomeas2()
{
   test_ibsi_glcm_feature(GLCM_INFOMEAS2, "GLCM_INFOMEAS2");
}

void test_ibsi_glcm_IV()
{
    test_ibsi_glcm_feature(GLCM_IV, "GLCM_IV");
}

void test_ibsi_glcm_JAVE()
{
    test_ibsi_glcm_feature(GLCM_JAVE, "GLCM_JAVE");
}

void test_ibsi_glcm_JE()
{
    test_ibsi_glcm_feature(GLCM_JE, "GLCM_JE");
}

void test_ibsi_glcm_JMAX()
{
    test_ibsi_glcm_feature(GLCM_JMAX, "GLCM_JMAX");
}

void test_ibsi_glcm_JVAR()
{
    test_ibsi_glcm_feature(GLCM_JVAR, "GLCM_JVAR");
}

void test_ibsi_glcm_inversed_difference_moment() {
    test_ibsi_glcm_feature(GLCM_IDM, "GLCM_IDM");
}

void test_ibsi_glcm_sum_average()
{
    test_ibsi_glcm_feature(GLCM_SUMAVERAGE, "GLCM_SUMAVERAGE");
}

void test_ibsi_glcm_sum_entropy()
{
   test_ibsi_glcm_feature(GLCM_SUMENTROPY, "GLCM_SUMENTROPY");
}

void test_ibsi_glcm_sum_variance()
{
    test_ibsi_glcm_feature(GLCM_SUMVARIANCE, "GLCM_SUMVARIANCE");
}
