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
// Calculated at 100 grey levels, offset 1, and asymmetric cooc matrix
static std::unordered_map<std::string, float> glcm_values 
{
    {"GLCM_ACOR", 1340.12},
    {"GLCM_ASM", 0.29},
    {"GLCM_CLUPROM", 6401030.04}, 
    {"GLCM_CLUSHADE", 20646.08},
    {"GLCM_CLUTEND", 1563.9},
    {"GLCM_CONTRAST", 1444.81},
    {"GLCM_CORRELATION", 0.01}, 
    {"GLCM_DIFAVE", 24.33}, 
    {"GLCM_DIFENTRO", 1.75},
    {"GLCM_DIFVAR", 771.58}, 
    {"GLCM_DIS", 24.33},
    {"GLCM_ID", 0.51},
    {"GLCM_IDN", 0.84},
    {"GLCM_IDM", 0.5},
    {"GLCM_IDMN", 0.9},
    {"GLCM_INFOMEAS1", -0.24},
    {"GLCM_INFOMEAS2", 0.6},
    {"GLCM_IV", 0.00056},
    {"GLCM_JAVE", 33.015},
    {"GLCM_JE", 2.26},
    {"GLCM_JMAX", 0.45},
    {"GLCM_JVAR", 818.5607},
    {"GLCM_SUMAVERAGE", 68.28},
    {"GLCM_SUMENTROPY", 1.952},
    {"GLCM_SUMVARIANCE", 1563.904}
};

void test_glcm_feature(const Feature2D& feature_, const std::string& feature_name) 
{
    // Set feature's state
    GLCMFeature::n_levels = 100;
    GLCMFeature::offset = 1;
    GLCMFeature::symmetric_glcm = false;
    Environment::ibsi_compliance = false;
    GLCMFeature::angles = { 0, 45, 90, 135 };

    int feature = int(feature_);

    double total = 0;

    // image 1

     LR roidata;
    GLCMFeature f;   
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

    LR roidata1;
    GLCMFeature f1;
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

    LR roidata2;
    GLCMFeature f2;
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
    
    LR roidata3;
    GLCMFeature f3;
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

    // Verdict
    ASSERT_TRUE(agrees_gt(total / 16, glcm_values[feature_name], 100.));
}

void test_glcm_ACOR()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ACOR, "GLCM_ACOR");
}

void test_glcm_angular_2d_moment()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ASM, "GLCM_ASM");
}

void test_glcm_CLUPROM()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUPROM, "GLCM_CLUPROM");
}

void test_glcm_CLUSHADE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUSHADE, "GLCM_CLUSHADE");
}

void test_glcm_CLUTEND()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUTEND, "GLCM_CLUTEND");
}

void test_glcm_contrast()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_CONTRAST, "GLCM_CONTRAST");
}

void test_glcm_correlation()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CORRELATION, "GLCM_CORRELATION");
}

void test_glcm_difference_average()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFAVE, "GLCM_DIFAVE");
}

void test_glcm_difference_entropy()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFENTRO, "GLCM_DIFENTRO");
}

void test_glcm_difference_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFVAR, "GLCM_DIFVAR");
}

void test_glcm_DIS()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIS, "GLCM_DIS");
}

void test_glcm_ID()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ID, "GLCM_ID");
}

void test_glcm_IDN()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDN, "GLCM_IDN");
}

void test_glcm_IDM()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDM, "GLCM_IDM");
}

void test_glcm_IDMN()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDMN, "GLCM_IDMN");
}

void test_glcm_infomeas1()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS1, "GLCM_INFOMEAS1");
}

void test_glcm_infomeas2()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS2, "GLCM_INFOMEAS2");
}

void test_glcm_IV()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IV, "GLCM_IV");
}

void test_glcm_JAVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JAVE, "GLCM_JAVE");
}

void test_glcm_JE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JE, "GLCM_JE");
}

void test_glcm_JMAX()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JMAX, "GLCM_JMAX");
}

void test_glcm_JVAR()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JVAR, "GLCM_JVAR");
}

void test_glcm_sum_average()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMAVERAGE, "GLCM_SUMAVERAGE");
}

void test_glcm_sum_entropy()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_SUMENTROPY, "GLCM_SUMENTROPY");
}

void test_glcm_sum_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMVARIANCE, "GLCM_SUMVARIANCE");
}

