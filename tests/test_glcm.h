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

// dig. phantom values for intensity based features
static std::unordered_map<std::string, float> glcm_values {
    {"GLCM_DIFFERENCEAVERAGE", 1.42},
    {"GLCM_DIFFERENCEVARIANCE", 2.9},
    {"GLCM_DIFFERENCEENTROPY", 1.4},
    {"GLCM_SUMAVERAGE", 4.28},
    {"GLCM_SUMVARIANCE", 5.47},
    {"GLCM_SUMENTROPY", 1.6}, 
    {"GLCM_ANGULAR2NDMOMENT", 0.368},
    {"GLCM_CONTRAST", 5.28},
    {"GLCM_INVERSEDIFFERENCEMOMENT", 0.619},
    {"GLCM_CORRELATION", -0.0121},
    {"GLCM_INFOMEAS1", -0.155},
    {"GLCM_INFOMEAS2", 0.487}

};


void test_glcm_feature(const AvailableFeatures& feature, const std::string& feature_name) {
    double total = 0;
    
    LR roidata;
    // Calculate features
    GLCMFeature f;
    Environment::ibsi_compliance = false; 
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
    Environment::ibsi_compliance = false; //<<< New!
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
    Environment::ibsi_compliance = false; //<<< New!
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
    Environment::ibsi_compliance = false; 
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

    std::cout << "value: " << total / 16 << std::endl;

    ASSERT_TRUE(agrees_gt(total / 16, glcm_values[feature_name], 100.));
}


void test_glcm_difference_average()
{
    test_glcm_feature(GLCM_DIFFERENCEAVERAGE, "GLCM_DIFFERENCEAVERAGE");
}


void test_glcm_difference_variance()
{
    test_glcm_feature(GLCM_DIFFERENCEVARIANCE, "GLCM_DIFFERENCEVARIANCE");
}


void test_glcm_difference_entropy()
{
    test_glcm_feature(GLCM_DIFFERENCEENTROPY, "GLCM_DIFFERENCEENTROPY");
}

void test_glcm_sum_average()
{
    test_glcm_feature(GLCM_SUMAVERAGE, "GLCM_SUMAVERAGE");
}

void test_glcm_sum_variance()
{
    test_glcm_feature(GLCM_SUMVARIANCE, "GLCM_SUMVARIANCE");
}

void test_glcm_sum_entropy()
{
   test_glcm_feature(GLCM_SUMENTROPY, "GLCM_SUMENTROPY");
}

void test_glcm_angular_2d_moment()
{
    test_glcm_feature(GLCM_ANGULAR2NDMOMENT, "GLCM_ANGULAR2NDMOMENT");
}

void test_glcm_contrast()
{
   test_glcm_feature(GLCM_CONTRAST, "GLCM_CONTRAST");
}

void test_glcm_correlation()
{
    test_glcm_feature(GLCM_CORRELATION, "GLCM_CORRELATION");
}

void test_glcm_infomeas1()
{
   test_glcm_feature(GLCM_INFOMEAS1, "GLCM_INFOMEAS1");
}

void test_glcm_infomeas2()
{
   test_glcm_feature(GLCM_INFOMEAS2, "GLCM_INFOMEAS2");
}

void test_glcm_inversed_difference_moment() {
    test_glcm_feature(GLCM_INVERSEDIFFERENCEMOMENT, "GLCM_INVERSEDIFFERENCEMOMENT");
}
