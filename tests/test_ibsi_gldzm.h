#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/features/gldzm.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// Digital phantom values for intensity based features
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021. https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
static std::unordered_map<std::string, float> ibsi_gldzm_gtruth
{
    {"GLDZM_SDE",		0.946}, // Small distance emphasis
    {"GLDZM_LDE",       1.21},  // Large distance emphasis
    {"GLDZM_LGLZE",     0.371}, // Low grey level zone emphasis
    {"GLDZM_HGLZE",     16.4},  // High grey level zone emphasis
    {"GLDZM_HGLZE",     0.367}, // Small distance low grey level emphasis
    {"GLDZM_SDHGLE",    15.2},  // Small distance high grey level emphasis
    {"GLDZM_LDLGLE",    0.386}, // Large distance low grey level emphasis
    {"GLDZM_LDHGLE",    21.3},  // Large distance high grey level emphasis
    {"GLDZM_GLNU",      1.41},  // Grey level non-uniformity
    {"GLDZM_GLNUN",     0.323}, // Normalised grey level non-uniformity
    {"GLDZM_ZDNU",      3.79},  // Zone distance non-uniformity
    {"GLDZM_ZDNUN",     0.898}, // Normalised zone distance non-uniformity
    {"GLDZM_ZP",        0.24},  // Zone percentage
    {"GLDZM_GLV",       3.97},  // Grey level variance
    {"GLDZM_ZDV",       0.051}, // Zone distance variance
    {"GLDZM_ZDE",       1.73}   // Zone distance entropy
};

void test_ibsi_gldzm_matrix()
{
    // Activate the IBSI compliance mode
    Environment::ibsi_compliance = true;

    // Load a test image
    LR roidata;
    load_masked_test_roi_data (roidata, ibsi_fig3_17a_gldzm_sample_image_int, ibsi_fig3_17a_gldzm_sample_image_mask, sizeof(ibsi_fig3_17a_gldzm_sample_image_mask) / sizeof(NyxusPixel));

    // In this test, we only calculate and examine the GLDZ-matrix without calculating features
    GLDZMFeature f;

    // Have the feature object to create the GLDZM matrix kit (matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and GLDZM's dimensions)
    std::vector<PixIntens> greysLUT;
    SimpleMatrix<unsigned int> GLDZM;
    int Ng,	// number of grey levels
        Nd;	// maximum number of non-zero dependencies
    ASSERT_NO_THROW(f.prepare_GLDZM_matrix_kit (GLDZM, Ng, Nd, greysLUT, roidata));

    // Count discrepancies
    int n_mismatches = 0;
    for (int g = 0; g < Ng; g++)
        for (int d = 0; d < Nd; d++)
        {
            auto gtruth = ibsi_fig3_17c_gldzm_ground_truth [g * Nd + d];
            auto actual = GLDZM.yx (g,d);
            if (gtruth != actual)
            {
                n_mismatches++;
                std::cout << "GLDZ-matrix mismatch! Expecting [g=" << g << ", d=" << d << "] = " << gtruth << " not " << actual << "\n";
            }
        }

    ASSERT_TRUE(n_mismatches == 0);
}

void test_ibsi_gldzm_feature (const AvailableFeatures& feature, const std::string& feature_name)
{
    // Activate the IBSI compliance mode
    Environment::ibsi_compliance = true;

    // Chck if ground truth is available for the feature
    ASSERT_TRUE (ibsi_gldzm_gtruth.count(feature_name) > 0);

    double total = 0;

    //==== image 1

    // Load data (slice #1)
    LR roidata1;
    load_masked_test_roi_data (roidata1, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask, sizeof(ibsi_phantom_z1_intensity) / sizeof(NyxusPixel));

    // Calculate features
    GLDZMFeature f1;
    ASSERT_NO_THROW(f1.calculate(roidata1));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];

    // Diagnostic
    //  std::cout << "image #1: " << feature_name << "=" << roidata1.fvals[feature][0] << "\n";
    //  std::cout << "running average=" << total << "\n";

    //==== image 2

    // Load data (slice #2)
    LR roidata2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask, sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    // Calculate features
    GLDZMFeature f2;
    ASSERT_NO_THROW(f2.calculate(roidata2));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];

    // Diagnostic
    //  std::cout << "image #2: " << feature_name << "=" << roidata2.fvals[feature][0] << "\n";
    //  std::cout << "running average=" << total / 2. << "\n";

    //==== image 3

    // Load data (slice #3)
    LR roidata3;
    load_masked_test_roi_data(roidata3, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask, sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    // Calculate features
    GLDZMFeature f3;
    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    total += roidata3.fvals[feature][0];

    // Diagnostic
    //  std::cout << "image #3: " << feature_name << "=" << roidata3.fvals[feature][0] << "\n";
    //  std::cout << "running average=" << total / 3. << "\n";

    //==== image 4

    // Load data (slice #4)
    LR roidata4;
    load_masked_test_roi_data(roidata4, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask, sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    // Calculate features
    GLDZMFeature f4;
    ASSERT_NO_THROW(f4.calculate(roidata4));

    // Initialize per-ROI feature value buffer with zeros
    roidata4.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f4.save_value(roidata4.fvals);

    total += roidata4.fvals[feature][0];

    // Diagnostic
    //  std::cout << "image #4: " << feature_name << "=" << roidata4.fvals[feature][0] << "\n";
    //  std::cout << "running average=" << total / 4. << "\n";

    // Check the feature values vs ground truth
    double aveTotal = total / 4.0;
    ASSERT_TRUE (agrees_gt(aveTotal, ibsi_gldzm_gtruth[feature_name], 2.));
}

void test_ibsi_GLDZM_matrix_correctness()
{
    test_ibsi_gldzm_matrix();
}

void test_ibsi_GLDZM_SDE()
{
    test_ibsi_gldzm_feature(GLDZM_SDE, "GLDZM_SDE");
}

void test_ibsi_GLDZM_LDE()
{
    test_ibsi_gldzm_feature(GLDZM_LDE, "GLDZM_LDE");
}

void test_ibsi_GLDZM_LGLZE()
{
    test_ibsi_gldzm_feature(GLDZM_LGLZE, "GLDZM_LGLZE");
}

void test_ibsi_GLDZM_HGLZE()
{
    test_ibsi_gldzm_feature(GLDZM_HGLZE, "GLDZM_HGLZE");
}

void test_ibsi_GLDZM_SDHGLE()
{
    test_ibsi_gldzm_feature(GLDZM_SDHGLE, "GLDZM_SDHGLE");
}

void test_ibsi_GLDZM_LDLGLE()
{
    test_ibsi_gldzm_feature(GLDZM_LDLGLE, "GLDZM_LDLGLE");
}

void test_ibsi_GLDZM_LDHGLE()
{
    test_ibsi_gldzm_feature(GLDZM_LDHGLE, "GLDZM_LDHGLE");
}

void test_ibsi_GLDZM_GLNU()
{
    test_ibsi_gldzm_feature(GLDZM_GLNU, "GLDZM_GLNU");
}

void test_ibsi_GLDZM_GLNUN()
{
    test_ibsi_gldzm_feature(GLDZM_GLNUN, "GLDZM_GLNUN");
}

void test_ibsi_GLDZM_ZDNU()
{
    test_ibsi_gldzm_feature(GLDZM_ZDNU, "GLDZM_ZDNU");
}

void test_ibsi_GLDZM_ZDNUN()
{
    test_ibsi_gldzm_feature(GLDZM_ZDNUN, "GLDZM_ZDNUN");
}

void test_ibsi_GLDZM_ZP()
{
    test_ibsi_gldzm_feature(GLDZM_ZP, "GLDZM_ZP");
}

void test_ibsi_GLDZM_GLV()
{
    test_ibsi_gldzm_feature(GLDZM_GLV, "GLDZM_GLV");
}

void test_ibsi_GLDZM_ZDV()
{
    test_ibsi_gldzm_feature(GLDZM_ZDV, "GLDZM_ZDV");
}

void test_ibsi_GLDZM_ZDE()
{
    test_ibsi_gldzm_feature(GLDZM_ZDE, "GLDZM_ZDE");
}

