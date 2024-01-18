#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/features/ngldm.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// Digital phantom values for intensity based features
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021. https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
static std::unordered_map<std::string, float> ibsi_ngldm_gtruth
{
	{"NGLDM_LDE",		0.158},	// Low dependence emphasis, p.120, consensus - strong
	{"NGLDM_HDE",		19.2},	// High dependence emphasis, p.121
	{"NGLDM_LGLCE",		0.702},	// Low grey level count emphasis, p. 121
	{"NGLDM_HGLCE",		7.49},	// High grey level count emphasis, p. 122
	{"NGLDM_LDLGLE",	0.0473},	// Low dependence low grey level emphasis, p. 122
	{"NGLDM_LDHGLE",	3.06},	// Low dependence high grey level emphasis, p. 123
	{"NGLDM_HDLGLE",	17.6},	// High dependence low grey level emphasis, p. 123
	{"NGLDM_HDHGLE",	49.5},	// High dependence high grey level emphasis, 124
	{"NGLDM_GLNU",		10.2},	// Grey level non-uniformity, p. 124
	{"NGLDM_GLNUN",		0.562},	// Normalised grey level non-uniformity, p. 125
	{"NGLDM_DCNU",		3.96},	// Dependence count non-uniformity, p. 125
	{"NGLDM_DCNUN",		0.212},	// Normalised dependence count non-uniformity
	//--not in IBSI-- {"NGLDM_GLM",		-1},    // Grey level mean
	{"NGLDM_GLV",		2.7},	// Grey level variance, p. 127
	//--not in IBSI-- {"NGLDM_DCM",		-1},    // Dependency count mean
	{"NGLDM_DCP", 1.0},	    // Dependence count percentage, p. 126
	{"NGLDM_DCV",		2.73},	// Dependence count variance, p. 127
	{"NGLDM_DCENT",		2.71},	// Dependence count energy, p. 128
	{"NGLDM_DCENE",		0.17}	// Dependence count entropy, p. 128
};

//
// Tests calculating the NGLD-matrix with the IBSI mode enabled, using IBSI-provided ground truth
//

void test_ibsi_ngld_matrix_ibsi ()
{
    // Activate the IBSI compliance mode
    Environment::ibsi_compliance = true;

    // Load a test image
    LR roidata;
    load_masked_test_roi_data (roidata, ibsi_fig3_19_ngldm_sample_image_int, ibsi_fig3_19_ngldm_sample_image_mask, sizeof(ibsi_fig3_19_ngldm_sample_image_mask) / sizeof(NyxusPixel));

    // In this test, we only calculate and examine the NGLD-matrix without calculating features
    NGLDMfeature f;
    
    // Have the feature object to create the NGLDM matrix kit (matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and NGLDM's dimensions)
    std::vector<PixIntens> greyLevelsLUT;
    SimpleMatrix<unsigned int> NGLDM;
    int Ng = -1,	// number of grey levels
        Nr = -1;	// maximum number of non-zero dependencies
    ASSERT_NO_THROW (f.prepare_NGLDM_matrix_kit (NGLDM, greyLevelsLUT, Ng, Nr, roidata));

    // Count discrepancies
    int n_mismatches = 0;
    for (int g=0; g<Ng; g++)
        for (int r = 0; r < Nr; r++)
        {
            auto gtruth = ibsi_fig3_19_ngldm_ground_truth[g * Nr + r];
            auto actual = NGLDM.yx(g, r);
            if (gtruth != actual)
            {
                n_mismatches++;
                std::cout << "NGLD-matrix #1 mismatch! Expecting [g=" << g << ", r=" << r << "] = " << gtruth << " not " << actual << "\n";
            }
        }

    ASSERT_TRUE (n_mismatches == 0);
}

//
// Tests calculating the NGLD-matrix with the IBSI mode disabled, using community-provided ground truth
//

void test_ibsi_ngld_matrix_nonibsi()
{
    // Disable the IBSI compliance mode
    Environment::ibsi_compliance = false;

    // Load a test image
    LR roidata;
    load_masked_test_roi_data (roidata, nonibsi_rayryeng_ngldm_sample_image_int, nonibsi_rayryeng_ngldm_sample_image_mask, sizeof(nonibsi_rayryeng_ngldm_sample_image_mask) / sizeof(NyxusPixel));

    // In this test, we only calculate and examine the NGLD-matrix without calculating features
    NGLDMfeature f;

    // Have the feature object to create the NGLDM matrix kit (matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and NGLDM's dimensions)
    std::vector<PixIntens> greyLevelsLUT;
    SimpleMatrix<unsigned int> NGLDM;
    int Ng = -1,	// number of grey levels
        Nr = -1;	// maximum number of non-zero dependencies
    ASSERT_NO_THROW(f.prepare_NGLDM_matrix_kit(NGLDM, greyLevelsLUT, Ng, Nr, roidata));

    // Count discrepancies
    int n_mismatches = 0;
    for (int g = 0; g < Ng; g++)
        for (int r = 0; r < Nr; r++)
        {
            auto gtruth = nonibsi_rayryeng_ngldm_ground_truth[g * Nr + r];
            auto actual = NGLDM.yx(g, r);
            if (gtruth != actual)
            {
                n_mismatches++;
                std::cout << "NGLD-matrix #2 mismatch! Expecting [g=" << g << ", r=" << r << "] = " << gtruth << " not " << actual << "\n";
            }
        }

    ASSERT_TRUE(n_mismatches == 0);
}

void test_ibsi_ngldm_feature (const Feature2D& feature_, const std::string& feature_name) 
{
    int feature = int(feature_);

    // Activate the IBSI compliance mode
    Environment::ibsi_compliance = true;

    // Check if ground truth is available for the feature
    ASSERT_TRUE(ibsi_ngldm_gtruth.count(feature_name) > 0);

    double total = 0;

    //==== image 1

    // Load data (slice #1)
    LR roidata1;
    load_masked_test_roi_data (roidata1, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask, sizeof(ibsi_phantom_z1_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f1;
    ASSERT_NO_THROW (f1.calculate(roidata1));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value (roidata1.fvals);

    total += roidata1.fvals[feature][0];

    std::cout << "image #1: " << feature_name << "=" << roidata1.fvals[feature][0] << "\n";
    std::cout << "running total=" << total << "\n";

    //==== image 2

    // Load data (slice #2)
    LR roidata2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask, sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f2;
    ASSERT_NO_THROW(f2.calculate(roidata2));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];

    std::cout << "image #2: " << feature_name << "=" << roidata2.fvals[feature][0] << "\n";
    std::cout << "running total=" << total/2. << "\n";

    //==== image 3

    // Load data (slice #3)
    LR roidata3;
    load_masked_test_roi_data (roidata3, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask, sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f3;
    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value (roidata3.fvals);

    total += roidata3.fvals[feature][0];

    std::cout << "image #3: " << feature_name << "=" << roidata3.fvals[feature][0] << "\n";
    std::cout << "running total=" << total/3. << "\n";

    //==== image 4

    // Load data (slice #4)
    LR roidata4;
    load_masked_test_roi_data (roidata4, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask, sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f4;
    ASSERT_NO_THROW(f4.calculate(roidata4));

    // Initialize per-ROI feature value buffer with zeros
    roidata4.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f4.save_value (roidata4.fvals);

    total += roidata4.fvals[feature][0];

    std::cout << "image #4: " << feature_name << "=" << roidata4.fvals[feature][0] << "\n";
    std::cout << "running total=" << total/4. << "\n";

    // Check the feature values vs ground truth
    double aveTotal = total / 4.0;
    ASSERT_TRUE(agrees_gt(aveTotal, ibsi_ngldm_gtruth[feature_name], 2.));
}

void test_ibsi_NGLDM_matrix_correctness_IBSI()
{
    test_ibsi_ngld_matrix_ibsi();
}

void test_ibsi_NGLDM_matrix_correctness_NONIBSI()
{
    test_ibsi_ngld_matrix_nonibsi();
}



