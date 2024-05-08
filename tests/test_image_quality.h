#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_feature_calculation.h"

#include "../src/nyx/features/focus_score.h"
#include "../src/nyx/features/power_spectrum.h"
#include "../src/nyx/features/saturation.h"
#include "../src/nyx/features/sharpness.h"
#include "../src/nyx/features/brisque.h"

/* GLCM dissimilarity and correlation for image quality are handled in GLCM tests */

void test_focus_score_feature() {
    
    FocusScoreFeature f;
    double truth_value = 2.810e18;

    test_feature(f, Nyxus::Feature2D::FOCUS_SCORE, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};

void test_local_focus_score_feature() {
    
    FocusScoreFeature f;
    double truth_value = 1.15292e+18;

    test_feature(f, Nyxus::Feature2D::LOCAL_FOCUS_SCORE, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};

void test_power_spectrum_feature() {
    
    PowerSpectrumFeature f;
    double truth_value = 0.0;

    test_feature(f, Nyxus::Feature2D::POWER_SPECTRUM_SLOPE, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};


void test_min_saturation_feature() {
    
    SaturationFeature f;
    double truth_value = 0.1875;

    test_feature(f, Nyxus::Feature2D::MIN_SATURATION, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};


void test_max_saturation_feature() {
    
    SaturationFeature f;
    double truth_value = 0.166667;

    test_feature(f, Nyxus::Feature2D::MAX_SATURATION, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};


void test_sharpness_feature() {
    
    SharpnessFeature f;
    double truth_value = 2.19047;

    test_feature(f, Nyxus::Feature2D::SHARPNESS, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};


void test_brisque_feature() {
    
    BrisqueFeature f;
    double truth_value = 0.563;

    test_feature(f, Nyxus::Feature2D::BRISQUE, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};
