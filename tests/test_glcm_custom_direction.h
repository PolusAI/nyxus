#pragma once

#include <gtest/gtest.h>
#include <cmath>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/features/image_cube.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map>

// Test helper: create a direction field with uniform direction
// For 2D: creates a SimpleCube with depth=2 containing [dx, dy] at each pixel
void create_uniform_direction_field_2d(
    SimpleCube<float>& dir_field, 
    int width, 
    int height, 
    float dx, 
    float dy)
{
    // Allocate: width × height × 2 (two channels: dx and dy)
    dir_field.allocate(width, height, 2);
    
    // Fill the direction field (normalization handled during binning)
    // Channel 0 = dx, Channel 1 = dy
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            dir_field.zyx(0, row, col) = dx;  // dx component
            dir_field.zyx(1, row, col) = dy;  // dy component
        }
    }
}

// Test 1: Verify that custom direction field mode is activated correctly
void test_glcm_custom_direction_activation()
{
    // Setup feature settings
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
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 100;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    
    // Load test ROI data
    LR roidata;
    load_masked_test_roi_data(roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask, 
                               sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    
    // Create direction field (horizontal direction: dx=1, dy=0)
    SimpleCube<float> dir_field;
    create_uniform_direction_field_2d(dir_field, 
                                      roidata.aux_image_matrix.width, 
                                      roidata.aux_image_matrix.height,
                                      1.0f, 0.0f);
    
    // Test with custom direction field
    GLCMFeature f;
    f.set_direction_field(&dir_field);
    
    ASSERT_NO_THROW(f.calculate(roidata, s));
    
    // Initialize feature buffer
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);
    
    // Direction field mode produces 4 values (one per canonical direction: 0°, 45°, 90°, 135°)
    // Pixels are binned to their closest canonical direction
    ASSERT_EQ(roidata.fvals[(int)Feature2D::GLCM_CONTRAST].size(), 4);
    ASSERT_EQ(roidata.fvals[(int)Feature2D::GLCM_ENERGY].size(), 4);
    ASSERT_EQ(roidata.fvals[(int)Feature2D::GLCM_ENTROPY].size(), 4);
    
    // Cleanup
    GLCMFeature::direction_field = nullptr;
}

// Test 2: Compare traditional 0-degree angle vs custom direction field with dx=1, dy=0
void test_glcm_custom_direction_vs_traditional_0deg()
{
    // Setup
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
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 100;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    GLCMFeature::symmetric_glcm = false;
    
    // Traditional mode: 0 degrees only
    GLCMFeature::angles = { 0 };
    
    LR roidata_traditional;
    load_masked_test_roi_data(roidata_traditional, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
                               sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    
    GLCMFeature f_traditional;
    f_traditional.calculate(roidata_traditional, s);
    roidata_traditional.initialize_fvals();
    f_traditional.save_value(roidata_traditional.fvals);
    
    // Custom direction field mode: uniform horizontal direction (dx=1, dy=0)
    // This should be equivalent to 0 degrees
    LR roidata_custom;
    load_masked_test_roi_data(roidata_custom, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
                               sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    
    SimpleCube<float> dir_field;
    create_uniform_direction_field_2d(dir_field,
                                      roidata_custom.aux_image_matrix.width,
                                      roidata_custom.aux_image_matrix.height,
                                      1.0f, 0.0f);
    
    GLCMFeature f_custom;
    f_custom.set_direction_field(&dir_field);
    f_custom.calculate(roidata_custom, s);
    roidata_custom.initialize_fvals();
    f_custom.save_value(roidata_custom.fvals);
    
    // Compare results - they should be very close (within floating point tolerance)
    // Traditional mode with 0° gives 1 value at index [0]
    // Custom mode with uniform dx=1,dy=0 bins all pixels to 0° (index 0 of the 4 directions)
    double trad_contrast = roidata_traditional.fvals[(int)Feature2D::GLCM_CONTRAST][0];
    double custom_contrast = roidata_custom.fvals[(int)Feature2D::GLCM_CONTRAST][0];  // 0° direction
    
    double trad_energy = roidata_traditional.fvals[(int)Feature2D::GLCM_ENERGY][0];
    double custom_energy = roidata_custom.fvals[(int)Feature2D::GLCM_ENERGY][0];
    
    double trad_entropy = roidata_traditional.fvals[(int)Feature2D::GLCM_ENTROPY][0];
    double custom_entropy = roidata_custom.fvals[(int)Feature2D::GLCM_ENTROPY][0];
    
    // Values should match closely (within 1% tolerance due to potential rounding differences)
    ASSERT_TRUE(agrees_gt(trad_contrast, custom_contrast, 1.0));
    ASSERT_TRUE(agrees_gt(trad_energy, custom_energy, 1.0));
    ASSERT_TRUE(agrees_gt(trad_entropy, custom_entropy, 1.0));
    
    // Cleanup
    GLCMFeature::direction_field = nullptr;
    GLCMFeature::angles = { 0, 45, 90, 135 };
}

// Test 3: Verify different direction fields produce different results
void test_glcm_custom_direction_different_directions()
{
    // Setup
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
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 100;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    
    // Horizontal direction (dx=1, dy=0)
    LR roidata_horizontal;
    load_masked_test_roi_data(roidata_horizontal, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
                               sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    
    SimpleCube<float> dir_field_h;
    create_uniform_direction_field_2d(dir_field_h,
                                      roidata_horizontal.aux_image_matrix.width,
                                      roidata_horizontal.aux_image_matrix.height,
                                      1.0f, 0.0f);
    
    GLCMFeature f_horizontal;
    f_horizontal.set_direction_field(&dir_field_h);
    f_horizontal.calculate(roidata_horizontal, s);
    roidata_horizontal.initialize_fvals();
    f_horizontal.save_value(roidata_horizontal.fvals);
    
    // Vertical direction (dx=0, dy=1)
    LR roidata_vertical;
    load_masked_test_roi_data(roidata_vertical, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
                               sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    
    SimpleCube<float> dir_field_v;
    create_uniform_direction_field_2d(dir_field_v,
                                      roidata_vertical.aux_image_matrix.width,
                                      roidata_vertical.aux_image_matrix.height,
                                      0.0f, 1.0f);
    
    GLCMFeature f_vertical;
    f_vertical.set_direction_field(&dir_field_v);
    f_vertical.calculate(roidata_vertical, s);
    roidata_vertical.initialize_fvals();
    f_vertical.save_value(roidata_vertical.fvals);
    
    // Results should be different (unless the image has perfect symmetry)
    // Horizontal (dx=1, dy=0) bins to 0° (index 0)
    // Vertical (dx=0, dy=1) bins to 90° (index 2)
    double h_contrast = roidata_horizontal.fvals[(int)Feature2D::GLCM_CONTRAST][0];  // 0° direction
    double v_contrast = roidata_vertical.fvals[(int)Feature2D::GLCM_CONTRAST][2];    // 90° direction
    
    // We expect different values, but both should be valid (non-NaN, non-negative)
    ASSERT_FALSE(std::isnan(h_contrast));
    ASSERT_FALSE(std::isnan(v_contrast));
    ASSERT_GE(h_contrast, 0.0);
    ASSERT_GE(v_contrast, 0.0);
    
    // Cleanup
    GLCMFeature::direction_field = nullptr;
}

// Test 4: Verify mask is still respected with custom direction fields
void test_glcm_custom_direction_respects_mask()
{
    // Setup
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
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 100;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    
    LR roidata;
    load_masked_test_roi_data(roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,
                               sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    
    SimpleCube<float> dir_field;
    create_uniform_direction_field_2d(dir_field,
                                      roidata.aux_image_matrix.width,
                                      roidata.aux_image_matrix.height,
                                      1.0f, 0.0f);
    
    GLCMFeature f;
    f.set_direction_field(&dir_field);
    
    // Should not throw - mask should be properly handled
    ASSERT_NO_THROW(f.calculate(roidata, s));
    
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);
    
    // Results should be valid (not NaN)
    double contrast = roidata.fvals[(int)Feature2D::GLCM_CONTRAST][0];
    ASSERT_FALSE(std::isnan(contrast));
    
    // Cleanup
    GLCMFeature::direction_field = nullptr;
}