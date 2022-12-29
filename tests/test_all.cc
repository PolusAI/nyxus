#include <gtest/gtest.h>
#include "test_gabor.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "test_pixel_intensity_features.h"
#include "test_morphology_features.h"
#include "test_ibsi.h"
#include "test_initialization.h"
#include "test_ibsi_glcm.h"

TEST(TEST_NYXUS, TEST_GABOR){
    test_gabor();
	
	#ifdef USE_GPU
		test_gabor(true);
	#endif
	
}

TEST(TEST_NYXUS, TEST_INITIALIZATION) {
	test_initialization();
}

//
//==== Pixel intensity features
//

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_INTEGRATED_INTENSITY) 
{
	ASSERT_NO_THROW(test_pixel_intensity_integrated_intensity());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_MIN_MAX_RANGE) 
{
	ASSERT_NO_THROW(test_pixel_intensity_min_max_range());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_MEAN) 
{
	ASSERT_NO_THROW(test_pixel_intensity_mean());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_MEDIAN) 
{
	ASSERT_NO_THROW(test_pixel_intensity_median());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_MODE) 
{
	ASSERT_NO_THROW(test_pixel_intensity_mode());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_STDDEV) 
{
	ASSERT_NO_THROW(test_pixel_intensity_standard_deviation());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_SKEWNESS) 
{
	ASSERT_NO_THROW(test_pixel_intensity_skewness());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_KURTOSIS) 
{
	ASSERT_NO_THROW(test_pixel_intensity_kurtosis());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_HYPERSKEWNESS) 
{
	ASSERT_NO_THROW(test_pixel_intensity_hyperskewness());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_HYPERFLATNESS) 
{
	ASSERT_NO_THROW(test_pixel_intensity_hyperflatness());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_MAD) 
{
	ASSERT_NO_THROW(test_pixel_intensity_mean_absolute_deviation());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_STANDARD_ERROR) 
{
	ASSERT_NO_THROW(test_pixel_intensity_standard_error());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_RMS) 
{
	ASSERT_NO_THROW(test_pixel_intensity_root_mean_squared());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_ENTROPY) 
{
	ASSERT_NO_THROW(test_pixel_intensity_entropy());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_ENERGY) 
{
	ASSERT_NO_THROW(test_pixel_intensity_energy());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_UNIFORMITY) 
{
	ASSERT_NO_THROW(test_pixel_intensity_uniformity());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_UNIFORMITY_PIU) 
{
	ASSERT_NO_THROW(test_pixel_intensity_uniformity_piu());
}

//
//==== Morphology features
//

TEST(TEST_NYXUS, TEST_MORPHOLOGY_PERIMETER) 
{
	ASSERT_NO_THROW(test_morphology_perimeter());
}

//
//==== IBSI tests
//

TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIFFERENCE_AVERAGE) 
{
	ASSERT_NO_THROW(test_ibsi_glcm_difference_average());
}


TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIFFERENCE_VARIANCE)
{
	ASSERT_NO_THROW(test_ibsi_glcm_difference_variance());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIFFERENCE_ENTROPY)
{
	ASSERT_NO_THROW(test_ibsi_glcm_difference_entropy());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_SUM_AVERAGE) 
{
	ASSERT_NO_THROW(test_ibsi_glcm_sum_average());
}


TEST(TEST_NYXUS, TEST_IBSI_GLCM_SUM_VARIANCE)
{
	ASSERT_NO_THROW(test_ibsi_glcm_sum_variance());
}


TEST(TEST_NYXUS, TEST_IBSI_GLCM_SUM_ENTROPY)
{
	ASSERT_NO_THROW(test_ibsi_glcm_sum_entropy());
}


TEST(TEST_NYXUS, TEST_IBSI_GLCM_ANGULAR_2D_MOMENT)
{
	ASSERT_NO_THROW(test_ibsi_glcm_angular_2d_moment());
}


TEST(TEST_NYXUS, TEST_IBSI_CONTRAST)
{
	ASSERT_NO_THROW(test_ibsi_glcm_contrast());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_INVERSED_DIFFERENCE_MOMENT)
{
	ASSERT_NO_THROW(test_ibsi_glcm_inversed_difference_moment());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_CORRELATION)
{
	ASSERT_NO_THROW(test_ibsi_glcm_correlation());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_INFOMEAS1)
{
	ASSERT_NO_THROW(test_ibsi_glcm_infomeas1());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_INFOMEAS2)
{
	ASSERT_NO_THROW(test_ibsi_glcm_infomeas2());
}




int main(int argc, char **argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}