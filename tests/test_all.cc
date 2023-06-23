#include <gtest/gtest.h>
#include "test_gabor.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "test_pixel_intensity_features.h"
#include "test_morphology_features.h"
#include "test_initialization.h"
#include "test_ibsi_glcm.h"
#include "test_ibsi_gldm.h"
#include "test_ibsi_glrlm.h"
#include "test_ibsi_glszm.h"
#include "test_ibsi_intensity.h"
#include "test_ibsi_ngldm.h"
#include "test_ibsi_ngtdm.h"
#include "test_glcm.h"
#include "test_gldm.h"
#include "test_glrlm.h"
#include "test_glszm.h"
#include "test_ngtdm.h"
#include "test_roi_blacklist.h"

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


// GLCM tests

TEST(TEST_NYXUS, TEST_IBSI_GLCM_ACOR)
{
	ASSERT_NO_THROW(test_ibsi_glcm_ACOR());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_ANGULAR_2D_MOMENT)
{
	ASSERT_NO_THROW(test_ibsi_glcm_angular_2d_moment());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_CLUPROM)
{
	ASSERT_NO_THROW(test_ibsi_glcm_CLUPROM());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_CLUSHADE)
{
	ASSERT_NO_THROW(test_ibsi_glcm_CLUSHADE());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_CLUTEND)
{
	ASSERT_NO_THROW(test_ibsi_glcm_CLUTEND());
}

TEST(TEST_NYXUS, TEST_IBSI_CONTRAST)
{
	ASSERT_NO_THROW(test_ibsi_glcm_contrast());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_CORRELATION)
{
	ASSERT_NO_THROW(test_ibsi_glcm_correlation());
}

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

TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIS)
{
	ASSERT_NO_THROW(test_ibsi_glcm_DIS());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIFFERENCE_ID)
{
	ASSERT_NO_THROW(test_ibsi_glcm_ID());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIFFERENCE_IDN)
{
	ASSERT_NO_THROW(test_ibsi_glcm_IDN());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIFFERENCE_IDM)
{
	ASSERT_NO_THROW(test_ibsi_glcm_IDM());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_DIFFERENCE_IDMN)
{
	ASSERT_NO_THROW(test_ibsi_glcm_IDMN());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_INFOMEAS1)
{
	ASSERT_NO_THROW(test_ibsi_glcm_infomeas1());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_INFOMEAS2)
{
	ASSERT_NO_THROW(test_ibsi_glcm_infomeas2());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_INVERSED_DIFFERENCE_MOMENT)
{
	ASSERT_NO_THROW(test_ibsi_glcm_inversed_difference_moment());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_IV)
{
	ASSERT_NO_THROW(test_ibsi_glcm_IV());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_JAVE)
{
	ASSERT_NO_THROW(test_ibsi_glcm_JAVE());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_JE)
{
	ASSERT_NO_THROW(test_ibsi_glcm_JE());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_JMAX)
{
	ASSERT_NO_THROW(test_ibsi_glcm_JMAX());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_JVAR)
{
	ASSERT_NO_THROW(test_ibsi_glcm_JVAR());
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

// GLDM tests

TEST(TEST_NYXUS, TEST_IBSI_GLDM_SDE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_sde());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_LDE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_lde());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_LGLE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_lgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_HGLE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_hgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_SDLGLE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_sdlgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_SDHGLE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_sdhgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_LDLGLE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_ldlgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_LDHGLE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_ldhgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_GLN) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_gln());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_DN) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_dn());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_DNN) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_dnn());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_GLV) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_glv());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_DV) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_dv());
}

TEST(TEST_NYXUS, TEST_IBSI_GLDM_DE) 
{
	ASSERT_NO_THROW(test_ibsi_gldm_de());
}

// GLRLM

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_SRE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_sre());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_LRE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_lre());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_LGLRE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_lglre());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_HGLRE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_hglre());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_SRLGLE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_srlgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_SRHGLE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_srhgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_LRLGLE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_lrlgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_LRHGLE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_lrhgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_GLN)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_gln());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_GLNN)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_glnn());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_RLN)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_rln());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_RLNN)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_rlnn());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_RP)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_rp());
}


TEST(TEST_NYXUS, TEST_IBSI_GLRLM_GLV)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_glv());
}


TEST(TEST_NYXUS, TEST_IBSI_GLRLM_RV)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_rv());
}

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_RE)
{
	ASSERT_NO_THROW(test_ibsi_glrlm_re());
}

// GLSZM

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_SAE) {
	ASSERT_NO_THROW(test_ibsi_glszm_sae());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_LAE) {
	ASSERT_NO_THROW(test_ibsi_glszm_lae());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_LGLZE) {
	ASSERT_NO_THROW(test_ibsi_glszm_lglze());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_HGLZE) {
	ASSERT_NO_THROW(test_ibsi_glszm_hglze());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_SALGLZE) {
	ASSERT_NO_THROW(test_ibsi_glszm_salgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_SAHGLZE) {
	ASSERT_NO_THROW(test_ibsi_glszm_sahgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_LALGLE) {
	ASSERT_NO_THROW(test_ibsi_glszm_lalgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_LAHGLE) {
	ASSERT_NO_THROW(test_ibsi_glszm_lahgle());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_gln) {
	ASSERT_NO_THROW(test_ibsi_glszm_gln());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_glnn) {
	ASSERT_NO_THROW(test_ibsi_glszm_glnn());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_szn) {
	ASSERT_NO_THROW(test_ibsi_glszm_szn());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_SZNN) {
	ASSERT_NO_THROW(test_ibsi_glszm_sznn());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_ZP) {
	ASSERT_NO_THROW(test_ibsi_glszm_zp());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_GLV) {
	ASSERT_NO_THROW(test_ibsi_glszm_glv());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_ZV) {
	ASSERT_NO_THROW(test_ibsi_glszm_zv());
}

TEST(TEST_NYXUS, TEST_IBSI_GLSZM_ZE) {
	ASSERT_NO_THROW(test_ibsi_glszm_ze());
}

//------ NGLDM

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_MATRIX_CORRECTNESS1)
{
	ASSERT_NO_THROW (test_ibsi_NGLDM_matrix_correctness1());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_MATRIX_CORRECTNESS2)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_matrix_correctness2());
}

// INTENSITY

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_MEAN) 
{
	ASSERT_NO_THROW(test_ibsi_mean_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_SKEWNESS) 
{
	ASSERT_NO_THROW(test_ibsi_skewness_intensity());
}


TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_KURTOSIS) 
{
	ASSERT_NO_THROW(test_ibsi_kurtosis_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_MEDIAN) 
{
	ASSERT_NO_THROW(test_ibsi_median_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_MINIMUM) 
{
	ASSERT_NO_THROW(test_ibsi_minimum_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_P10) 
{
	ASSERT_NO_THROW(test_ibsi_p10_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_P90) 
{
	ASSERT_NO_THROW(test_ibsi_p90_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_INTERQUARTILE) 
{
	ASSERT_NO_THROW(test_ibsi_interquartile_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_RANGE) 
{
	ASSERT_NO_THROW(test_ibsi_range_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_MEAN_ABSOLUTE_DEVIATION) 
{
	ASSERT_NO_THROW(test_ibsi_mean_absolute_deviation_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_ENERGY) 
{
	ASSERT_NO_THROW(test_ibsi_energy_intensity());
}

TEST(TEST_NYXUS, TEST_IBSI_INTENSITY_ROOT_MEAN_SQUARED) 
{
	ASSERT_NO_THROW(test_ibsi_root_mean_squared_intensity());
}

// NGTDM

TEST(TEST_NYXUS, TEST_IBSI_NGTDM_COARSENESS) 
{
	ASSERT_NO_THROW(test_ibsi_ngtdm_coarseness());
}

TEST(TEST_NYXUS, TEST_IBSI_NGTDM_CONTRAST) 
{
	ASSERT_NO_THROW(test_ibsi_ngtdm_contrast());
}

TEST(TEST_NYXUS, TEST_IBSI_NGTDM_BUSYNESS) 
{
	ASSERT_NO_THROW(test_ibsi_ngtdm_busyness());
}

TEST(TEST_NYXUS, TEST_IBSI_NGTDM_COMPLEXITY) 
{
	ASSERT_NO_THROW(test_ibsi_ngtdm_complexity());
}

TEST(TEST_NYXUS, TEST_IBSI_NGTDM_STRENGTH) 
{
	ASSERT_NO_THROW(test_ibsi_ngtdm_strength());
}

//==== Tests of texture feature extraction witout binning

// GLCM tests

TEST(TEST_NYXUS, TEST_GLCM_ACOR)
{
	ASSERT_NO_THROW(test_glcm_ACOR());
}

TEST(TEST_NYXUS, TEST_GLCM_ANGULAR_2D_MOMENT)
{
	ASSERT_NO_THROW(test_glcm_angular_2d_moment());
}

TEST(TEST_NYXUS, TEST_GLCM_CLUPROM)
{
	ASSERT_NO_THROW(test_glcm_CLUPROM());
}

TEST(TEST_NYXUS, TEST_GLCM_CLUSHADE)
{
	ASSERT_NO_THROW(test_glcm_CLUSHADE());
}

TEST(TEST_NYXUS, TEST_GLCM_CLUTEND)
{
	ASSERT_NO_THROW(test_glcm_CLUTEND());
}

TEST(TEST_NYXUS, TEST_CONTRAST)
{
	ASSERT_NO_THROW(test_glcm_contrast());
}

TEST(TEST_NYXUS, TEST_GLCM_CORRELATION)
{
	ASSERT_NO_THROW(test_glcm_correlation());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFFERENCE_AVERAGE)
{
	ASSERT_NO_THROW(test_glcm_difference_average());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFFERENCE_VARIANCE)
{
	ASSERT_NO_THROW(test_glcm_difference_variance());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFFERENCE_ENTROPY)
{
	ASSERT_NO_THROW(test_glcm_difference_entropy());
}

TEST(TEST_NYXUS, TEST_GLCM_DIS)
{
	ASSERT_NO_THROW(test_glcm_DIS());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFFERENCE_ID)
{
	ASSERT_NO_THROW(test_glcm_ID());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFFERENCE_IDN)
{
	ASSERT_NO_THROW(test_glcm_IDN());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFFERENCE_IDM)
{
	ASSERT_NO_THROW(test_glcm_IDM());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFFERENCE_IDMN)
{
	ASSERT_NO_THROW(test_glcm_IDMN());
}

TEST(TEST_NYXUS, TEST_GLCM_INFOMEAS1)
{
	ASSERT_NO_THROW(test_glcm_infomeas1());
}

TEST(TEST_NYXUS, TEST_GLCM_INFOMEAS2)
{
	ASSERT_NO_THROW(test_glcm_infomeas2());
}

TEST(TEST_NYXUS, TEST_GLCM_IV)
{
	ASSERT_NO_THROW(test_glcm_IV());
}

TEST(TEST_NYXUS, TEST_GLCM_JAVE)
{
	ASSERT_NO_THROW(test_glcm_JAVE());
}

TEST(TEST_NYXUS, TEST_GLCM_JE)
{
	ASSERT_NO_THROW(test_glcm_JE());
}

TEST(TEST_NYXUS, TEST_GLCM_JMAX)
{
	ASSERT_NO_THROW(test_glcm_JMAX());
}

TEST(TEST_NYXUS, TEST_GLCM_JVAR)
{
	ASSERT_NO_THROW(test_glcm_JVAR());
}

TEST(TEST_NYXUS, TEST_GLCM_SUM_AVERAGE)
{
	ASSERT_NO_THROW(test_glcm_sum_average());
}

TEST(TEST_NYXUS, TEST_GLCM_SUM_VARIANCE)
{
	ASSERT_NO_THROW(test_glcm_sum_variance());
}

TEST(TEST_NYXUS, TEST_GLCM_SUM_ENTROPY)
{
	ASSERT_NO_THROW(test_glcm_sum_entropy());
}


// GLDM tests

TEST(TEST_NYXUS, TEST_GLDM_SDE) 
{
	ASSERT_NO_THROW(test_gldm_sde());
}

TEST(TEST_NYXUS, TEST_GLDM_LDE) 
{
	ASSERT_NO_THROW(test_gldm_lde());
}

TEST(TEST_NYXUS, TEST_GLDM_LGLE) 
{
	ASSERT_NO_THROW(test_gldm_lgle());
}

TEST(TEST_NYXUS, TEST_GLDM_HGLE) 
{
	ASSERT_NO_THROW(test_gldm_hgle());
}

TEST(TEST_NYXUS, TEST_GLDM_SDLGLE) 
{
	ASSERT_NO_THROW(test_gldm_sdlgle());
}

TEST(TEST_NYXUS, TEST_GLDM_SDHGLE) 
{
	ASSERT_NO_THROW(test_gldm_sdhgle());
}

TEST(TEST_NYXUS, TEST_GLDM_LDLGLE) 
{
	ASSERT_NO_THROW(test_gldm_ldlgle());
}

TEST(TEST_NYXUS, TEST_GLDM_LDHGLE) 
{
	ASSERT_NO_THROW(test_gldm_ldhgle());
}

TEST(TEST_NYXUS, TEST_GLDM_GLN) 
{
	ASSERT_NO_THROW(test_gldm_gln());
}

TEST(TEST_NYXUS, TEST_GLDM_DN) 
{
	ASSERT_NO_THROW(test_gldm_dn());
}

TEST(TEST_NYXUS, TEST_GLDM_DNN) 
{
	ASSERT_NO_THROW(test_gldm_dnn());
}

TEST(TEST_NYXUS, TEST_GLDM_GLV) 
{
	ASSERT_NO_THROW(test_gldm_glv());
}

TEST(TEST_NYXUS, TEST_GLDM_DV) 
{
	ASSERT_NO_THROW(test_gldm_dv());
}

TEST(TEST_NYXUS, TEST_GLDM_DE) 
{
	ASSERT_NO_THROW(test_gldm_de());
}

// GLRLM

TEST(TEST_NYXUS, TEST_GLRLM_SRE)
{
	ASSERT_NO_THROW(test_glrlm_sre());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRE)
{
	ASSERT_NO_THROW(test_glrlm_lre());
}

TEST(TEST_NYXUS, TEST_GLRLM_LGLRE)
{
	ASSERT_NO_THROW(test_glrlm_lglre());
}

TEST(TEST_NYXUS, TEST_GLRLM_HGLRE)
{
	ASSERT_NO_THROW(test_glrlm_hglre());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRLGLE)
{
	ASSERT_NO_THROW(test_glrlm_srlgle());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRHGLE)
{
	ASSERT_NO_THROW(test_glrlm_srhgle());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRLGLE)
{
	ASSERT_NO_THROW(test_glrlm_lrlgle());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRHGLE)
{
	ASSERT_NO_THROW(test_glrlm_lrhgle());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLN)
{
	ASSERT_NO_THROW(test_glrlm_gln());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLNN)
{
	ASSERT_NO_THROW(test_glrlm_glnn());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLN)
{
	ASSERT_NO_THROW(test_glrlm_rln());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLNN)
{
	ASSERT_NO_THROW(test_glrlm_rlnn());
}

TEST(TEST_NYXUS, TEST_GLRLM_RP)
{
	ASSERT_NO_THROW(test_glrlm_rp());
}


TEST(TEST_NYXUS, TEST_GLRLM_GLV)
{
	ASSERT_NO_THROW(test_glrlm_glv());
}


TEST(TEST_NYXUS, TEST_GLRLM_RV)
{
	ASSERT_NO_THROW(test_glrlm_rv());
}

TEST(TEST_NYXUS, TEST_GLRLM_RE)
{
	ASSERT_NO_THROW(test_glrlm_re());
}

// GLSZM

TEST(TEST_NYXUS, TEST_GLSZM_SAE) {
	ASSERT_NO_THROW(test_glszm_sae());
}

TEST(TEST_NYXUS, TEST_GLSZM_LAE) {
	ASSERT_NO_THROW(test_glszm_lae());
}

TEST(TEST_NYXUS, TEST_GLSZM_LGLZE) {
	ASSERT_NO_THROW(test_glszm_lglze());
}

TEST(TEST_NYXUS, TEST_GLSZM_HGLZE) {
	ASSERT_NO_THROW(test_glszm_hglze());
}

TEST(TEST_NYXUS, TEST_GLSZM_SALGLZE) {
	ASSERT_NO_THROW(test_glszm_salgle());
}

TEST(TEST_NYXUS, TEST_GLSZM_SAHGLZE) {
	ASSERT_NO_THROW(test_glszm_sahgle());
}

TEST(TEST_NYXUS, TEST_GLSZM_LALGLE) {
	ASSERT_NO_THROW(test_glszm_lalgle());
}

TEST(TEST_NYXUS, TEST_GLSZM_LAHGLE) {
	ASSERT_NO_THROW(test_glszm_lahgle());
}

TEST(TEST_NYXUS, TEST_GLSZM_gln) {
	ASSERT_NO_THROW(test_glszm_gln());
}

TEST(TEST_NYXUS, TEST_GLSZM_glnn) {
	ASSERT_NO_THROW(test_glszm_glnn());
}

TEST(TEST_NYXUS, TEST_GLSZM_szn) {
	ASSERT_NO_THROW(test_glszm_szn());
}

TEST(TEST_NYXUS, TEST_GLSZM_SZNN) {
	ASSERT_NO_THROW(test_glszm_sznn());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZP) {
	ASSERT_NO_THROW(test_glszm_zp());
}

TEST(TEST_NYXUS, TEST_GLSZM_GLV) {
	ASSERT_NO_THROW(test_glszm_glv());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZV) {
	ASSERT_NO_THROW(test_glszm_zv());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZE) {
	ASSERT_NO_THROW(test_glszm_ze());
}

// NGTDM

TEST(TEST_NYXUS, TEST_NGTDM_COARSENESS) 
{
	ASSERT_NO_THROW(test_ngtdm_coarseness());
}

TEST(TEST_NYXUS, TEST_NGTDM_CONTRAST) 
{
	ASSERT_NO_THROW(test_ngtdm_contrast());
}

TEST(TEST_NYXUS, TEST_NGTDM_BUSYNESS) 
{
	ASSERT_NO_THROW(test_ngtdm_busyness());
}

TEST(TEST_NYXUS, TEST_NGTDM_COMPLEXITY) 
{
	ASSERT_NO_THROW(test_ngtdm_complexity());
}

TEST(TEST_NYXUS, TEST_NGTDM_STRENGTH) 
{
	ASSERT_NO_THROW(test_ngtdm_strength());
}

// ROI blacklisting

TEST(TEST_NYXUS, TEST_ROI_BLACKLISTING) 
{
	ASSERT_NO_THROW(test_roi_blacklist());
}

int main(int argc, char **argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}