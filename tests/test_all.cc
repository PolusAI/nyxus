#define NOMINMAX	// keep Windows min/max macros from breaking dcmtk's OFvariant (DICOM tests)
#include <gtest/gtest.h>
#include "test_2d_gabor_skimage.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "test_2d_contour_analytic.h"
#include "test_2d_firstorder_common.h"
#include "test_2d_firstorder_regression.h"
#include "test_2d_firstorder_matlab.h"
#include "test_2d_intensity_histogram_regression.h"
#include "test_2d_radial_regression.h"
#include "test_2d_intensity_histogram_mechanics.h"
#include "test_2d_intensity_histogram_ibsi.h"
#include "test_2d_intensity_histogram_mirp.h"
#include "test_2d_intensity_histogram_analytic.h"
#include "test_hu_analytic.h"
#include "test_2d_hu_mechanics.h"
#include "test_2d_morphology_regression.h"
#include "test_2d_morphology_analytic.h"
#include "test_2d_morphology_imea.h"
#include "test_2d_morphology_skimage.h"
#include "test_2d_morphology_matlab.h"
#include "test_2d_morphology_cellprofiler.h"
#include "test_2d_morphology_fraclac.h"
#include "test_2d_moments_skimage.h"
#include "test_2d_moments_regression.h"
#include "test_2d_zernike_regression.h"
#include "test_2d_neighbor_common.h"
#include "test_2d_neighbor_regression.h"
#include "test_2d_neighbor_cellprofiler.h"
#include "test_2d_neighbor_analytic.h"
#include "test_initialization_mechanics.h"
#include "test_2d_glcm_ibsi.h"
#include "test_2d_glcm_pyradiomics.h"
#include "test_2d_glcm_mirp.h"
#include "test_2d_gldm_ibsi.h"
#include "test_2d_glrlm_ibsi.h"
#include "test_2d_glrlm_pyradiomics.h"
#include "test_2d_glrlm_mirp.h"
#include "test_2d_gldzm_ibsi.h"
#include "test_2d_glszm_ibsi.h"
#include "test_2d_firstorder_ibsi.h"
#include "test_2d_firstorder_pyradiomics.h"
#include "test_2d_ngldm_ibsi.h"
#include "test_2d_ngldm_mirp.h"
#include "test_2d_ngldm_regression.h"
#include "test_2d_ngtdm_ibsi.h"
#include "test_2d_glcm_regression.h"
#include "test_2d_gldm_regression.h"
#include "test_2d_gldm_mechanics.h"
#include "test_2d_glrlm_regression.h"
#include "test_2d_glszm_regression.h"
#include "test_2d_ngtdm_regression.h"
#include "test_roi_blacklist_mechanics.h"
#include "test_2d_tiff_loader_mechanics.h"
#include "test_imq_regression.h"
#include "test_imq_opencv.h"
#include "test_imq_cellprofiler.h"
#include "test_3d_nifti_mechanics.h"
#include "test_2d_omezarr_mechanics.h"
#include "test_3d_morphology_common.h"
#include "test_3d_morphology_regression.h"
#include "test_3d_morphology_matlab.h"
#include "test_3d_gldzm_regression.h"
#include "test_3d_ngldm_regression.h"
#include "test_3d_firstorder_pyradiomics.h"
#include "test_3d_glcm_pyradiomics.h"
#include "test_3d_glcm_regression.h"
#include "test_3d_gldm_pyradiomics.h"
#include "test_3d_ngtdm_pyradiomics.h"
#include "test_3d_glrlm_pyradiomics.h"
#include "test_3d_glszm_pyradiomics.h"
#include "test_3d_coverage_common.h"
#include "test_3d_firstorder_coverage.h"
#include "test_3d_morphology_coverage.h"
#include "test_3d_gldm_coverage.h"
#include "test_3d_gldzm_coverage.h"
#include "test_3d_glrlm_coverage.h"
#include "test_3d_glszm_coverage.h"
#include "test_3d_ngldm_coverage.h"
#include "test_3d_ngtdm_coverage.h"
#include "test_2d_glcm_mechanics.h"
#ifdef USE_ARROW
    #include "test_arrow_mechanics.h"
    #include "test_arrow_file_name_mechanics.h"
#endif


//***** 2D contour and multicontour *****

TEST(TEST_NYXUS, TEST_2D_CONTOUR_MULTI_DISCONNECTED_ANALYTIC) {
	ASSERT_NO_THROW(test_2d_contour_multi_disconnected_analytic());
}

TEST(TEST_NYXUS, TEST_2D_CONTOUR_SINGLE_ANALYTIC) {
	ASSERT_NO_THROW(test_2d_contour_single_analytic());
}

TEST(TEST_NYXUS, TEST_2D_CONTOUR_SINGLE_TAILED_ANALYTIC) {
	ASSERT_NO_THROW(test_2d_contour_single_tailed_analytic());
}

TEST(TEST_NYXUS, TEST_2D_CONTOUR_VOID_ANALYTIC) {
	ASSERT_NO_THROW(test_2d_contour_void_analytic());
}

TEST(TEST_NYXUS, TEST_2D_CONTOUR_MULTI_CONNECTED_ANALYTIC) {
	ASSERT_NO_THROW(test_2d_contour_multi_connected_analytic());
}


//***** first-order compatibility *****

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_P10_PYRADIOMICS) {
	ASSERT_NO_THROW(assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::P10, "3P10"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_P90_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::P90, "3P90"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ENERGY_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::ENERGY, "3ENERGY"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ENTROPY_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::ENTROPY, "3ENTROPY"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_INTERQUARTILE_RANGE_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::INTERQUARTILE_RANGE, "3INTERQUARTILE_RANGE"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_KURTOSIS_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::KURTOSIS, "3KURTOSIS"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MAX_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::MAX, "3MAX"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MEAN_ABSOLUTE_DEVIATION_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::MEAN_ABSOLUTE_DEVIATION, "3MEAN_ABSOLUTE_DEVIATION"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MEAN_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::MEAN, "3MEAN"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MEDIAN_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::MEDIAN, "3MEDIAN"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MIN_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::MIN, "3MIN"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_RANGE_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::RANGE, "3RANGE"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ROBUST_MEAN_ABSOLUTE_DEVIATION_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION, "3ROBUST_MEAN_ABSOLUTE_DEVIATION"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ROOT_MEAN_SQUARED_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::ROOT_MEAN_SQUARED, "3ROOT_MEAN_SQUARED"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_SKEWNESS_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::SKEWNESS, "3SKEWNESS"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_UNIFORMITY_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::UNIFORMITY, "3UNIFORMITY"));
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_VARIANCE_PYRADIOMICS) {
	ASSERT_NO_THROW (assert_3d_firstorder_feature_pyradiomics(Nyxus::Feature3D::VARIANCE, "3VARIANCE"));
}


//***** 3D NGTDM compatibility *****

TEST(TEST_NYXUS, TEST_3D_NGTDM_BUSYNESS_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_busyness_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_COARSENESS_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_coarseness_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_COMPLEXITY_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_complexity_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_CONTRAST_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_contrast_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_STRENGTH_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_strength_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_MATRIX_CORRECTNESS_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_matrix_correctness_pyradiomics());
}


//***** 3D GLRLM compatibility *****

TEST(TEST_NYXUS, TEST_3D_GLRLM_MATRIX_CORRECTNESS_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_matrix_correctness_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_GLN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_gln_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_GLNN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_glnn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_GLV_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_glv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_HGLRE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_hglre_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_LRE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_lre_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_LRHGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_lrhgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_LRLGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_lrlgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_LGLRE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_lglre_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_RE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_re_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_RLN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_rln_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_RLNN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_rlnn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_RP_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_rp_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_RV_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_rv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_SRE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_sre_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_SRHGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_srhgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_SRLGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_srlgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLRLM_AVE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glrlm_ave_pyradiomics());
}


//***** 3D GLSZM compatibility *****

TEST(TEST_NYXUS, TEST_3D_GLSZM_MATRIX_CORRECTNESS_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_glszm_matrix_correctness_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SAE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_sae_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LAE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_lae_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LGLZE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_lglze_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_HGLZE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_hglze_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SALGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_salgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SAHGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_sahgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LALGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_lalgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LAHGLE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_lahgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_GLN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_gln_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_GLNN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_glnn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SZN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_szn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SZNN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_sznn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_ZP_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_zp_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_GLV_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_glv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_ZV_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_zv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_ZE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glszm_ze_pyradiomics());
}


//***** 3D GLDM compatibility *****

TEST(TEST_NYXUS, TEST_3D_GLDM_DE_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_de_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_DN_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_dn_pyradiomics()); 
}

TEST(TEST_NYXUS, TEST_3D_GLDM_DNN_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_dnn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_DV_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_dv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_GLN_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_gln_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_GLV_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_glv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_HGLE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_hgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_LDE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_lde_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_LDHGLE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_ldhgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_LDLGLE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_ldlgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_LGLE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_lgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_SDE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_sde_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_SDHGLE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_sdhgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_SDLGLE_PYRADIOMICS) { 
	ASSERT_NO_THROW (test_3d_gldm_sdlgle_pyradiomics());
}

//***** 3D GLCM compatibility *****

TEST(TEST_NYXUS, TEST_3D_GLCM_EQUIVALENCE_DUMP_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_equivalence_dump_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_ACOR_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_acor_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_ASM_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_asm_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_CLUPROM_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_cluprom_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_CLUSHADE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_clushade_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_CLUTEND_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_clutend_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_CONTRAST_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_contrast_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_CORRELATION_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_correlation_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_DIFFERENCE_AVERAGE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_difference_average_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_DIFFERENCE_VARIANCE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_difference_variance_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_DIFFERENCE_ENTROPY_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_difference_entropy_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_ID_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_id_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_IDN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_idn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_IDM_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_idm_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_IDMN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_idmn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS1_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_infomeas1_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS2_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_infomeas2_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_IV_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_iv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_JAVE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_jave_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_JE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_je_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_JMAX_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_jmax_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_SUM_AVERAGE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_sum_average_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLCM_SUM_ENTROPY_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_glcm_sum_entropy_pyradiomics());
}


//***** Apache I/O tests *****

#ifdef USE_ARROW

TEST(TEST_NYXUS, TEST_ARROW_FILE_NAMING_MECHANICS) {
	test_arrow_file_naming_mechanics();
}

TEST(TEST_NYXUS, TEST_ARROW_IPC_MECHANICS) {
	test_arrow_ipc_mechanics();
}

TEST(TEST_NYXUS, TEST_ARROW_PARQUET_MECHANICS) {
	test_arrow_parquet_mechanics();
}

#endif


//***** 3D shape *****

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_MESH_VOLUME_MATLAB) {
	ASSERT_NO_THROW(test_3d_morphology_mesh_volume_matlab());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_AREA_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_area_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_AREA_2_VOLUME_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_area_2_volume_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_COMPACTNESS1_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_compactness1_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_COMPACTNESS2_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_compactness2_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_SPHERICAL_DISPROPORTION_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_spherical_disproportion_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_SPHERICITY_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_sphericity_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_VOLUME_CONVEX_HULL_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_volume_convex_hull_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_VOXEL_VOLUME_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_voxel_volume_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_COVMATRIX_AND_EIGENVALS_MATLAB) {
	ASSERT_NO_THROW(test_3d_morphology_covmatrix_and_eigenvals_matlab());
}


//***** 3D GLDZM regression *****

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLM_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_glm_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLV_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_glv_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_lde_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_sde_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_lglze_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_HGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_hglze_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDLGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_sdlgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_sdhgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDLGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_ldlgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_ldhgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLNU_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_glnu_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLNUN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_glnun_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDNU_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zdnu_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDNUN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zdnun_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDV_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zdv_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zp_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zde_regression());
}


//***** 3D NGLDM regression *****

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_lde_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_hde_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LGLCE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_lglce_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HGLCE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_hglce_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDLGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_ldlgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_ldhgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDLGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_hdlgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_hdhgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLNU_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_glnu_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLNUN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_glnun_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCNU_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcnu_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCNUN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcnun_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcp_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLM_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_glm_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLV_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_glv_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCM_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcm_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCV_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcv_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCENT_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcent_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCENE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcene_regression());
}


//***** Gabor (vetted vs scikit-image) *****

TEST(TEST_NYXUS, TEST_2D_GABOR_SKIMAGE){
    assert_2d_gabor_skimage();

    #ifdef USE_GPU
        assert_2d_gabor_skimage(true);
    #endif
}


//***** helper functionality ***** 

TEST(TEST_NYXUS, TEST_ROI_BLACKLIST_MECHANICS)
{
	ASSERT_NO_THROW(test_roi_blacklist_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_TIFF_LOADER_UINT32_STRIP_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_tiff_loader_uint32_strip_mechanics());
}

TEST(TEST_NYXUS, TEST_INITIALIZATION_MECHANICS) {
	test_initialization_mechanics();
}


//***** Pixel intensity features ***** 

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_firstorder_pyradiomics());
}

//***** IBSI Intensity Histogram (IH) family *****

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_INTEGER_DOMAIN_VALUES_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_integer_domain_values_regression());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_INDEX_AND_PERCENTILE_BOUNDS_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_index_and_percentile_bounds_regression());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_GATE_OFF_RETURNS_NAN_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_gate_off_returns_nan_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_FLOAT_DOMAIN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_float_domain_regression());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_FLOAT_DOMAIN_NEGATIVE_MIN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_float_domain_negative_min_regression());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_FLOAT_DOMAIN_PRESERVE_HU_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_float_domain_preserve_hu_regression());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_FLOAT_DOMAIN_PRESERVE_HU_FPACTIVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_float_domain_preserve_hu_fpactive_regression());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_REQUIRED_PREDICATE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_required_predicate_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_DISPERSION_IBSI)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_dispersion_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_DISPERSION_ROBUST_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_dispersion_robust_analytic());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_BIN_COUNTS_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_bin_counts_analytic());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_PHANTOM_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_phantom_analytic());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_FAMILY_MIRP)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_family_mirp());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_PHANTOM_PERCENTILE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_phantom_percentile_regression());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_HISTOGRAM_DISPERSION_PERCENTILE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_intensity_histogram_dispersion_percentile_regression());
}

TEST(TEST_NYXUS, TEST_HU_UINT_FRIENDLY_NORMALIZATION_CT_RANGE_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_uint_friendly_normalization_ct_range_analytic());
}

TEST(TEST_NYXUS, TEST_HU_UINT_FRIENDLY_RAWCAST_NONNEGATIVE_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_uint_friendly_rawcast_nonnegative_analytic());
}

TEST(TEST_NYXUS, TEST_HU_UINT_FRIENDLY_PRESERVE_OFFSET_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_uint_friendly_preserve_offset_analytic());
}

TEST(TEST_NYXUS, TEST_2D_HU_FPIMAGE_OPTIONS_PARSE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_fpimage_options_parse_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_INT16_PRESERVE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_int16_preserve_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_FLOAT_PRESERVE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_float_preserve_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_FLOAT_NONPRESERVE_BASELINE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_float_nonpreserve_baseline_mechanics());
}

#ifdef DICOM_SUPPORT
TEST(TEST_NYXUS, TEST_2D_HU_LOADER_DICOM_U16_PRESERVE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_dicom_u16_preserve_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_DICOM_I16_PRESERVE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_dicom_i16_preserve_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_DICOM_CT_SMALL_PRESERVE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_dicom_ct_small_preserve_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_DICOM_CT_SMALL_BASELINE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_dicom_ct_small_baseline_mechanics());
}
#endif

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ROBUST_MEAN_ABSOLUTE_DEVIATION_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_firstorder_robust_mean_absolute_deviation_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ENTROPY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_entropy_regression());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P01_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_p01_regression());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P25_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_p25_regression());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P75_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_p75_regression());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P99_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_p99_regression());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_QCOD_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_qcod_regression());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ROBUST_MEAN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_robust_mean_regression());
}

//***** Morphology features ***** 

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_PERIMETER_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_morphology_perimeter_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_BASIC_MATLAB)
{
	ASSERT_NO_THROW(test_2d_morphology_basic_matlab());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_BBOX_MATLAB)
{
	ASSERT_NO_THROW(test_2d_morphology_bbox_matlab());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ELLIPSE_MATLAB)
{
	ASSERT_NO_THROW(test_2d_morphology_ellipse_matlab());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_EULER_MATLAB)
{
	ASSERT_NO_THROW(test_2d_morphology_euler_matlab());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_DIAMETER_EQUAL_AREA_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_morphology_diameter_equal_area_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_BASIC_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_basic_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ELLIPSE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_ellipse_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CONTOUR_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_contour_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CONVEX_HULL_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_morphology_convex_hull_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ORIENTATION_AND_EROSIONS_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_morphology_orientation_and_erosions_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_EXTREMA_MATLAB)
{
	ASSERT_NO_THROW(test_2d_morphology_extrema_matlab());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_MISC_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_misc_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_FRACTAL_CIRCLE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_fractal_circle_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_FRACTAL_DIMENSION_BLOB512_FRACLAC)
{
	ASSERT_NO_THROW(test_2d_morphology_fractal_dimension_blob512_fraclac());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_RADIUS_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_radius_regression());
}

TEST(TEST_NYXUS, TEST_2D_MOMENTS_SHAPE_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_moments_shape_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MOMENTS_HU_WEDGE_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_moments_hu_wedge_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MOMENTS_SHAPE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_moments_shape_regression());
}

TEST(TEST_NYXUS, TEST_2D_MOMENTS_NORMRAW_SHAPE_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_moments_normraw_shape_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MOMENTS_NORMRAW_INTENSITY_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_moments_normraw_intensity_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MOMENTS_INTENSITY_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_moments_intensity_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MOMENTS_INTENSITY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_moments_intensity_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_EROSION_COMPLEMENT_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_erosion_complement_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CALIPER_SPREAD_IMEA)
{
	ASSERT_NO_THROW(test_2d_morphology_caliper_spread_imea());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CALIPER_SHAPE2D_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_caliper_shape2d_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CALIPER_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_caliper_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CALIPER_MARTIN_NASSENSTEIN_IMEA)
{
	ASSERT_NO_THROW(test_2d_morphology_caliper_martin_nassenstein_imea());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CALIPER_FERET_IMEA)
{
	ASSERT_NO_THROW(test_2d_morphology_caliper_feret_imea());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_MIN_ENCLOSING_CIRCLE_IMEA)
{
	ASSERT_NO_THROW(test_2d_morphology_min_enclosing_circle_imea());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_DOCUMENTED_FORMULA_CONFORMANCE_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_morphology_documented_formula_conformance_analytic());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_DIAMETER_EQUAL_PERIMETER_IMEA)
{
	ASSERT_NO_THROW(test_2d_morphology_diameter_equal_perimeter_imea());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_GEODETIC_LENGTH_THICKNESS_IMEA)
{
	ASSERT_NO_THROW(test_2d_morphology_geodetic_length_thickness_imea());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CHORD_STAT_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_chord_stat_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_CHORD_ANGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_chord_angle_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_POLYGONALITY_HEXAGONALITY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_morphology_polygonality_hexagonality_regression());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_EDGE_INTENSITY_CELLPROFILER)
{
	ASSERT_NO_THROW(test_2d_morphology_edge_intensity_cellprofiler());
}

TEST(TEST_NYXUS, TEST_2D_RADIAL_DISTRIBUTION_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_radial_distribution_regression());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_MOMENTS_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_zernike_moments_regression());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_COUNTS_AND_TOUCHING_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_neighbor_counts_and_touching_regression());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_PERCENT_TOUCHING_ENCLOSED_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_neighbor_percent_touching_enclosed_analytic());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_CLOSEST_NEIGHBORS_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_neighbor_closest_neighbors_regression());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_COUNTS_AND_FIRST_DISTANCE_CELLPROFILER)
{
	ASSERT_NO_THROW(test_2d_neighbor_counts_and_first_distance_cellprofiler());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_SECOND_DISTANCE_AND_ANGLES_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_neighbor_second_distance_and_angles_analytic());
}


//***** IBSI tests of NGTDM

TEST(TEST_NYXUS, TEST_2D_NGTDM_COARSENESS_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngtdm_coarseness_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_CONTRAST_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngtdm_contrast_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_BUSYNESS_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngtdm_busyness_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_COMPLEXITY_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngtdm_complexity_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_STRENGTH_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngtdm_strength_ibsi());
}


//***** IBSI tests of GLCM ***** 

TEST(TEST_NYXUS, TEST_2D_GLCM_ACOR_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_acor_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ASM_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_asm_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUPROM_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_cluprom_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUSHADE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_clushade_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUTEND_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_clutend_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CONTRAST_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_contrast_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CORRELATION_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_correlation_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_AVERAGE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_average_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_VARIANCE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_variance_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_ENTROPY_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_entropy_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIS_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_dis_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ID_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_id_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDN_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_idn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDM_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_idm_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDMN_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_idmn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS1_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas1_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS2_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas2_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IV_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_iv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JAVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_jave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_je_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_HOM2_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_hom2_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ENTROPY_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_entropy_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JMAX_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_jmax_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JVAR_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_jvar_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_AVERAGE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_glcm_sum_average_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_VARIANCE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_variance_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_ENTROPY_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_entropy_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ACOR_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_acor_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ASM_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_asm_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CONTRAST_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_contrast_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CORRELATION_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_correlation_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDMN_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_idmn_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDN_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_idn_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_AVERAGE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_average_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUPROM_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_cluprom_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUSHADE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_clushade_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUTEND_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_clutend_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_AVERAGE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_average_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_ENTROPY_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_entropy_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_VARIANCE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_variance_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIS_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_dis_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ENTROPY_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_entropy_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ID_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_id_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDM_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_idm_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS1_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas1_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS2_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas2_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IV_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_iv_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JAVE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_jave_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_je_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JMAX_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_jmax_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JVAR_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_jvar_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_ENTROPY_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_entropy_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_VARIANCE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_variance_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_VARIANCE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_variance_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_VARIANCE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glcm_variance_ave_ibsi());
}


//***** 2D GLCM vs the third-party tools *****

TEST(TEST_NYXUS, TEST_2D_GLCM_FAMILY_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_glcm_family_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_FAMILY_MIRP)
{
	ASSERT_NO_THROW(test_2d_glcm_family_mirp());
}


//***** 2D GLCM regression *****

TEST(TEST_NYXUS, TEST_2D_GLCM_ACOR_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_acor_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ASM_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_asm_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUPROM_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_cluprom_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUSHADE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_clushade_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUTEND_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_clutend_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CONTRAST_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_contrast_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CORRELATION_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_correlation_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_AVERAGE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_average_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_ENTROPY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_entropy_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_VARIANCE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_variance_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIS_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_dis_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ENERGY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_energy_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ENTROPY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_entropy_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_HOM1_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_hom1_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_HOM2_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_hom2_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ID_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_id_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_idn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDM_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_idm_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDMN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_idmn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS1_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas1_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS2_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas2_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IV_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_iv_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JAVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_jave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_je_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JMAX_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_jmax_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JVAR_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_jvar_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_AVERAGE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_average_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_ENTROPY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_entropy_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_VARIANCE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_variance_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_VARIANCE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_variance_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ASM_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_asm_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ACOR_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_acor_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUPROM_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_cluprom_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUSHADE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_clushade_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CLUTEND_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_clutend_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CONTRAST_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_contrast_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_CORRELATION_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_correlation_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_AVERAGE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_average_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_ENTROPY_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_entropy_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIFFERENCE_VARIANCE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_difference_variance_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_DIS_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_dis_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ENERGY_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_energy_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ENTROPY_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_entropy_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_HOM1_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_hom1_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_ID_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_id_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_idn_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDM_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_idm_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IDMN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_idmn_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_IV_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_iv_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JAVE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_jave_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_je_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS1_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas1_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_INFOMEAS2_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_infomeas2_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_VARIANCE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_variance_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JMAX_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_jmax_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_JVAR_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_jvar_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_AVERAGE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_average_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_ENTROPY_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_entropy_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLCM_SUM_VARIANCE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glcm_sum_variance_ave_regression());
}

// Regression guard: GLCM co-occurrence distance must default to 1 via the production
// settings path (exposes the offset=0 default defect that the hard-coded tests above miss).
TEST(TEST_NYXUS, TEST_2D_GLCM_BUG_OFFSET_DEFAULT_IS_ONE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_glcm_bug_offset_default_is_one_mechanics());
}

//***** IBSI tests of GLDM *****

TEST(TEST_NYXUS, TEST_2D_GLDM_SDE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_sde_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_lde_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LGLE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_lgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_HGLE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_hgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_SDLGLE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_sdlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_SDHGLE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_sdhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDLGLE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_ldlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDHGLE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_ldhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_GLN_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_gln_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DN_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_dn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DNN_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_dnn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_GLV_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DV_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_dv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_gldm_de_ibsi());
}


//***** IBSI tests of GLRLM ***** 

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_sre_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lre_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LGLRE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lglre_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_HGLRE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_hglre_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRLGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_srlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRHGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_srhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRLGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRHGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LGLRE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lglre_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_HGLRE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_hglre_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRLGLE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_srlgle_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRHGLE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_srhgle_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRLGLE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrlgle_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRHGLE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrhgle_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_sre_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_lre_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLN_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_gln_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLNN_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_glnn_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLN_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rln_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLNN_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rlnn_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RP_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rp_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLV_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_glv_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RV_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rv_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RE_AVE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_re_ave_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_FAMILY_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_glrlm_family_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_FAMILY_MIRP)
{
	ASSERT_NO_THROW(test_2d_glrlm_family_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLN_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_gln_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLNN_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_glnn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLN_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rln_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLNN_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rlnn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RP_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rp_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLV_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RV_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_rv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RE_IBSI)
{
	ASSERT_NO_THROW(test_2d_glrlm_re_ibsi());
}


//***** IBSI tests of GLSZM ***** 

TEST(TEST_NYXUS, TEST_2D_GLSZM_SAE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_sae_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LAE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_lae_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LGLZE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_lglze_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_HGLZE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_hglze_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SALGLE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_salgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SAHGLE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_sahgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LALGLE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_lalgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LAHGLE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_lahgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLN_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_gln_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLNN_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_glnn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SZN_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_szn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SZNN_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_sznn_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZP_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_zp_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLV_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZV_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_zv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZE_IBSI) {
	ASSERT_NO_THROW(test_2d_glszm_ze_ibsi());
}


//***** IBSI tests of NGLDM ***** 

TEST(TEST_NYXUS, TEST_2D_NGLDM_MATRIX_CORRECTNESS_IBSI)
{
	ASSERT_NO_THROW (test_2d_ngldm_matrix_correctness_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_LDE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_lde_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_HDE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_hde_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_LGLCE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_lglce_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_HGLCE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_hglce_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_LDLGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_ldlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_LDHGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_ldhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_HDLGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_hdlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_HDHGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_hdhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_GLNU_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_glnu_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_GLNUN_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_glnun_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_DCNU_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_dcnu_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_DCNUN_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_dcnun_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_DCP_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_dcp_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_MATRIX_CORRECTNESS_NONIBSI_MODE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_ngldm_matrix_correctness_nonibsi_mode_regression());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_GLM_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_ngldm_glm_regression());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_GLV_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_DCM_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_ngldm_dcm_regression());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_DCV_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_dcv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_DCENT_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_dcent_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_DCENE_IBSI)
{
	ASSERT_NO_THROW(test_2d_ngldm_dcene_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_NGLDM_LDE_MIRP)    { ASSERT_NO_THROW(test_2d_ngldm_lde_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_HDE_MIRP)    { ASSERT_NO_THROW(test_2d_ngldm_hde_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_LGLCE_MIRP)  { ASSERT_NO_THROW(test_2d_ngldm_lglce_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_HGLCE_MIRP)  { ASSERT_NO_THROW(test_2d_ngldm_hglce_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_LDLGLE_MIRP) { ASSERT_NO_THROW(test_2d_ngldm_ldlgle_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_LDHGLE_MIRP) { ASSERT_NO_THROW(test_2d_ngldm_ldhgle_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_HDLGLE_MIRP) { ASSERT_NO_THROW(test_2d_ngldm_hdlgle_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_HDHGLE_MIRP) { ASSERT_NO_THROW(test_2d_ngldm_hdhgle_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_GLNU_MIRP)   { ASSERT_NO_THROW(test_2d_ngldm_glnu_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_GLNUN_MIRP)  { ASSERT_NO_THROW(test_2d_ngldm_glnun_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_DCNU_MIRP)   { ASSERT_NO_THROW(test_2d_ngldm_dcnu_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_DCNUN_MIRP)  { ASSERT_NO_THROW(test_2d_ngldm_dcnun_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_GLV_MIRP)    { ASSERT_NO_THROW(test_2d_ngldm_glv_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_DCP_MIRP)    { ASSERT_NO_THROW(test_2d_ngldm_dcp_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_DCV_MIRP)    { ASSERT_NO_THROW(test_2d_ngldm_dcv_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_DCENT_MIRP)  { ASSERT_NO_THROW(test_2d_ngldm_dcent_mirp()); }
TEST(TEST_NYXUS, TEST_2D_NGLDM_DCENE_MIRP)  { ASSERT_NO_THROW(test_2d_ngldm_dcene_mirp()); }


//***** 2D intensity ***** 

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEAN_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_mean_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_SKEWNESS_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_skewness_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_KURTOSIS_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_kurtosis_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEDIAN_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_median_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MINIMUM_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_minimum_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P10_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_p10_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P90_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_p90_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_INTERQUARTILE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_interquartile_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_RANGE_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_range_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEAN_ABSOLUTE_DEVIATION_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_mean_absolute_deviation_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ENERGY_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_energy_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ROOT_MEAN_SQUARED_IBSI) 
{
	ASSERT_NO_THROW(test_2d_firstorder_root_mean_squared_ibsi());
}


//***** 2D first-order vs MATLAB (oracle_coverage.csv: oracle=matlab, target_test=test_2d_firstorder_matlab.h) *****

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_HYPERSKEWNESS_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_hyperskewness_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_HYPERFLATNESS_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_hyperflatness_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_UNIFORMITY_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_uniformity_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_UNIFORMITY_PIU_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_uniformity_piu_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_COVERED_IMAGE_INTENSITY_RANGE_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_covered_image_intensity_range_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_INTEGRATED_INTENSITY_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_integrated_intensity_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MIN_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_min_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MAX_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_max_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_RANGE_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_range_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEAN_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_mean_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEDIAN_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_median_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MODE_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_mode_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_STANDARD_DEVIATION_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_standard_deviation_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_SKEWNESS_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_skewness_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_EXCESS_KURTOSIS_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_excess_kurtosis_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_KURTOSIS_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_kurtosis_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEAN_ABSOLUTE_DEVIATION_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_mean_absolute_deviation_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_STANDARD_ERROR_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_standard_error_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ROOT_MEAN_SQUARED_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_root_mean_squared_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ENERGY_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_energy_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P10_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_p10_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_P90_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_p90_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_INTERQUARTILE_RANGE_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_interquartile_range_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_COV_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_cov_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEDIAN_ABSOLUTE_DEVIATION_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_median_absolute_deviation_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_STANDARD_DEVIATION_BIASED_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_standard_deviation_biased_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_VARIANCE_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_variance_matlab());
}
TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_VARIANCE_BIASED_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_variance_biased_matlab());
}

//***** 2D GLDM regression ***** 

TEST(TEST_NYXUS, TEST_2D_GLDM_SDE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_sde_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_lde_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_lgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_HGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_hgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_SDLGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_sdlgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_SDHGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_sdhgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDLGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_ldlgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDHGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_ldhgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_GLN_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_gln_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DN_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_dn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DNN_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_dnn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_GLV_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_glv_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DV_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_dv_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DE_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_gldm_de_regression());
}

// Bug #14b: background inside a concave ROI's bounding box must not enter the dependence matrix
TEST(TEST_NYXUS, TEST_2D_GLDM_BUG_BACKGROUND_EXCLUDED_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_gldm_bug_background_excluded_mechanics());
}


//***** 2D GLRLM regression ***** 

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_sre_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lre_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LGLRE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lglre_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_HGLRE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_hglre_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRLGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_srlgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRHGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_srhgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRLGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrlgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRHGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrhgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_gln_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLNN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_glnn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rln_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLNN_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rlnn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RP_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rp_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLV_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_glv_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RV_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rv_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_re_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_sre_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lre_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_gln_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLNN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_glnn_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rln_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RLNN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rlnn_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RP_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rp_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_GLV_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_glv_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RV_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_rv_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_RE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_re_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LGLRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lglre_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_HGLRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_hglre_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRLGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_srlgle_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_SRHGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_srhgle_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRLGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrlgle_ave_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLRLM_LRHGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_glrlm_lrhgle_ave_regression());
}


//***** 2D GLDZM regression ***** 

TEST(TEST_NYXUS, TEST_2D_GLDZM_MATRIX_CORRECTNESS_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_matrix_correctness_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_SDE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_sde_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LDE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_lde_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LGLZE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_lglze_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_HGLZE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_hglze_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_SDLGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_sdlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_SDHGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_sdhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LDLGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_ldlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LDHGLE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_ldhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLNU_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_glnu_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLNUN_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_glnun_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDNU_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdnu_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDNUN_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdnun_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZP_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_zp_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLM_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_glm_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLV_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDM_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdm_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDV_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdv_ibsi());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDE_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_zde_ibsi());
}


//***** 2D GLSZM regression ***** 

TEST(TEST_NYXUS, TEST_2D_GLSZM_SAE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_sae_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LAE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_lae_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_lglze_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_HGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_hglze_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SALGLE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_salgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SAHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_sahgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LALGLE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_lalgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LAHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_lahgle_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLN_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_gln_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLNN_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_glnn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SZN_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_szn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SZNN_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_sznn_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZP_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_zp_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLV_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_glv_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZV_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_zv_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZE_REGRESSION) {
	ASSERT_NO_THROW(test_2d_glszm_ze_regression());
}


//***** 2D NGTDM regression ***** 

TEST(TEST_NYXUS, TEST_2D_NGTDM_COARSENESS_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_ngtdm_coarseness_regression());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_CONTRAST_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_ngtdm_contrast_regression());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_BUSYNESS_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_ngtdm_busyness_regression());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_COMPLEXITY_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_ngtdm_complexity_regression());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_STRENGTH_REGRESSION) 
{
	ASSERT_NO_THROW(test_2d_ngtdm_strength_regression());
}

TEST(TEST_NYXUS, TEST_IMQ_FOCUS_SCORE_OPENCV)
{
	ASSERT_NO_THROW(test_imq_focus_score_opencv());
}

TEST(TEST_NYXUS, TEST_IMQ_LOCAL_FOCUS_SCORE_OPENCV)
{
	ASSERT_NO_THROW(test_imq_local_focus_score_opencv());
}

TEST(TEST_NYXUS, TEST_IMQ_POWER_SPECTRUM_SLOPE_REGRESSION)
{
	ASSERT_NO_THROW(test_imq_power_spectrum_slope_regression());
}

TEST(TEST_NYXUS, TEST_IMQ_MIN_SATURATION_CELLPROFILER)
{
	ASSERT_NO_THROW(test_imq_min_saturation_cellprofiler());
}

TEST(TEST_NYXUS, TEST_IMQ_MAX_SATURATION_CELLPROFILER)
{
	ASSERT_NO_THROW(test_imq_max_saturation_cellprofiler());
}

TEST(TEST_NYXUS, TEST_IMQ_SHARPNESS_REGRESSION) 
{
	ASSERT_NO_THROW(test_imq_sharpness_regression());
}


//***** 3D i/o ***** 

TEST(TEST_NYXUS, TEST_3D_NIFTI_LOADER_MECHANICS) {
	ASSERT_NO_THROW (test_3d_nifti_loader_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_NIFTI_DATA_ACCESS_CONSISTENCY_MECHANICS) {
	ASSERT_NO_THROW (test_3d_nifti_data_access_consistency_mechanics());
}


//***** OME-Zarr i/o *****

#ifdef OMEZARR_SUPPORT

TEST(TEST_NYXUS, TEST_2D_OMEZARR_TILELOADER_GEOMETRY_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_tileloader_geometry_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_TILELOADER_CONTENT_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_tileloader_content_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_TILELOADER_MULTITILE_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_tileloader_multitile_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_GEOMETRY_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_geometry_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_CONTENT_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_content_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_MULTITILE_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_multitile_mechanics());
}

#endif // OMEZARR_SUPPORT


int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}

// 3D GLCM drift guards on the ut_ segmented phantom. This file was unreachable until now:
// it carried its own definition of get_3d_segmented_phantom(), which redefines the one in
// test_3d_glcm_pyradiomics.h inside the single test_all.cc translation unit.

TEST(TEST_NYXUS, TEST_3D_GLCM_ACOR_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_acor_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ASM_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_asm_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUPROM_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_cluprom_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUSHADE_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_clushade_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUTEND_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_clutend_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CONTRAST_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_contrast_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CORRELATION_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_correlation_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFFERENCE_AVERAGE_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_difference_average_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFFERENCE_ENTROPY_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_difference_entropy_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFFERENCE_VARIANCE_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_difference_variance_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIS_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_dis_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ID_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_id_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDN_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_idn_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDM_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_idm_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDMN_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_idmn_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS1_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_infomeas1_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS2_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_infomeas2_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IV_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_iv_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JAVE_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_jave_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JE_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_je_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JMAX_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_jmax_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JVAR_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_jvar_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUM_AVERAGE_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_sum_average_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUM_ENTROPY_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_sum_entropy_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUM_VARIANCE_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_sum_variance_regression()); }

TEST(TEST_NYXUS, TEST_3D_GLCM_DUMP_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_dump_regression()); }

// 3D GLCM "grey64" drift guards, ported from the retired Wave-9 coverage sweep (formerly
// test_3d_glcm_coverage.h's glcm_3d_regression_coverage_ref_vals + its two INSTANTIATE_TEST_SUITE_P
// calls). Same phantom/values, named individually instead of swept via TEST_P.
TEST(TEST_NYXUS, TEST_3D_GLCM_ACOR_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_acor_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ASM_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_asm_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUPROM_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_cluprom_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUSHADE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_clushade_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUTEND_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_clutend_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CONTRAST_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_contrast_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CORRELATION_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_correlation_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFAVE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_difave_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFENTRO_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_difentro_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFVAR_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_difvar_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIS_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_dis_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIS_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_dis_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ENERGY_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_energy_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ENERGY_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_energy_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ENTROPY_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_entropy_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ENTROPY_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_entropy_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_HOM1_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_hom1_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_HOM1_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_hom1_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_HOM2_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_hom2_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDMN_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_idmn_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDM_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_idm_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDN_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_idn_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ID_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_id_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS1_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_infomeas1_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS2_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_infomeas2_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IV_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_iv_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JAVE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_jave_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_je_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JMAX_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_jmax_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JVAR_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_jvar_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUMAVERAGE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_sumaverage_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUMENTROPY_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_sumentropy_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUMVARIANCE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_sumvariance_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUMVARIANCE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_sumvariance_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_VARIANCE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_variance_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_VARIANCE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glcm_variance_ave_grey64_regression()); }

// 3D GLCM _AVE features: the aggregation PyRadiomics actually reports
TEST(TEST_NYXUS, TEST_3D_GLCM_ACOR_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_acor_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ASM_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_asm_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUPROM_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_cluprom_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUSHADE_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_clushade_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CLUTEND_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_clutend_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CONTRAST_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_contrast_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_CORRELATION_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_correlation_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFAVE_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_difave_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFENTRO_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_difentro_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_DIFVAR_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_difvar_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_ID_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_id_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDM_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_idm_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDMN_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_idmn_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IDN_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_idn_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS1_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_infomeas1_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_INFOMEAS2_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_infomeas2_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_IV_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_iv_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JAVE_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_jave_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JE_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_je_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JMAX_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_jmax_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_JVAR_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_jvar_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUMAVERAGE_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_sumaverage_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_SUMENTROPY_AVE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_sumentropy_ave_pyradiomics()); }
TEST(TEST_NYXUS, TEST_3D_GLCM_AVE_EQUIVALENCE_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_ave_equivalence_pyradiomics()); }


// JVAR had a complete assertion that no TEST() ever called (not_covered.md B.2)
TEST(TEST_NYXUS, TEST_3D_GLCM_JVAR_PYRADIOMICS) { ASSERT_NO_THROW(test_3d_glcm_jvar_pyradiomics()); }
