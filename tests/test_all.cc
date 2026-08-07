#define NOMINMAX	// keep Windows min/max macros from breaking dcmtk's OFvariant (DICOM tests)
#include <gtest/gtest.h>
#include "test_gabor_skimage.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "test_contour.h"
#include "test_firstorder_common.h"
#include "test_firstorder_regression.h"
#include "test_firstorder_matlab.h"
#include "test_intensity_histogram_regression.h"
#include "test_intensity_histogram_ibsi.h"
#include "test_hu_analytic.h"
#include "test_hu_mechanics.h"
#include "test_morphology_features.h"
#include "test_morphology_regression.h"
#include "test_morphology_analytic.h"
#include "test_morphology_imea.h"
#include "test_morphology_skimage.h"
#include "test_morphology_matlab.h"
#include "test_morphology_fraclac.h"
#include "test_moments_skimage.h"
#include "test_moments_regression.h"
#include "test_remaining2d_common.h"
#include "test_zernike_regression.h"
#include "test_neighbor_regression.h"
#include "test_neighbor_cellprofiler.h"
#include "test_neighbor_analytic.h"
#include "test_initialization_mechanics.h"
#include "test_glcm_ibsi.h"
#include "test_gldm_ibsi.h"
#include "test_glrlm_ibsi.h"
#include "test_gldzm_ibsi.h"
#include "test_glszm_ibsi.h"
#include "test_firstorder_ibsi.h"
#include "test_firstorder_pyradiomics.h"
#include "test_ngldm_ibsi.h"
#include "test_ngtdm_ibsi.h"
#include "test_glcm_regression.h"
#include "test_gldm_regression.h"
#include "test_gldm_mechanics.h"
#include "test_glrlm_regression.h"
#include "test_glszm_regression.h"
#include "test_ngtdm_regression.h"
#include "test_roi_blacklist_mechanics.h"
#include "test_tiff_loader_mechanics.h"
#include "test_imq_regression.h"
#include "test_imq_opencv.h"
#include "test_imq_cellprofiler.h"
#include "test_3d_nifti_mechanics.h"
#include "test_omezarr_mechanics.h"
#include "test_3d_morphology_common.h"
#include "test_3d_morphology_regression.h"
#include "test_3d_morphology_matlab.h"
#include "test_3d_gldzm_ibsi.h"
#include "test_3d_ngldm_regression.h"
#include "test_3d_firstorder_pyradiomics.h"
#include "test_3d_glcm_pyradiomics.h"
#include "test_3d_gldm_pyradiomics.h"
#include "test_3d_ngtdm_pyradiomics.h"
#include "test_3d_glrlm_pyradiomics.h"
#include "test_3d_glszm_pyradiomics.h"
#include "test_3d_coverage_common.h"
#include "test_3d_firstorder_coverage.h"
#include "test_3d_morphology_coverage.h"
#include "test_3d_glcm_coverage.h"
#include "test_3d_gldm_coverage.h"
#include "test_3d_gldzm_coverage.h"
#include "test_3d_glrlm_coverage.h"
#include "test_3d_glszm_coverage.h"
#include "test_3d_ngldm_coverage.h"
#include "test_3d_ngtdm_coverage.h"
#include "test_glcm_mechanics.h"
#ifdef USE_ARROW
    #include "test_arrow_mechanics.h"
    #include "test_arrow_file_name_mechanics.h"
#endif


//***** 2D contour and multicontour *****

TEST(TEST_NYXUS, TEST_CONTOUR_MULTI_1) {
	ASSERT_NO_THROW(test_contour_multi_disconnected());
}

TEST(TEST_NYXUS, TEST_CONTOUR_SINGLE) {
	ASSERT_NO_THROW(test_contour_single());
}

TEST(TEST_NYXUS, TEST_CONTOUR_SINGLE_TAILED) {
	ASSERT_NO_THROW(test_contour_single_tailed());
}

TEST(TEST_NYXUS, TEST_CONTOUR_VOID) {
	ASSERT_NO_THROW(test_contour_void());
}

TEST(TEST_NYXUS, TEST_CONTOUR_MULTI_2) {
	ASSERT_NO_THROW(test_contour_multi_connected());
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

TEST(TEST_NYXUS, TEST_3DGLCM_EQUIVALENCE_DUMP) {
	ASSERT_NO_THROW(test_3dglcm_equivalence_dump());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_ACOR) {
	ASSERT_NO_THROW(test_compat_3glcm_ACOR());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_ANGULAR_2D_MOMENT) {
	ASSERT_NO_THROW(test_compat_3glcm_angular_2d_moment());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_CLUPROM) {
	ASSERT_NO_THROW(test_compat_3glcm_CLUPROM());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_CLUSHADE) {
	ASSERT_NO_THROW(test_compat_3glcm_CLUSHADE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_CLUTEND) {
	ASSERT_NO_THROW(test_compat_3glcm_CLUTEND());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_CONTRAST) {
	ASSERT_NO_THROW(test_compat_3glcm_contrast());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_CORRELATION) {
	ASSERT_NO_THROW(test_compat_3glcm_correlation());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_DIFFERENCE_AVERAGE) {
	ASSERT_NO_THROW(test_compat_3glcm_difference_average());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_DIFFERENCE_VARIANCE) {
	ASSERT_NO_THROW(test_compat_3glcm_difference_variance());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_DIFFERENCE_ENTROPY) {
	ASSERT_NO_THROW(test_compat_3glcm_difference_entropy());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_DIFFERENCE_ID) {
	ASSERT_NO_THROW(test_compat_3glcm_ID());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_DIFFERENCE_IDN) {
	ASSERT_NO_THROW(test_compat_3glcm_IDN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_DIFFERENCE_IDM) {
	ASSERT_NO_THROW(test_compat_3glcm_IDM());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_DIFFERENCE_IDMN) {
	ASSERT_NO_THROW(test_compat_3glcm_IDMN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_INFOMEAS1) {
	ASSERT_NO_THROW(test_compat_3glcm_infomeas1());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_INFOMEAS2) {
	ASSERT_NO_THROW(test_compat_3glcm_infomeas2());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_IV) {
	ASSERT_NO_THROW(test_compat_3glcm_IV());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_JAVE) {
	ASSERT_NO_THROW(test_compat_3glcm_JAVE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_JE) {
	ASSERT_NO_THROW(test_compat_3glcm_JE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_JMAX) {
	ASSERT_NO_THROW(test_compat_3glcm_JMAX());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_SUM_AVERAGE) {
	ASSERT_NO_THROW(test_compat_3glcm_sum_average());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLCM_SUM_ENTROPY) {
	ASSERT_NO_THROW(test_compat_3glcm_sum_entropy());
}


//***** Apache I/O tests *****

#ifdef USE_ARROW

TEST(TEST_NYXUS, TEST_ARROW_FILE_NAME) {
	test_file_naming();
}

TEST(TEST_NYXUS, TEST_ARROW) {
	test_arrow();
}

TEST(TEST_NYXUS, TEST_PARQUET) {
	test_parquet();
}

#endif


//***** 3D shape *****

TEST(TEST_NYXUS, TEST_3SHAPE_3MESH_VOLUME) {
	ASSERT_NO_THROW(test_3shape_meshvolume());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3AREA) {
	ASSERT_NO_THROW(test_3shape_area());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3AREA_2_VOLUME) {
	ASSERT_NO_THROW(test_3shape_area2volume());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3COMPACTNESS1) {
	ASSERT_NO_THROW(test_3shape_compactness1());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3COMPACTNESS2) {
	ASSERT_NO_THROW(test_3shape_compactness2());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3SPHERICAL_DISPROPORTION) {
	ASSERT_NO_THROW(test_3shape_sprericaldisproportion());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3SPHERICITY) {
	ASSERT_NO_THROW(test_3shape_sphericity());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3VOLUME_CONVEXHULL) {
	ASSERT_NO_THROW(test_3shape_volumeconvhull());
}

TEST(TEST_NYXUS, TEST_3SHAPE_3VOXEL_VOLUME) {
	ASSERT_NO_THROW(test_3shape_voxelvolume());
}

TEST(TEST_NYXUS, TEST_3SHAPE_COVMATRIX_AND_EIGENVALS) {
	ASSERT_NO_THROW(test_3shape_covmatrix_and_eigenvals());
}


//***** 3D GLDZM regression *****

TEST(TEST_NYXUS, TEST_3GLDZM_GLM) {
	ASSERT_NO_THROW(test_3GLDZM_GLM());
}

TEST(TEST_NYXUS, TEST_3GLDZM_GLV) {
	ASSERT_NO_THROW(test_3GLDZM_GLV());
}

TEST(TEST_NYXUS, TEST_3GLDZM_LDE) {
	ASSERT_NO_THROW(test_3GLDZM_LDE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_SDE) {
	ASSERT_NO_THROW(test_3GLDZM_SDE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_LGLZE) {
	ASSERT_NO_THROW(test_3GLDZM_LGLZE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_HGLZE) {
	ASSERT_NO_THROW(test_3GLDZM_HGLZE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_SDLGLE) {
	ASSERT_NO_THROW(test_3GLDZM_SDLGLE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_SDHGLE) {
	ASSERT_NO_THROW(test_3GLDZM_SDHGLE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_LDLGLE) {
	ASSERT_NO_THROW(test_3GLDZM_LDLGLE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_LDHGLE) {
	ASSERT_NO_THROW(test_3GLDZM_LDHGLE());
}

TEST(TEST_NYXUS, TEST_3GLDZM_GLNU) {
	ASSERT_NO_THROW(test_3GLDZM_GLNU());
}

TEST(TEST_NYXUS, TEST_3GLDZM_GLNUN) {
	ASSERT_NO_THROW(test_3GLDZM_GLNUN());
}

TEST(TEST_NYXUS, TEST_3GLDZM_ZDNU) {
	ASSERT_NO_THROW(test_3GLDZM_ZDNU());
}

TEST(TEST_NYXUS, TEST_3GLDZM_ZDNUN) {
	ASSERT_NO_THROW(test_3GLDZM_ZDNUN());
}

TEST(TEST_NYXUS, TEST_3GLDZM_ZDV) {
	ASSERT_NO_THROW(test_3GLDZM_ZDV());
}

TEST(TEST_NYXUS, TEST_3GLDZM_ZP) {
	ASSERT_NO_THROW(test_3GLDZM_ZP());
}

TEST(TEST_NYXUS, TEST_3GLDZM_ZDE) {
	ASSERT_NO_THROW(test_3GLDZM_ZDE());
}


//***** 3D NGLDM regression *****

TEST(TEST_NYXUS, TEST_3NGLDM_LDE) {
	ASSERT_NO_THROW(test_3ngldm_lde());
}

TEST(TEST_NYXUS, TEST_3NGLDM_HDE) {
	ASSERT_NO_THROW(test_3ngldm_hde());
}

TEST(TEST_NYXUS, TEST_3NGLDM_LGLCE) {
	ASSERT_NO_THROW(test_3ngldm_lglce());
}

TEST(TEST_NYXUS, TEST_3NGLDM_HGLCE) {
	ASSERT_NO_THROW(test_3ngldm_hglce());
}

TEST(TEST_NYXUS, TEST_3NGLDM_LDLGLE) {
	ASSERT_NO_THROW(test_3ngldm_ldlgle());
}

TEST(TEST_NYXUS, TEST_3NGLDM_LDHGLE) {
	ASSERT_NO_THROW(test_3ngldm_ldhgle());
}

TEST(TEST_NYXUS, TEST_3NGLDM_HDLGLE) {
	ASSERT_NO_THROW(test_3ngldm_hdlgle());
}

TEST(TEST_NYXUS, TEST_3NGLDM_HDHGLE) {
	ASSERT_NO_THROW(test_3ngldm_hdhgle());
}

TEST(TEST_NYXUS, TEST_3NGLDM_GLNU) {
	ASSERT_NO_THROW(test_3ngldm_glnu());
}

TEST(TEST_NYXUS, TEST_3NGLDM_GLNUN) {
	ASSERT_NO_THROW(test_3ngldm_glnun());
}

TEST(TEST_NYXUS, TEST_3NGLDM_DCNU) {
	ASSERT_NO_THROW(test_3ngldm_dcnu());
}

TEST(TEST_NYXUS, TEST_3NGLDM_DCNUN) {
	ASSERT_NO_THROW(test_3ngldm_dcnun());
}

TEST(TEST_NYXUS, TEST_3NGLDM_DCP) {
	ASSERT_NO_THROW(test_3ngldm_dcp());
}

TEST(TEST_NYXUS, TEST_3NGLDM_GLM) {
	ASSERT_NO_THROW(test_3ngldm_glm());
}

TEST(TEST_NYXUS, TEST_3NGLDM_GLV) {
	ASSERT_NO_THROW(test_3ngldm_glv());
}

TEST(TEST_NYXUS, TEST_3NGLDM_DCM) {
	ASSERT_NO_THROW(test_3ngldm_dcm());
}

TEST(TEST_NYXUS, TEST_3NGLDM_DCV) {
	ASSERT_NO_THROW(test_3ngldm_dcv());
}

TEST(TEST_NYXUS, TEST_3NGLDM_DCENT) {
	ASSERT_NO_THROW(test_3ngldm_dcent());
}

TEST(TEST_NYXUS, TEST_3NGLDM_DCENE) {
	ASSERT_NO_THROW(test_3ngldm_dcene());
}


//***** Gabor (vetted vs scikit-image) *****

TEST(TEST_NYXUS, TEST_GABOR_SKIMAGE){
    test_gabor_skimage();

    #ifdef USE_GPU
        test_gabor_skimage(true);
    #endif
}


//***** helper functionality ***** 

TEST(TEST_NYXUS, TEST_ROI_BLACKLISTING)
{
	ASSERT_NO_THROW(test_roi_blacklist());
}

TEST(TEST_NYXUS, TEST_TIFF_UINT32_STRIP_LOADER)
{
	ASSERT_NO_THROW(test_uint32_strip_tiff_loader());
}

TEST(TEST_NYXUS, TEST_INITIALIZATION) {
	test_initialization();
}


//***** Pixel intensity features ***** 

TEST(TEST_NYXUS, TEST_FIRSTORDER_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_firstorder_pyradiomics());
}

//***** IBSI Intensity Histogram (IH) family *****

TEST(TEST_NYXUS, TEST_IH_INTEGER_DOMAIN_VALUES)
{
	ASSERT_NO_THROW(test_ih_integer_domain_values());
}

TEST(TEST_NYXUS, TEST_IH_INDEX_AND_PERCENTILE_BOUNDS)
{
	ASSERT_NO_THROW(test_ih_index_and_percentile_bounds());
}

TEST(TEST_NYXUS, TEST_IH_IBSI_GATE_OFF_RETURNS_NAN)
{
	ASSERT_NO_THROW(test_ih_ibsi_gate_off_returns_nan());
}

TEST(TEST_NYXUS, TEST_IH_FLOAT_DOMAIN_RECONSTRUCTION)
{
	ASSERT_NO_THROW(test_ih_float_domain_reconstruction());
}

TEST(TEST_NYXUS, TEST_IH_FLOAT_DOMAIN_RECONSTRUCTION_NEGATIVE_MIN)
{
	ASSERT_NO_THROW(test_ih_float_domain_reconstruction_negative_min());
}

TEST(TEST_NYXUS, TEST_IH_FLOAT_DOMAIN_RECONSTRUCTION_PRESERVE_HU)
{
	ASSERT_NO_THROW(test_ih_float_domain_reconstruction_preserve_hu());
}

TEST(TEST_NYXUS, TEST_IH_FLOAT_DOMAIN_RECONSTRUCTION_PRESERVE_HU_FPACTIVE)
{
	ASSERT_NO_THROW(test_ih_float_domain_reconstruction_preserve_hu_fpactive());
}

TEST(TEST_NYXUS, TEST_IH_REQUIRED_PREDICATE)
{
	ASSERT_NO_THROW(test_ih_required_predicate());
}

TEST(TEST_NYXUS, TEST_IH_DISPERSION_IBSI)
{
	ASSERT_NO_THROW(test_ih_dispersion_ibsi());
}

TEST(TEST_NYXUS, TEST_IH_DISPERSION_ROBUST_ANALYTIC)
{
	ASSERT_NO_THROW(test_ih_dispersion_robust_analytic());
}

TEST(TEST_NYXUS, TEST_IH_HISTOGRAM_ANALYTIC)
{
	ASSERT_NO_THROW(test_ih_histogram_analytic());
}

TEST(TEST_NYXUS, TEST_HU_UINT_FRIENDLY_NORMALIZATION_CT_RANGE)
{
	ASSERT_NO_THROW(test_hu_uint_friendly_normalization_ct_range());
}

TEST(TEST_NYXUS, TEST_HU_UINT_FRIENDLY_RAWCAST_NONNEGATIVE)
{
	ASSERT_NO_THROW(test_hu_uint_friendly_rawcast_nonnegative());
}

TEST(TEST_NYXUS, TEST_HU_UINT_FRIENDLY_PRESERVE_OFFSET)
{
	ASSERT_NO_THROW(test_hu_uint_friendly_preserve_offset());
}

TEST(TEST_NYXUS, TEST_HU_FPIMAGE_OPTIONS_PARSE)
{
	ASSERT_NO_THROW(test_hu_fpimage_options_parse());
}

TEST(TEST_NYXUS, TEST_HU_LOADER_INT16_PRESERVE)
{
	ASSERT_NO_THROW(test_hu_loader_int16_preserve());
}

TEST(TEST_NYXUS, TEST_HU_LOADER_FLOAT_PRESERVE)
{
	ASSERT_NO_THROW(test_hu_loader_float_preserve());
}

TEST(TEST_NYXUS, TEST_HU_LOADER_FLOAT_NONPRESERVE_BASELINE)
{
	ASSERT_NO_THROW(test_hu_loader_float_nonpreserve_baseline());
}

#ifdef DICOM_SUPPORT
TEST(TEST_NYXUS, TEST_HU_LOADER_DICOM_U16_PRESERVE)
{
	ASSERT_NO_THROW(test_hu_loader_dicom_u16_preserve());
}

TEST(TEST_NYXUS, TEST_HU_LOADER_DICOM_I16_PRESERVE)
{
	ASSERT_NO_THROW(test_hu_loader_dicom_i16_preserve());
}

TEST(TEST_NYXUS, TEST_HU_LOADER_DICOM_CT_SMALL_PRESERVE)
{
	ASSERT_NO_THROW(test_hu_loader_dicom_ct_small_preserve());
}

TEST(TEST_NYXUS, TEST_HU_LOADER_DICOM_CT_SMALL_BASELINE)
{
	ASSERT_NO_THROW(test_hu_loader_dicom_ct_small_baseline());
}
#endif

TEST(TEST_NYXUS, TEST_FIRSTORDER_ROBUST_MEAN_ABSOLUTE_DEVIATION_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_firstorder_robust_mean_absolute_deviation_pyradiomics());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_ENTROPY_REGRESSION) 
{
	ASSERT_NO_THROW(test_firstorder_entropy_regression());
}

//***** Morphology features ***** 

TEST(TEST_NYXUS, TEST_MORPHOLOGY_PERIMETER) 
{
	ASSERT_NO_THROW(test_morphology_perimeter());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_BASIC_MORPHOLOGY_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_basic_morphology_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_ELLIPSE_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_ellipse_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_CONTOUR_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_contour_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_CONVEX_HULL_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_convex_hull_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_SKIMAGE_ORIENTATION_AND_EROSIONS)
{
	ASSERT_NO_THROW(test_shape2d_skimage_orientation_and_erosions());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_EXTREMA_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_verifiable_with_3p_builtin_oracle_extrema_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_MISC_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_misc_shape_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_FRACTAL_CIRCLE_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_verifiable_with_3p_builtin_oracle_fractal_circle_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_FRACTAL_DIMENSION_BLOB512_ORACLE)
{
	ASSERT_NO_THROW(test_shape2d_fractal_dimension_blob512_oracle());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_UNVETTED_NO_DIRECT_ORACLE_RADIUS_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_unvetted_no_direct_oracle_radius_features());
}

TEST(TEST_NYXUS, TEST_MOMENTS_SHAPE_SKIMAGE)
{
	ASSERT_NO_THROW(test_moments_shape_skimage());
}

TEST(TEST_NYXUS, TEST_MOMENTS_HU_WEDGE_SKIMAGE)
{
	ASSERT_NO_THROW(test_moments_hu_wedge_skimage());
}

TEST(TEST_NYXUS, TEST_MOMENTS_SHAPE_REGRESSION)
{
	ASSERT_NO_THROW(test_moments_shape_regression());
}

TEST(TEST_NYXUS, TEST_MOMENTS_INTENSITY_SKIMAGE)
{
	ASSERT_NO_THROW(test_moments_intensity_skimage());
}

TEST(TEST_NYXUS, TEST_MOMENTS_INTENSITY_REGRESSION)
{
	ASSERT_NO_THROW(test_moments_intensity_regression());
}

TEST(TEST_NYXUS, TEST_REMAINING2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_EROSION_COMPLEMENT)
{
	ASSERT_NO_THROW(test_remaining2d_verifiable_with_3p_builtin_oracle_erosion_complement_feature());
}

TEST(TEST_NYXUS, TEST_REMAINING2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_CALIPER_FEATURES)
{
	ASSERT_NO_THROW(test_remaining2d_verifiable_with_3p_builtin_oracle_caliper_features());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_CALIPER_MARTIN_NASSENSTEIN_IMEA_ELLIPSE_ORACLE)
{
	ASSERT_NO_THROW(test_shape2d_caliper_martin_nassenstein_imea_ellipse_oracle());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_CALIPER_FERET_IMEA_ELLIPSE_ORACLE)
{
	ASSERT_NO_THROW(test_shape2d_caliper_feret_imea_ellipse_oracle());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_MIN_ENCLOSING_CIRCLE_IMEA_ORACLE)
{
	ASSERT_NO_THROW(test_shape2d_min_enclosing_circle_imea_oracle());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_DOCUMENTED_FORMULA_CONFORMANCE_ANALYTIC)
{
	ASSERT_NO_THROW(test_shape2d_documented_formula_conformance_analytic());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_DIAMETER_EQUAL_PERIMETER_IMEA)
{
	ASSERT_NO_THROW(test_shape2d_diameter_equal_perimeter_imea());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_GEODETIC_LENGTH_THICKNESS_IMEA)
{
	ASSERT_NO_THROW(test_shape2d_geodetic_length_thickness_imea());
}

TEST(TEST_NYXUS, TEST_REMAINING2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_CHORD_STAT_FEATURES)
{
	ASSERT_NO_THROW(test_remaining2d_verifiable_with_3p_builtin_oracle_chord_stat_features());
}

TEST(TEST_NYXUS, TEST_REMAINING2D_UNVETTED_NO_DIRECT_ORACLE_CHORD_ANGLE_FEATURES)
{
	ASSERT_NO_THROW(test_remaining2d_unvetted_no_direct_oracle_chord_angle_features());
}

TEST(TEST_NYXUS, TEST_REMAINING2D_UNVETTED_NO_DIRECT_ORACLE_POLYGONALITY_HEXAGONALITY)
{
	ASSERT_NO_THROW(test_remaining2d_unvetted_no_direct_oracle_polygonality_hexagonality_features());
}

TEST(TEST_NYXUS, TEST_REMAINING2D_UNVETTED_NO_DIRECT_ORACLE_RADIAL_DISTRIBUTION)
{
	ASSERT_NO_THROW(test_remaining2d_unvetted_no_direct_oracle_radial_distribution_features());
}

TEST(TEST_NYXUS, TEST_ZERNIKE_MOMENTS_REGRESSION)
{
	ASSERT_NO_THROW(test_zernike_moments_regression());
}

TEST(TEST_NYXUS, TEST_NEIGHBOR_COUNTS_AND_TOUCHING_REGRESSION)
{
	ASSERT_NO_THROW(test_neighbor_counts_and_touching_regression());
}

TEST(TEST_NYXUS, TEST_NEIGHBOR_PERCENT_TOUCHING_ENCLOSED_ANALYTIC)
{
	ASSERT_NO_THROW(test_neighbor_percent_touching_enclosed_analytic());
}

TEST(TEST_NYXUS, TEST_NEIGHBOR_CLOSEST_NEIGHBORS_REGRESSION)
{
	ASSERT_NO_THROW(test_neighbor_closest_neighbors_regression());
}

TEST(TEST_NYXUS, TEST_NEIGHBOR_COUNTS_AND_FIRST_DISTANCE_CELLPROFILER)
{
	ASSERT_NO_THROW(test_neighbor_counts_and_first_distance_cellprofiler());
}

TEST(TEST_NYXUS, TEST_NEIGHBOR_SECOND_DISTANCE_AND_ANGLES_ANALYTIC)
{
	ASSERT_NO_THROW(test_neighbor_second_distance_and_angles_analytic());
}


//***** IBSI tests of NGTDM

TEST(TEST_NYXUS, TEST_NGTDM_COARSENESS_IBSI)
{
	ASSERT_NO_THROW(test_ngtdm_coarseness_ibsi());
}

TEST(TEST_NYXUS, TEST_NGTDM_CONTRAST_IBSI)
{
	ASSERT_NO_THROW(test_ngtdm_contrast_ibsi());
}

TEST(TEST_NYXUS, TEST_NGTDM_BUSYNESS_IBSI)
{
	ASSERT_NO_THROW(test_ngtdm_busyness_ibsi());
}

TEST(TEST_NYXUS, TEST_NGTDM_COMPLEXITY_IBSI)
{
	ASSERT_NO_THROW(test_ngtdm_complexity_ibsi());
}

TEST(TEST_NYXUS, TEST_NGTDM_STRENGTH_IBSI)
{
	ASSERT_NO_THROW(test_ngtdm_strength_ibsi());
}


//***** IBSI tests of GLCM ***** 

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

TEST(TEST_NYXUS, TEST_IBSI_GLCM_HOM2)
{
	ASSERT_NO_THROW(test_ibsi_glcm_HOM2());
}

TEST(TEST_NYXUS, TEST_IBSI_GLCM_ENTROPY)
{
	ASSERT_NO_THROW(test_ibsi_glcm_ENTROPY());
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


//***** 2D GLCM regression ***** 

TEST(TEST_NYXUS, TEST_GLCM_ACOR)
{
	ASSERT_NO_THROW(test_glcm_ACOR());
}

TEST(TEST_NYXUS, TEST_GLCM_ASM)
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

TEST(TEST_NYXUS, TEST_GLCM_CONTRAST)
{
	ASSERT_NO_THROW(test_glcm_contrast());
}

TEST(TEST_NYXUS, TEST_GLCM_CORRELATION)
{
	ASSERT_NO_THROW(test_glcm_correlation());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFAVE)
{
	ASSERT_NO_THROW(test_glcm_difference_average());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFENTRO)
{
	ASSERT_NO_THROW(test_glcm_difference_entropy());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFVAR)
{
	ASSERT_NO_THROW(test_glcm_difference_variance());
}

TEST(TEST_NYXUS, TEST_GLCM_DIS)
{
	ASSERT_NO_THROW(test_glcm_DIS());
}

TEST(TEST_NYXUS, TEST_GLCM_ENERGY)
{
	ASSERT_NO_THROW(test_glcm_energy());
}

TEST(TEST_NYXUS, TEST_GLCM_ENTROPY)
{
	ASSERT_NO_THROW(test_glcm_entropy());
}

TEST(TEST_NYXUS, TEST_GLCM_HOM1)
{
	ASSERT_NO_THROW(test_glcm_hom1());
}

TEST(TEST_NYXUS, TEST_GLCM_HOM2)
{
	ASSERT_NO_THROW(test_glcm_hom2());
}

TEST(TEST_NYXUS, TEST_GLCM_ID)
{
	ASSERT_NO_THROW(test_glcm_ID());
}

TEST(TEST_NYXUS, TEST_GLCM_IDN)
{
	ASSERT_NO_THROW(test_glcm_IDN());
}

TEST(TEST_NYXUS, TEST_GLCM_IDM)
{
	ASSERT_NO_THROW(test_glcm_IDM());
}

TEST(TEST_NYXUS, TEST_GLCM_IDMN)
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

TEST(TEST_NYXUS, TEST_GLCM_SUMAVERAGE)
{
	ASSERT_NO_THROW(test_glcm_sum_average());
}

TEST(TEST_NYXUS, TEST_GLCM_SUMENTROPY)
{
	ASSERT_NO_THROW(test_glcm_sum_entropy());
}

TEST(TEST_NYXUS, TEST_GLCM_SUMVARIANCE)
{
	ASSERT_NO_THROW(test_glcm_sum_variance());
}

TEST(TEST_NYXUS, TEST_GLCM_VARIANCE)
{
	ASSERT_NO_THROW(test_glcm_variance());
}

TEST(TEST_NYXUS, TEST_GLCM_ASM_AVE)
{
	ASSERT_NO_THROW(test_glcm_ASM_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_ACOR_AVE)
{
	ASSERT_NO_THROW(test_glcm_ACOR_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_CLUPROM_AVE)
{
	ASSERT_NO_THROW(test_glcm_CLUPROM_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_CLUSHADE_AVE)
{
	ASSERT_NO_THROW(test_glcm_CLUSHADE_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_CLUTEND_AVE)
{
	ASSERT_NO_THROW(test_glcm_CLUTEND_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_CONTRAST_AVE)
{
	ASSERT_NO_THROW(test_glcm_CONTRAST_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_CORRELATION_AVE)
{
	ASSERT_NO_THROW(test_glcm_CORRELATION_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFAVE_AVE)
{
	ASSERT_NO_THROW(test_glcm_DIFAVE_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFENTRO_AVE)
{
	ASSERT_NO_THROW(test_glcm_DIFENTRO_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_DIFVAR_AVE)
{
	ASSERT_NO_THROW(test_glcm_DIFVAR_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_DIS_AVE)
{
	ASSERT_NO_THROW(test_glcm_DIS_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_ENERGY_AVE)
{
	ASSERT_NO_THROW(test_glcm_ENERGY_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_ENTROPY_AVE)
{
	ASSERT_NO_THROW(test_glcm_ENTROPY_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_HOM1_AVE)
{
	ASSERT_NO_THROW(test_glcm_HOM1_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_ID_AVE)
{
	ASSERT_NO_THROW(test_glcm_ID_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_IDN_AVE)
{
	ASSERT_NO_THROW(test_glcm_IDN_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_IDM_AVE)
{
	ASSERT_NO_THROW(test_glcm_IDM_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_IDMN_AVE)
{
	ASSERT_NO_THROW(test_glcm_IDMN_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_IV_AVE)
{
	ASSERT_NO_THROW(test_glcm_IV_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_JAVE_AVE)
{
	ASSERT_NO_THROW(test_glcm_JAVE_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_JE_AVE)
{
	ASSERT_NO_THROW(test_glcm_JE_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_INFOMEAS1_AVE)
{
	ASSERT_NO_THROW(test_glcm_INFOMEAS1_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_INFOMEAS2_AVE)
{
	ASSERT_NO_THROW(test_glcm_INFOMEAS2_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_VARIANCE_AVE)
{
	ASSERT_NO_THROW(test_glcm_VARIANCE_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_JMAX_AVE)
{
	ASSERT_NO_THROW(test_glcm_JMAX_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_JVAR_AVE)
{
	ASSERT_NO_THROW(test_glcm_JVAR_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_SUMAVERAGE_AVE)
{
	ASSERT_NO_THROW(test_glcm_SUMAVERAGE_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_SUMENTROPY_AVE)
{
	ASSERT_NO_THROW(test_glcm_SUMENTROPY_AVE());
}

TEST(TEST_NYXUS, TEST_GLCM_SUMVARIANCE_AVE)
{
	ASSERT_NO_THROW(test_glcm_SUMVARIANCE_AVE());
}

// Regression guard: GLCM co-occurrence distance must default to 1 via the production
// settings path (exposes the offset=0 default defect that the hard-coded tests above miss).
TEST(TEST_NYXUS, TEST_GLCM_BUG_OFFSET_DEFAULT)
{
	ASSERT_NO_THROW(test_glcm_bug_offset_default_is_one());
}

//***** IBSI tests of GLDM *****

TEST(TEST_NYXUS, TEST_GLDM_SDE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_sde_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_LDE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_lde_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_LGLE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_lgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_HGLE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_hgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_SDLGLE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_sdlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_SDHGLE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_sdhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_LDLGLE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_ldlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_LDHGLE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_ldhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_GLN_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_gln_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_DN_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_dn_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_DNN_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_dnn_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_GLV_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_DV_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_dv_ibsi());
}

TEST(TEST_NYXUS, TEST_GLDM_DE_IBSI) 
{
	ASSERT_NO_THROW(test_gldm_de_ibsi());
}


//***** IBSI tests of GLRLM ***** 

TEST(TEST_NYXUS, TEST_GLRLM_SRE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_sre_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_lre_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_LGLRE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_lglre_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_HGLRE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_hglre_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRLGLE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_srlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRHGLE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_srhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRLGLE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_lrlgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRHGLE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_lrhgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_LGLRE_AVE_IBSI)  { ASSERT_NO_THROW(test_glrlm_lglre_ave_ibsi()); }
TEST(TEST_NYXUS, TEST_GLRLM_HGLRE_AVE_IBSI)  { ASSERT_NO_THROW(test_glrlm_hglre_ave_ibsi()); }
TEST(TEST_NYXUS, TEST_GLRLM_SRLGLE_AVE_IBSI) { ASSERT_NO_THROW(test_glrlm_srlgle_ave_ibsi()); }
TEST(TEST_NYXUS, TEST_GLRLM_SRHGLE_AVE_IBSI) { ASSERT_NO_THROW(test_glrlm_srhgle_ave_ibsi()); }
TEST(TEST_NYXUS, TEST_GLRLM_LRLGLE_AVE_IBSI) { ASSERT_NO_THROW(test_glrlm_lrlgle_ave_ibsi()); }
TEST(TEST_NYXUS, TEST_GLRLM_LRHGLE_AVE_IBSI) { ASSERT_NO_THROW(test_glrlm_lrhgle_ave_ibsi()); }

TEST(TEST_NYXUS, TEST_GLRLM_GLN_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_gln_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLNN_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_glnn_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLN_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_rln_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLNN_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_rlnn_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_RP_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_rp_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLV_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_RV_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_rv_ibsi());
}

TEST(TEST_NYXUS, TEST_GLRLM_RE_IBSI)
{
	ASSERT_NO_THROW(test_glrlm_re_ibsi());
}


//***** IBSI tests of GLSZM ***** 

TEST(TEST_NYXUS, TEST_GLSZM_SAE_IBSI) {
	ASSERT_NO_THROW(test_glszm_sae_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_LAE_IBSI) {
	ASSERT_NO_THROW(test_glszm_lae_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_LGLZE_IBSI) {
	ASSERT_NO_THROW(test_glszm_lglze_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_HGLZE_IBSI) {
	ASSERT_NO_THROW(test_glszm_hglze_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_SALGLE_IBSI) {
	ASSERT_NO_THROW(test_glszm_salgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_SAHGLE_IBSI) {
	ASSERT_NO_THROW(test_glszm_sahgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_LALGLE_IBSI) {
	ASSERT_NO_THROW(test_glszm_lalgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_LAHGLE_IBSI) {
	ASSERT_NO_THROW(test_glszm_lahgle_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_GLN_IBSI) {
	ASSERT_NO_THROW(test_glszm_gln_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_GLNN_IBSI) {
	ASSERT_NO_THROW(test_glszm_glnn_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_SZN_IBSI) {
	ASSERT_NO_THROW(test_glszm_szn_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_SZNN_IBSI) {
	ASSERT_NO_THROW(test_glszm_sznn_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZP_IBSI) {
	ASSERT_NO_THROW(test_glszm_zp_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_GLV_IBSI) {
	ASSERT_NO_THROW(test_glszm_glv_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZV_IBSI) {
	ASSERT_NO_THROW(test_glszm_zv_ibsi());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZE_IBSI) {
	ASSERT_NO_THROW(test_glszm_ze_ibsi());
}


//***** IBSI tests of NGLDM ***** 

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_MATRIX_CORRECTNESS_IBSI)
{
	ASSERT_NO_THROW (test_ibsi_NGLDM_matrix_correctness_IBSI());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_MATRIX_CORRECTNESS_NONIBSI)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_matrix_correctness_NONIBSI());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_LDE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_LDE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_HDE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_HDE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_LGLCE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_LGLCE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_HGLCE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_HGLCE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_LDLGLE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_LDLGLE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_LDHGLE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_LDHGLE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_HDLGLE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_HDLGLE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_HDHGLE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_HDHGLE());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_GLNU)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_GLNU());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_GLNUN)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_GLNUN());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_DCNU)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_DCNU());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_DCNUN)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_DCNUN());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_DCP)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_DCP());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_UNVETTED_NO_DIRECT_ORACLE_GLM)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_unvetted_no_direct_oracle_GLM());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_GLV)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_GLV());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_UNVETTED_NO_DIRECT_ORACLE_DCM)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_unvetted_no_direct_oracle_DCM());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_DCV)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_DCV());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_DCENT)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_DCENT());
}

TEST(TEST_NYXUS, TEST_IBSI_NGLDM_DCENE)
{
	ASSERT_NO_THROW(test_ibsi_NGLDM_DCENE());
}


//***** 2D intensity ***** 

TEST(TEST_NYXUS, TEST_FIRSTORDER_MEAN_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_mean_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_SKEWNESS_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_skewness_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_KURTOSIS_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_kurtosis_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_MEDIAN_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_median_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_MINIMUM_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_minimum_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_P10_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_p10_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_P90_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_p90_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_INTERQUARTILE_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_interquartile_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_RANGE_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_range_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_MEAN_ABSOLUTE_DEVIATION_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_mean_absolute_deviation_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_ENERGY_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_energy_ibsi());
}

TEST(TEST_NYXUS, TEST_FIRSTORDER_ROOT_MEAN_SQUARED_IBSI) 
{
	ASSERT_NO_THROW(test_firstorder_root_mean_squared_ibsi());
}


//***** 2D first-order vs MATLAB (oracle_coverage.csv: oracle=matlab, target_test=test_firstorder_matlab.h) *****

TEST(TEST_NYXUS, TEST_FIRSTORDER_HYPERSKEWNESS_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_hyperskewness_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_HYPERFLATNESS_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_hyperflatness_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_UNIFORMITY_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_uniformity_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_UNIFORMITY_PIU_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_uniformity_piu_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_COVERED_IMAGE_INTENSITY_RANGE_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_covered_image_intensity_range_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_ROBUST_MEAN_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_robust_mean_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_INTEGRATED_INTENSITY_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_integrated_intensity_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_MIN_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_min_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_MAX_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_max_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_RANGE_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_range_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_MEAN_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_mean_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_MEDIAN_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_median_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_MODE_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_mode_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_STANDARD_DEVIATION_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_standard_deviation_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_SKEWNESS_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_skewness_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_EXCESS_KURTOSIS_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_excess_kurtosis_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_KURTOSIS_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_kurtosis_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_MEAN_ABSOLUTE_DEVIATION_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_mean_absolute_deviation_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_STANDARD_ERROR_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_standard_error_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_ROOT_MEAN_SQUARED_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_root_mean_squared_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_ENERGY_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_energy_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_P01_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_p01_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_P10_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_p10_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_P25_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_p25_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_P75_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_p75_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_P90_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_p90_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_P99_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_p99_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_INTERQUARTILE_RANGE_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_interquartile_range_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_COV_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_cov_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_MEDIAN_ABSOLUTE_DEVIATION_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_median_absolute_deviation_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_QCOD_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_qcod_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_STANDARD_DEVIATION_BIASED_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_standard_deviation_biased_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_VARIANCE_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_variance_matlab());
}
TEST(TEST_NYXUS, TEST_FIRSTORDER_VARIANCE_BIASED_MATLAB)
{
	ASSERT_NO_THROW(test_firstorder_variance_biased_matlab());
}

//***** 2D GLDM regression ***** 

TEST(TEST_NYXUS, TEST_GLDM_SDE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_sde_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_LDE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_lde_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_LGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_lgle_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_HGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_hgle_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_SDLGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_sdlgle_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_SDHGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_sdhgle_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_LDLGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_ldlgle_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_LDHGLE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_ldhgle_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_GLN_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_gln_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_DN_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_dn_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_DNN_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_dnn_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_GLV_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_glv_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_DV_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_dv_regression());
}

TEST(TEST_NYXUS, TEST_GLDM_DE_REGRESSION) 
{
	ASSERT_NO_THROW(test_gldm_de_regression());
}

// Bug #14b: background inside a concave ROI's bounding box must not enter the dependence matrix
TEST(TEST_NYXUS, TEST_GLDM_BUG_BACKGROUND_EXCLUDED_MECHANICS)
{
	ASSERT_NO_THROW(test_gldm_bug_background_excluded_mechanics());
}


//***** 2D GLRLM regression ***** 

TEST(TEST_NYXUS, TEST_GLRLM_SRE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_sre_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lre_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LGLRE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lglre_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_HGLRE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_hglre_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRLGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_srlgle_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRHGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_srhgle_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRLGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lrlgle_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRHGLE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lrhgle_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLN_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_gln_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLNN_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_glnn_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLN_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rln_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLNN_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rlnn_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RP_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rp_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLV_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_glv_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RV_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rv_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_re_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_sre_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lre_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_gln_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLNN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_glnn_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rln_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLNN_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rlnn_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RP_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rp_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLV_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_glv_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RV_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_rv_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_RE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_re_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LGLRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lglre_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_HGLRE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_hglre_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRLGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_srlgle_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRHGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_srhgle_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRLGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lrlgle_ave_regression());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRHGLE_AVE_REGRESSION)
{
	ASSERT_NO_THROW(test_glrlm_lrhgle_ave_regression());
}


//***** 2D GLDZM regression ***** 

TEST(TEST_NYXUS, TEST_IBSI_GLDZM_MATRIX_CORRECTNESS)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_matrix_correctness());
}

TEST(TEST_NYXUS, TEST_GLDZM_SDE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_SDE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_LDE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_LDE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_LGLZE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_LGLZE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_HGLZE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_HGLZE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_SDLGLE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_SDLGLE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_SDHGLE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_SDHGLE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_LDLGLE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_LDLGLE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_LDHGLE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_LDHGLE());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_GLNU)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_GLNU());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_GLNUN)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_GLNUN());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_ZDNU)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_ZDNU());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_ZDNUN)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_ZDNUN());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_ZP)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_ZP());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_GLM)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_GLM());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_GLV)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_GLV());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_ZDM)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_ZDM());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_ZDV)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_ZDV());
}

TEST(TEST_NYXUS, TEST_GLDZM_MATRIX_ZDE)
{
	ASSERT_NO_THROW(test_ibsi_GLDZM_ZDE());
}


//***** 2D GLSZM regression ***** 

TEST(TEST_NYXUS, TEST_GLSZM_SAE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_sae_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_LAE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_lae_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_LGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_lglze_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_HGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_hglze_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_SALGLE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_salgle_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_SAHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_sahgle_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_LALGLE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_lalgle_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_LAHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_lahgle_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_GLN_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_gln_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_GLNN_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_glnn_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_SZN_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_szn_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_SZNN_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_sznn_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZP_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_zp_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_GLV_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_glv_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZV_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_zv_regression());
}

TEST(TEST_NYXUS, TEST_GLSZM_ZE_REGRESSION) {
	ASSERT_NO_THROW(test_glszm_ze_regression());
}


//***** 2D NGTDM regression ***** 

TEST(TEST_NYXUS, TEST_NGTDM_COARSENESS_REGRESSION) 
{
	ASSERT_NO_THROW(test_ngtdm_coarseness_regression());
}

TEST(TEST_NYXUS, TEST_NGTDM_CONTRAST_REGRESSION) 
{
	ASSERT_NO_THROW(test_ngtdm_contrast_regression());
}

TEST(TEST_NYXUS, TEST_NGTDM_BUSYNESS_REGRESSION) 
{
	ASSERT_NO_THROW(test_ngtdm_busyness_regression());
}

TEST(TEST_NYXUS, TEST_NGTDM_COMPLEXITY_REGRESSION) 
{
	ASSERT_NO_THROW(test_ngtdm_complexity_regression());
}

TEST(TEST_NYXUS, TEST_NGTDM_STRENGTH_REGRESSION) 
{
	ASSERT_NO_THROW(test_ngtdm_strength_regression());
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

TEST(TEST_NYXUS, TEST_3D_NIFTY_LOADER) {
	ASSERT_NO_THROW (test_3d_nifti_loader());
}

TEST(TEST_NYXUS, TEST_3D_NIFTY_DACC_CONSISTENCY) {
	ASSERT_NO_THROW (test_3d_nifti_data_access_consistency());
}


//***** OME-Zarr i/o *****

#ifdef OMEZARR_SUPPORT

TEST(TEST_NYXUS, TEST_OMEZARR_TILELOADER_GEOMETRY) {
	ASSERT_NO_THROW (test_omezarr_tileloader_geometry());
}

TEST(TEST_NYXUS, TEST_OMEZARR_TILELOADER_CONTENT) {
	ASSERT_NO_THROW (test_omezarr_tileloader_content());
}

TEST(TEST_NYXUS, TEST_OMEZARR_TILELOADER_MULTITILE) {
	ASSERT_NO_THROW (test_omezarr_tileloader_multitile());
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_GEOMETRY) {
	ASSERT_NO_THROW (test_raw_omezarr_geometry());
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_CONTENT) {
	ASSERT_NO_THROW (test_raw_omezarr_content());
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_MULTITILE) {
	ASSERT_NO_THROW (test_raw_omezarr_multitile());
}

#endif // OMEZARR_SUPPORT


int main(int argc, char **argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
