#define NOMINMAX	// keep Windows min/max macros from breaking dcmtk's OFvariant (DICOM tests)
#include <gtest/gtest.h>
#include "test_gabor_regression.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "test_contour.h"
#include "test_ome_meta.h"		// native-OME metadata parsers / OmeAxes descriptor
#include "test_ometiff_mechanics.h"	// OME-TIFF native (z,c,t)->IFD read (core; no USE_Z5)
#include "test_firstorder_regression.h"
#include "test_intensity_histogram_regression.h"
#include "test_intensity_histogram_ibsi.h"
#include "test_hu_analytic.h"
#include "test_hu_mechanics.h"
#include "test_morphology_features.h"
#include "test_morphology_regression.h"
#include "test_morphology_skimage.h"
#include "test_morphology_matlab.h"
#include "test_morphology_fraclac.h"
#include "test_moments_skimage.h"
#include "test_moments_regression.h"
#include "test_remaining2d_common.h"
#include "test_zernike_regression.h"
#include "test_neighbor_regression.h"
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

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3P10) {
	ASSERT_NO_THROW(test_compat_radiomics_3fo_feature(Nyxus::Feature3D::P10, "3P10"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3P90) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::P90, "3P90"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3ENERGY) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::ENERGY, "3ENERGY"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3ENTROPY) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::ENTROPY, "3ENTROPY"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3INTERQUARTILE_RANGE) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::INTERQUARTILE_RANGE, "3INTERQUARTILE_RANGE"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3KURTOSIS) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::KURTOSIS, "3KURTOSIS"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3MAX) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::MAX, "3MAX"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3MEAN_ABSOLUTE_DEVIATION) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::MEAN_ABSOLUTE_DEVIATION, "3MEAN_ABSOLUTE_DEVIATION"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3MEAN) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::MEAN, "3MEAN"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3MEDIAN) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::MEDIAN, "3MEDIAN"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3MIN) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::MIN, "3MIN"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3RANGE) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::RANGE, "3RANGE"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3ROBUST_MEAN_ABSOLUTE_DEVIATION) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION, "3ROBUST_MEAN_ABSOLUTE_DEVIATION"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3ROOT_MEAN_SQUARED) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::ROOT_MEAN_SQUARED, "3ROOT_MEAN_SQUARED"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3SKEWNESS) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::SKEWNESS, "3SKEWNESS"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3UNIFORMITY) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::UNIFORMITY, "3UNIFORMITY"));
}

TEST(TEST_NYXUS, TEST_COMPAT_3FO_RADIOMICS_3VARIANCE) {
	ASSERT_NO_THROW (test_compat_radiomics_3fo_feature(Nyxus::Feature3D::VARIANCE, "3VARIANCE"));
}


//***** 3D NGTDM compatibility *****

TEST(TEST_NYXUS, TEST_COMPAT_3NGTDM_BUSYNESS) {
	ASSERT_NO_THROW(test_compat_3NGTDM_BUSYNESS());
}

TEST(TEST_NYXUS, TEST_COMPAT_3NGTDM_COARSENESS) {
	ASSERT_NO_THROW(test_compat_3NGTDM_COARSENESS());
}

TEST(TEST_NYXUS, TEST_COMPAT_3NGTDM_COMPLEXITY) {
	ASSERT_NO_THROW(test_compat_3NGTDM_COMPLEXITY());
}

TEST(TEST_NYXUS, TEST_COMPAT_3NGTDM_CONTRAST) {
	ASSERT_NO_THROW(test_compat_3NGTDM_CONTRAST());
}

TEST(TEST_NYXUS, TEST_COMPAT_3NGTDM_STRENGTH) {
	ASSERT_NO_THROW(test_compat_3NGTDM_STRENGTH());
}

TEST(TEST_NYXUS, TEST_3NGTD_MATRIX_CORRECTNESS) {
	ASSERT_NO_THROW(test_ngtd_matrix_correctness());
}


//***** 3D GLRLM compatibility *****

TEST(TEST_NYXUS, TEST_COMPAT_3GLRL_MATRIX_CORRECTNESS) {
	ASSERT_NO_THROW(test_glrl_matrix_correctness());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_GLN) {
	ASSERT_NO_THROW(test_compat_3GLRLM_GLN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_GLNN) {
	ASSERT_NO_THROW(test_compat_3GLRLM_GLNN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_GLV) {
	ASSERT_NO_THROW(test_compat_3GLRLM_GLV());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_HGLRE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_HGLRE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_LRE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_LRE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_LRHGLE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_LRHGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_LRLGLE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_LRLGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_LGLRE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_LGLRE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_RE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_RE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_RLN) {
	ASSERT_NO_THROW(test_compat_3GLRLM_RLN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_RLNN) {
	ASSERT_NO_THROW(test_compat_3GLRLM_RLNN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_RP) {
	ASSERT_NO_THROW(test_compat_3GLRLM_RP());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_RV) {
	ASSERT_NO_THROW(test_compat_3GLRLM_RV());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_SRE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_SRE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_SRHGLE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_SRHGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_SRLGLE) {
	ASSERT_NO_THROW(test_compat_3GLRLM_SRLGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLRLM_AVE_FEATURES) {
	ASSERT_NO_THROW(test_compat_3glrlm_ave_features());
}


//***** 3D GLSZM compatibility *****

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZ_MATRIX_CORRECTNESS) {
	ASSERT_NO_THROW (test_glsz_matrix_correctness());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_SAE) {
	ASSERT_NO_THROW(test_compat_3glszm_sae());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_LAE) {
	ASSERT_NO_THROW(test_compat_3glszm_lae());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_LGLZE) {
	ASSERT_NO_THROW(test_compat_3glszm_lglze());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_HGLZE) {
	ASSERT_NO_THROW(test_compat_3glszm_hglze());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_SALGLZE) {
	ASSERT_NO_THROW(test_compat_3glszm_salgle());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_SAHGLZE) {
	ASSERT_NO_THROW(test_compat_3glszm_sahgle());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_LALGLE) {
	ASSERT_NO_THROW(test_compat_3glszm_lalgle());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_LAHGLE) {
	ASSERT_NO_THROW(test_compat_3glszm_lahgle());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_gln) {
	ASSERT_NO_THROW(test_compat_3glszm_gln());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_glnn) {
	ASSERT_NO_THROW(test_compat_3glszm_glnn());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_szn) {
	ASSERT_NO_THROW(test_compat_3glszm_szn());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_SZNN) {
	ASSERT_NO_THROW(test_compat_3glszm_sznn());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_ZP) {
	ASSERT_NO_THROW(test_compat_3glszm_zp());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_GLV) {
	ASSERT_NO_THROW(test_compat_3glszm_glv());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_ZV) {
	ASSERT_NO_THROW(test_compat_3glszm_zv());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLSZM_ZE) {
	ASSERT_NO_THROW(test_compat_3glszm_ze());
}


//***** 3D GLDM compatibility *****

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_DE) {
	ASSERT_NO_THROW (test_compat_3GLDM_DE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_DN) {
	ASSERT_NO_THROW (test_compat_3GLDM_DN()); 
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_DNN) {
	ASSERT_NO_THROW (test_compat_3GLDM_DNN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_DV) {
	ASSERT_NO_THROW (test_compat_3GLDM_DV());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_GLN) {
	ASSERT_NO_THROW (test_compat_3GLDM_GLN());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_GLV) {
	ASSERT_NO_THROW (test_compat_3GLDM_GLV());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_HGLE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_HGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_LDE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_LDE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_LDHGLE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_LDHGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_LDLGLE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_LDLGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_LGLE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_LGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_SDE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_SDE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_SDHGLE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_SDHGLE());
}

TEST(TEST_NYXUS, TEST_COMPAT_3GLDM_SDLGLE) { 
	ASSERT_NO_THROW (test_compat_3GLDM_SDLGLE());
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


//***** Gabor regression ***** 

TEST(TEST_NYXUS, TEST_UNVETTED_NO_DIRECT_ORACLE_GABOR){
    test_unvetted_no_direct_oracle_gabor();

    #ifdef USE_GPU
        test_unvetted_no_direct_oracle_gabor(true);
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

TEST(TEST_NYXUS, TEST_FIRSTORDER_PYRADIOMICS_ORACLE)
{
	ASSERT_NO_THROW(test_firstorder_pyradiomics_oracle());
}

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

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_COV)
{
	ASSERT_NO_THROW(test_pixel_intensity_cov());
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

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_STDDEV_BIASED)
{
	ASSERT_NO_THROW(test_pixel_intensity_standard_deviation_biased());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_VARIANCE)
{
	ASSERT_NO_THROW(test_pixel_intensity_variance());
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

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_VARIANCE_BIASED)
{
	ASSERT_NO_THROW(test_pixel_intensity_variance_biased());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_SKEWNESS) 
{
	ASSERT_NO_THROW(test_pixel_intensity_skewness());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_EXCESS_KURTOSIS)
{
	ASSERT_NO_THROW(test_pixel_intensity_kurtosis());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_KURTOSIS)
{
	ASSERT_NO_THROW(test_pixel_intensity_pearson_kurtosis());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_HYPERSKEWNESS)
{
	ASSERT_NO_THROW(test_pixel_intensity_verifiable_with_3p_builtin_oracle_hyperskewness());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_HYPERFLATNESS)
{
	ASSERT_NO_THROW(test_pixel_intensity_verifiable_with_3p_builtin_oracle_hyperflatness());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_MAD) 
{
	ASSERT_NO_THROW(test_pixel_intensity_mean_absolute_deviation());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_MEDIAN_ABSOLUTE_DEVIATION)
{
	ASSERT_NO_THROW(test_pixel_intensity_median_absolute_deviation());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_ROBUST_MEAN)
{
	ASSERT_NO_THROW(test_pixel_intensity_verifiable_with_3p_builtin_oracle_robust_mean());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_ROBUST_MAD)
{
	ASSERT_NO_THROW(test_pixel_intensity_robust_mean_absolute_deviation());
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

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_UNIFORMITY_PIU)
{
	ASSERT_NO_THROW(test_pixel_intensity_verifiable_with_3p_builtin_oracle_uniformity_piu());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_PERCENTILES_IQR)
{
	ASSERT_NO_THROW(test_pixel_intensity_percentiles_iqr());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_QCOD)
{
	ASSERT_NO_THROW(test_pixel_intensity_qcod());
}

TEST(TEST_NYXUS, TEST_PIXEL_INTENSITY_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_COVERED_IMAGE_INTENSITY_RANGE)
{
	ASSERT_NO_THROW(test_pixel_intensity_verifiable_with_3p_builtin_oracle_covered_image_intensity_range());
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

TEST(TEST_NYXUS, TEST_SHAPE2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_CONTOUR_DIAMETER_EQUAL_PERIMETER)
{
	ASSERT_NO_THROW(test_shape2d_verifiable_with_3p_builtin_oracle_contour_diameter_equal_perimeter());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_CONVEX_HULL_FEATURES)
{
	ASSERT_NO_THROW(test_shape2d_convex_hull_features());
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

TEST(TEST_NYXUS, TEST_2D_SHAPE_GEOMETRIC_MOMENTS_VERIFIABLE_WITH_3P_BUILTIN_ORACLE)
{
	ASSERT_NO_THROW(test_2d_shape_geometric_moments_verifiable_with_3p_builtin_oracle());
}

TEST(TEST_NYXUS, TEST_MOMENTS_HU_WEDGE_SKIMAGE)
{
	ASSERT_NO_THROW(test_moments_hu_wedge_skimage());
}

TEST(TEST_NYXUS, TEST_2D_SHAPE_GEOMETRIC_MOMENTS_UNVETTED_NO_DIRECT_ORACLE)
{
	ASSERT_NO_THROW(test_2d_shape_geometric_moments_unvetted_no_direct_oracle());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_GEOMETRIC_MOMENTS_VERIFIABLE_WITH_3P_BUILTIN_ORACLE)
{
	ASSERT_NO_THROW(test_2d_intensity_geometric_moments_verifiable_with_3p_builtin_oracle());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_GEOMETRIC_MOMENTS_UNVETTED_NO_DIRECT_ORACLE)
{
	ASSERT_NO_THROW(test_2d_intensity_geometric_moments_unvetted_no_direct_oracle());
}

TEST(TEST_NYXUS, TEST_SHAPE2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_GEODETIC_THICKNESS_EROSION)
{
	ASSERT_NO_THROW(test_shape2d_verifiable_with_3p_builtin_oracle_geodetic_thickness_erosion_features());
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

TEST(TEST_NYXUS, TEST_SHAPE2D_DOCUMENTED_FORMULA_CONFORMANCE_NO_EXTERNAL_ORACLE)
{
	ASSERT_NO_THROW(test_shape2d_documented_formula_conformance_no_external_oracle());
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

TEST(TEST_NYXUS, TEST_REMAINING2D_VERIFIABLE_WITH_3P_BUILTIN_ORACLE_ZERNIKE2D)
{
	ASSERT_NO_THROW(test_remaining2d_verifiable_with_3p_builtin_oracle_zernike2d_feature());
}

TEST(TEST_NYXUS, TEST_NEIGHBORHOOD2D_COUNTS_TOUCHING)
{
	ASSERT_NO_THROW(test_neighborhood2d_counts_and_touching());
}

TEST(TEST_NYXUS, TEST_NEIGHBORHOOD2D_PERCENT_TOUCHING_ENCLOSED_ANALYTIC)
{
	ASSERT_NO_THROW(test_neighborhood2d_percent_touching_enclosed_analytic());
}

TEST(TEST_NYXUS, TEST_NEIGHBORHOOD2D_CLOSEST_NEIGHBORS)
{
	ASSERT_NO_THROW(test_neighborhood2d_closest_neighbors());
}

TEST(TEST_NYXUS, TEST_NEIGHBORHOOD2D_UNVETTED_NO_DIRECT_ORACLE_CLOSEST_NEIGHBOR_ANGLES)
{
	ASSERT_NO_THROW(test_neighborhood2d_unvetted_no_direct_oracle_closest_neighbor_angles());
}

TEST(TEST_NYXUS, TEST_NEIGHBORHOOD2D_UNVETTED_NO_DIRECT_ORACLE_ANGLE_STATS)
{
	ASSERT_NO_THROW(test_neighborhood2d_unvetted_no_direct_oracle_neighbor_angle_stats());
}


//***** IBSI tests of NGTDM

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


//***** IBSI tests of GLRLM ***** 

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

TEST(TEST_NYXUS, TEST_IBSI_GLRLM_LGLRE_AVE)  { ASSERT_NO_THROW(test_ibsi_glrlm_lglre_ave()); }
TEST(TEST_NYXUS, TEST_IBSI_GLRLM_HGLRE_AVE)  { ASSERT_NO_THROW(test_ibsi_glrlm_hglre_ave()); }
TEST(TEST_NYXUS, TEST_IBSI_GLRLM_SRLGLE_AVE) { ASSERT_NO_THROW(test_ibsi_glrlm_srlgle_ave()); }
TEST(TEST_NYXUS, TEST_IBSI_GLRLM_SRHGLE_AVE) { ASSERT_NO_THROW(test_ibsi_glrlm_srhgle_ave()); }
TEST(TEST_NYXUS, TEST_IBSI_GLRLM_LRLGLE_AVE) { ASSERT_NO_THROW(test_ibsi_glrlm_lrlgle_ave()); }
TEST(TEST_NYXUS, TEST_IBSI_GLRLM_LRHGLE_AVE) { ASSERT_NO_THROW(test_ibsi_glrlm_lrhgle_ave()); }

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


//***** IBSI tests of GLSZM ***** 

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


//***** 2D GLDM regression ***** 

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

// Bug #14b: background inside a concave ROI's bounding box must not enter the dependence matrix
TEST(TEST_NYXUS, TEST_GLDM_BUG_BACKGROUND_EXCLUDED)
{
	ASSERT_NO_THROW(test_gldm_bug_background_excluded());
}


//***** 2D GLRLM regression ***** 

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

TEST(TEST_NYXUS, TEST_GLRLM_SRE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_sre_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_lre_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLN_AVE)
{
	ASSERT_NO_THROW(test_glrlm_gln_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLNN_AVE)
{
	ASSERT_NO_THROW(test_glrlm_glnn_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLN_AVE)
{
	ASSERT_NO_THROW(test_glrlm_rln_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_RLNN_AVE)
{
	ASSERT_NO_THROW(test_glrlm_rlnn_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_RP_AVE)
{
	ASSERT_NO_THROW(test_glrlm_rp_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_GLV_AVE)
{
	ASSERT_NO_THROW(test_glrlm_glv_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_RV_AVE)
{
	ASSERT_NO_THROW(test_glrlm_rv_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_RE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_re_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_LGLRE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_lglre_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_HGLRE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_hglre_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRLGLE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_srlgle_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_SRHGLE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_srhgle_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRLGLE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_lrlgle_ave());
}

TEST(TEST_NYXUS, TEST_GLRLM_LRHGLE_AVE)
{
	ASSERT_NO_THROW(test_glrlm_lrhgle_ave());
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


//***** 2D NGTDM regression ***** 

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

TEST(TEST_IMAGE_QUALITY, TEST_FOCUS_SCORE) 
{
	ASSERT_NO_THROW(test_focus_score_feature());
}

TEST(TEST_IMAGE_QUALITY, TEST_LOCAL_FOCUS_SCORE) 
{
	ASSERT_NO_THROW(test_local_focus_score_feature());
}

TEST(TEST_IMAGE_QUALITY, TEST_POWER_SPECTRUM) 
{
	ASSERT_NO_THROW(test_power_spectrum_feature());
}

TEST(TEST_IMAGE_QUALITY, TEST_MIN_SATURATION) 
{
	ASSERT_NO_THROW(test_min_saturation_feature());
}

TEST(TEST_IMAGE_QUALITY, TEST_MAX_SATURATION) 
{
	ASSERT_NO_THROW(test_max_saturation_feature());
}

TEST(TEST_IMAGE_QUALITY, TEST_SHARPNESS) 
{
	ASSERT_NO_THROW(test_sharpness_feature());
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

TEST(TEST_NYXUS, TEST_OMEZARR_5D_CHANNEL_TIME_ADDRESSING) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim5.ome.zarr", 2, 3, 4));
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_5D_CHANNEL_TIME_ADDRESSING) {
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim5.ome.zarr", 2, 3, 4));
}

// All 6 legal orderings of {t,c,z}: passes only if the loader honors the NGFF
// 'axes' metadata instead of assuming a fixed [T,C,Z,Y,X] order.
TEST(TEST_NYXUS, TEST_OMEZARR_ALL_5D_PERMUTATIONS) {
	ASSERT_NO_THROW (test_omezarr_all_5d_permutations());
}

// 4D (rank-4): time-only and channel-only.
TEST(TEST_NYXUS, TEST_OMEZARR_4D_TZYX) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim4_tzyx.ome.zarr", 2, 1, 4));
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim4_tzyx.ome.zarr", 2, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_4D_CZYX) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim4_czyx.ome.zarr", 1, 3, 4));
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim4_czyx.ome.zarr", 1, 3, 4));
}

// Lower-rank stores: a fixed shape[2..4] loader would crash; the axis-role loader
// reads 3D (ZYX) and 2D (YX) correctly.
TEST(TEST_NYXUS, TEST_OMEZARR_3D_ZYX) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim3_zyx.ome.zarr", 1, 1, 4));
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_3D_ZYX) {
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim3_zyx.ome.zarr", 1, 1, 4));
}

TEST(TEST_NYXUS, TEST_OMEZARR_2D_YX) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim2_yx.ome.zarr", 1, 1, 1));
}

// No 'axes' metadata -> the loader falls back to legacy 5D TCZYX and still reads.
TEST(TEST_NYXUS, TEST_OMEZARR_NOAXES_FALLBACK) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim5_noaxes.ome.zarr", 2, 3, 4));
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_NOAXES_FALLBACK) {
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim5_noaxes.ome.zarr", 2, 3, 4));
}

// Negative: out-of-range Z/C/T plane index must throw.
TEST(TEST_NYXUS, TEST_OMEZARR_OUT_OF_RANGE_THROWS) {
	ASSERT_NO_THROW (test_omezarr_out_of_range_throws());
}

// Illegal / adversarial: malformed metadata must be rejected cleanly, not crash.
TEST(TEST_NYXUS, TEST_OMEZARR_MALFORMED_THROWS) {
	ASSERT_NO_THROW (test_omezarr_malformed_throws());
}

#endif // OMEZARR_SUPPORT


//***** OME-TIFF native (z,c,t) -> IFD read (core; runs in every build) *****

TEST(TEST_NYXUS, TEST_OMETIFF_5D_CHANNEL_TIME_ADDRESSING) {
	ASSERT_NO_THROW (test_ometiff_addressing("dim5.ome.tif", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_RAW_OMETIFF_5D_CHANNEL_TIME_ADDRESSING) {
	ASSERT_NO_THROW (test_raw_ometiff_addressing("dim5.ome.tif", 2, 3, 4));
}

// All 6 legal DimensionOrder values: passes only if ifdForPlane honors DimensionOrder.
TEST(TEST_NYXUS, TEST_OMETIFF_ALL_5D_PERMUTATIONS) {
	ASSERT_NO_THROW (test_ometiff_all_5d_permutations());
}

// 4D (rank-4): time-only and channel-only.
TEST(TEST_NYXUS, TEST_OMETIFF_4D_TZYX) {
	ASSERT_NO_THROW (test_ometiff_addressing("dim4_tzyx.ome.tif", 2, 1, 4));
	ASSERT_NO_THROW (test_raw_ometiff_addressing("dim4_tzyx.ome.tif", 2, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMETIFF_4D_CZYX) {
	ASSERT_NO_THROW (test_ometiff_addressing("dim4_czyx.ome.tif", 1, 3, 4));
	ASSERT_NO_THROW (test_raw_ometiff_addressing("dim4_czyx.ome.tif", 1, 3, 4));
}

// Lower-rank OME-TIFF.
TEST(TEST_NYXUS, TEST_OMETIFF_3D_ZYX) {
	ASSERT_NO_THROW (test_ometiff_addressing("dim3_zyx.ome.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_RAW_OMETIFF_3D_ZYX) {
	ASSERT_NO_THROW (test_raw_ometiff_addressing("dim3_zyx.ome.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMETIFF_2D_YX) {
	ASSERT_NO_THROW (test_ometiff_addressing("dim2_yx.ome.tif", 1, 1, 1));
}

// Plain multi-page TIFF (no OME-XML): the legacy page=Z fallback must still work.
TEST(TEST_NYXUS, TEST_OMETIFF_PLAIN_MULTIPAGE_FALLBACK) {
	ASSERT_NO_THROW (test_ometiff_addressing("dim3_plain.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_RAW_OMETIFF_PLAIN_MULTIPAGE_FALLBACK) {
	ASSERT_NO_THROW (test_raw_ometiff_addressing("dim3_plain.tif", 1, 1, 4));
}

// Negative: out-of-range Z/C/T plane index must throw.
TEST(TEST_NYXUS, TEST_OMETIFF_OUT_OF_RANGE_THROWS) {
	ASSERT_NO_THROW (test_ometiff_out_of_range_throws());
}

// Illegal / adversarial: RGB / corrupt / missing files must be rejected cleanly.
TEST(TEST_NYXUS, TEST_OMETIFF_MALFORMED_THROWS) {
	ASSERT_NO_THROW (test_ometiff_malformed_throws());
}


int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
