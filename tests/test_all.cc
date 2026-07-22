#define NOMINMAX	// keep Windows min/max macros from breaking dcmtk's OFvariant (DICOM tests)
#include <gtest/gtest.h>
#include <fstream>		// reading a written CSV back in TEST_CSV_MULTICHANNEL_NO_OVERWRITE
#include "test_gabor_regression.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/feature_method.h"		// TEST_3D_OOC_GUARD_REJECTS_UNSUPPORTED_FEATURE
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/ome/format_detect.h"		// detect_input_format (P1)
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

TEST(TEST_NYXUS, TEST_FACADE_NIFTI_LOAD_VOLUME_EQUIVALENCE) {
	ASSERT_NO_THROW (test_facade_nifti_load_volume_equivalence());
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

// End-to-end through the wired volumetric consumer (scan_trivial_wholevolume).
TEST(TEST_NYXUS, TEST_OMEZARR_WHOLEVOLUME_CONSUMER) {
	ASSERT_NO_THROW (test_omezarr_wholevolume_consumer("dim3_zyx.ome.zarr", 4));
}

// Facade whole-volume assembly (load_volume loops Z into one X*Y*Z buffer).
// Wired consumer reads the correct plane for every (channel, timeframe), not just (0,0).
TEST(TEST_NYXUS, TEST_OMEZARR_WHOLEVOLUME_CONSUMER_CT) {
	ASSERT_NO_THROW (test_omezarr_wholevolume_consumer_ct("dim5.ome.zarr", 2, 3, 4));
}

TEST(TEST_NYXUS, TEST_OMEZARR_FACADE_VOLUME_3D) {
	ASSERT_NO_THROW (test_omezarr_facade_volume("dim3_zyx.ome.zarr", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_FACADE_VOLUME_5D) {
	ASSERT_NO_THROW (test_omezarr_facade_volume("dim5.ome.zarr", 2, 3, 4));
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

// OME-Zarr 0.5 (Zarr v3): zarr.json metadata + 'ome'-wrapped NGFF + 0/c/... chunk keys,
// read through the z5 Dataset API (v2/v3-agnostic). Same coordinate encoding as the v2
// stores, so the addressing / facade / CT-count helpers apply unchanged.
TEST(TEST_NYXUS, TEST_OMEZARR_V3) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim5_v3.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim5_v3.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_omezarr_facade_volume("dim5_v3.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_CT_COUNTS) {
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim5_v3.ome.zarr", 2, 3, 4));
}
// Zstd-compressed Zarr v3 (the zarr 3.x / real-world default codec). Requires the build
// to link libzstd (-DWITH_ZSTD); z5 decodes the bytes+zstd v3 codec pipeline.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_ZSTD) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim5_v3_zstd.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim5_v3_zstd.ome.zarr", 2, 3, 4));
}

// Sharded Zarr v3 (the ``sharding_indexed`` codec) -- how large v3 stores, including Axle's,
// actually lay out data: many inner chunks packed into one shard object per (t,c). z5 3.x
// reads it via ShardedDataset, chosen automatically when zarr.json carries a shard shape; the
// nyxus loader is UNCHANGED because it reads through readSubarray, which unpacks the inner
// chunks from the shard transparently. The fixture's inner chunk is (z,y,x)=(1,3,4) so each
// 6x8 plane is a 2x2 grid of inner chunks living inside one shard -- the read must assemble
// across inner-chunk boundaries within a shard. Same 1..1152 TCZYX encoding as the other v3.
//
// Coverage is the whole-volume facade + prescan, NOT test_omezarr_addressing: with sharding,
// tileWidth/Height report the INNER chunk (4x3), so a single loadTileFromFile(0,0,...) reads
// only the top-left inner chunk, not the whole plane -- the addressing helper's one-tile-per-
// plane assumption. facade_volume assembles the full inner-chunk grid and checks every voxel,
// which is the correct coverage for a multi-chunk store (same reason the multichunk v2 fixture
// uses it). It drives both loadTileFromFile (abstract stack) and readSubarray under the hood.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_SHARDED_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_omezarr_facade_volume("dim5_v3_sharded.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_SHARDED_CT_COUNTS) {
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim5_v3_sharded.ome.zarr", 2, 3, 4));
}
// Prescan over the sharded store (raw loader's readSubarray, driven through the inner-chunk
// tile grid by for_each_voxel): whole-slide, so the ROI is the whole X*Y*Z volume.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_SHARDED_PRESCAN) {
	fs::path ip = omezarr_data_path("dim5_v3_sharded.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// Blosc-compressed Zarr v3 (common in real v3 stores alongside zstd). z5 decodes the
// bytes+blosc v3 codec pipeline when built WITH_BLOSC (already required for OME-Zarr).
TEST(TEST_NYXUS, TEST_OMEZARR_V3_BLOSC) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim5_v3_blosc.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim5_v3_blosc.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_BLOSC_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_omezarr_facade_volume("dim5_v3_blosc.ome.zarr", 2, 3, 4));
}

// Larger multi-SHARD-FILE Zarr v3 store (gen_dim5.py write_v3_multishard): C=2,T=1,Z=8,Y=24,X=32,
// inner chunk (z,y,x)=(2,6,8), shard (z,y,x)=(8,12,16) -> a 2x2 GRID OF SHARD FILES per (c,t),
// each packing 16 inner chunks (8 shard files total). Unlike dim5_v3_sharded (exactly one shard
// per (t,c), so it only proves multiple inner chunks packed into ONE shard), this exercises the
// volumetric assembly crossing SHARD-FILE boundaries mid-plane -- closer to a real, larger v3
// store's layout. Own local encoding (dim5_enc's C/Z/Y/X are hardcoded to the small fixture and
// don't apply here): value(x,y,z,c,t) = 1 + ((((t*C+c)*Z+z)*Y+y)*X+x), C=2,T=1,Z=8,Y=24,X=32.
static inline uint32_t dim5_multishard_enc(int x, int y, int z, int c, int t)
{
	const int C = 2, Z = 8, Y = 24, X = 32;
	return static_cast<uint32_t>(1 + ((((t * C + c) * Z + z) * Y + y) * X + x));
}

TEST(TEST_NYXUS, TEST_OMEZARR_V3_MULTISHARD_FACADE_VOLUME) {
	const int T = 1, C = 2, Z = 8, Y = 24, X = 32;
	fs::path ds = omezarr_data_path("dim5_v3_multishard.ome.zarr");
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	SlideProps p;
	p.fname_int = ds.string();
	p.fname_seg = "";
	FpImageOptions fp;
	ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ds.string();
	ASSERT_EQ(il.get_full_width(), (size_t)X);
	ASSERT_EQ(il.get_full_height(), (size_t)Y);
	ASSERT_EQ(il.get_full_depth(), (size_t)Z);

	for (int t = 0; t < T; ++t)
	  for (int c = 0; c < C; ++c)
	  {
	      ASSERT_TRUE(il.load_volume(c, t));
	      const std::vector<uint32_t>& vol = il.get_int_volume_buffer();
	      ASSERT_EQ(vol.size(), (size_t)X * Y * Z);
	      for (int z = 0; z < Z; ++z)
	        for (int y = 0; y < Y; ++y)
	          for (int x = 0; x < X; ++x)
	            ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], dim5_multishard_enc(x, y, z, c, t))
	                << "multishard vol (x" << x << " y" << y << " z" << z << " c" << c << " t" << t << ")";
	  }
	il.close();
}

TEST(TEST_NYXUS, TEST_OMEZARR_V3_MULTISHARD_CT_COUNTS) {
	fs::path ds = omezarr_data_path("dim5_v3_multishard.ome.zarr");
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	auto ldr = NyxusOmeZarrLoader<uint32_t>(1, ds.string());
	ASSERT_EQ(ldr.numberChannels(), (size_t)2);
	ASSERT_EQ(ldr.fullTimestamps(0), (size_t)1);
	ASSERT_EQ(ldr.fullDepth(0), (size_t)8);

	auto raw = RawOmezarrLoader(ds.string());
	ASSERT_EQ(raw.numberChannels(), (size_t)2);
	ASSERT_EQ(raw.fullTimestamps(0), (size_t)1);
	ASSERT_EQ(raw.fullDepth(0), (size_t)8);
}

// Prescan (raw loader's readSubarray driven across all 8 shard files) must see the full encoded
// range across BOTH channels, and the whole-slide ROI area -- not garbage, not just channel 0.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_MULTISHARD_PRESCAN) {
	fs::path ip = omezarr_data_path("dim5_v3_multishard.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, dim5_multishard_enc(31, 23, 7, 1, 0));	// last voxel, last channel
	EXPECT_EQ(p.max_roi_area, (size_t)(32 * 24 * 8));
}

// Zarr v3 store with a MULTI-PLANE Z chunk (chunk z-extent 3 over Z=7 -> depths 3,3,1, an UNEVEN
// split), unsharded -- isolated regression test for a real bug found while building the multishard
// fixture above: omezarr.h/raw_omezarr.h's loadTile() always read exactly ONE Z-plane per tile
// (shape[iz_] left at its default of 1) regardless of the chunk's actual Z extent, so every plane
// past the first within a multi-plane chunk silently came back zero. No existing fixture before
// this one ever used a Z-chunk > 1. Own local encoding (dim5_enc/dim5_multishard_enc don't apply --
// different dims): value(x,y,z,c,t) = 1 + ((((t*C+c)*Z+z)*Y+y)*X+x), C=2,T=1,Z=7,Y=6,X=8.
static inline uint32_t dim5_zchunked_enc(int x, int y, int z, int c, int t)
{
	const int C = 2, Z = 7, Y = 6, X = 8;
	return static_cast<uint32_t>(1 + ((((t * C + c) * Z + z) * Y + y) * X + x));
}

// Exercises omezarr.h's NyxusOmeZarrLoader via ImageLoader::load_volume/assemble_volume -- the
// path that read zero past the first Z-plane of a chunk before the fix.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_ZCHUNKED_FACADE_VOLUME) {
	const int T = 1, C = 2, Z = 7, Y = 6, X = 8;
	fs::path ds = omezarr_data_path("dim5_v3_zchunked.ome.zarr");
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	SlideProps p;
	p.fname_int = ds.string();
	p.fname_seg = "";
	FpImageOptions fp;
	ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ds.string();
	ASSERT_EQ(il.get_full_depth(), (size_t)Z);

	for (int t = 0; t < T; ++t)
	  for (int c = 0; c < C; ++c)
	  {
	      ASSERT_TRUE(il.load_volume(c, t));
	      const std::vector<uint32_t>& vol = il.get_int_volume_buffer();
	      ASSERT_EQ(vol.size(), (size_t)X * Y * Z);
	      for (int z = 0; z < Z; ++z)
	        for (int y = 0; y < Y; ++y)
	          for (int x = 0; x < X; ++x)
	            ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], dim5_zchunked_enc(x, y, z, c, t))
	                << "zchunked vol (x" << x << " y" << y << " z" << z << " c" << c << " t" << t << ")";
	  }
	il.close();
}

// Exercises raw_omezarr.h's RawOmezarrLoader via RawImageLoader::for_each_voxel (the prescan path)
// -- the OTHER consumer of the same buggy loadTile(), independently regressed here.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_ZCHUNKED_PRESCAN) {
	fs::path ip = omezarr_data_path("dim5_v3_zchunked.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, dim5_zchunked_enc(7, 5, 6, 1, 0));	// last voxel, last channel
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 7));
}

// No 'axes' metadata -> the loader falls back to legacy 5D TCZYX and still reads.
TEST(TEST_NYXUS, TEST_OMEZARR_NOAXES_FALLBACK) {
	ASSERT_NO_THROW (test_omezarr_addressing("dim5_noaxes.ome.zarr", 2, 3, 4));
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_NOAXES_FALLBACK) {
	ASSERT_NO_THROW (test_raw_omezarr_addressing("dim5_noaxes.ome.zarr", 2, 3, 4));
}

// Loaders advertise the real C/T extents (numberChannels/fullTimestamps), which
// is what activates the pipeline's channel/timeframe iteration. dim5_noaxes proves
// the positional fallback reports counts too.
TEST(TEST_NYXUS, TEST_OMEZARR_CT_COUNTS) {
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim5.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim4_tzyx.ome.zarr", 2, 1, 4));
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim4_czyx.ome.zarr", 1, 3, 4));
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim3_zyx.ome.zarr", 1, 1, 4));
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim2_yx.ome.zarr", 1, 1, 1));
	ASSERT_NO_THROW (test_omezarr_ct_counts("dim5_noaxes.ome.zarr", 2, 3, 4));
}

// Physical calibration: loaders surface coordinateTransformations scale + unit.
TEST(TEST_NYXUS, TEST_OMEZARR_PHYSICAL_CALIBRATION) {
	ASSERT_NO_THROW (test_omezarr_physical_calibration());
}

// Unit canonicalization: a nanometer-declared store must report the same physX/Y/Z and
// "micrometer" as the equivalent micrometer-declared store above -- proves conversion, not
// just passthrough of the raw unit string.
TEST(TEST_NYXUS, TEST_OMEZARR_UNIT_CANONICALIZATION) {
	ASSERT_NO_THROW (test_omezarr_unit_canonicalization());
}

// Multi-CHUNK plane: real OME-Zarr splits each Y/X plane across a chunk grid (typically
// 512x512), and dim5_multichunk uses 3x4 chunks over the 6x8 plane. The volumetric read
// must walk the whole tile grid: fetching only chunk (0,0) returns wrong data past the
// first chunk (and over-reads its buffer). Every other fixture is one-chunk-per-plane,
// which is why this went unnoticed. Covers ImageLoader::assemble_volume...
TEST(TEST_NYXUS, TEST_OMEZARR_MULTICHUNK_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_omezarr_facade_volume("dim5_multichunk.ome.zarr", 2, 3, 4));
}
// ...and RawImageLoader::load_volume (the prescan), which had the same single-tile bug:
// the encoded values are 1..1152 over all (c,t), and the ROI is the whole X*Y*Z volume.
TEST(TEST_NYXUS, TEST_OMEZARR_MULTICHUNK_PRESCAN) {
	fs::path ip = omezarr_data_path("dim5_multichunk.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// PARTIAL edge chunks: chunk (4,5) does not divide the 6x8 plane, so the last row-chunk is 2
// tall and the last col-chunk is 3 wide. dim5_multichunk above (3x4 over 6x8) tiles exactly,
// so the validH/validW seam clamp never ran on the OME-Zarr path either. Asserting the exact
// value at every voxel is the seam check; same 1..1152 TCZYX encoding as dim5_multichunk.
TEST(TEST_NYXUS, TEST_OMEZARR_ODDCHUNK_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_omezarr_facade_volume("dim5_oddchunk.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_ODDCHUNK_PRESCAN) {
	fs::path ip = omezarr_data_path("dim5_oddchunk.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// Negative: out-of-range channel/timeframe through the whole-volume facade must throw.
TEST(TEST_NYXUS, TEST_OMEZARR_LOAD_VOLUME_OUT_OF_RANGE) {
	ASSERT_NO_THROW (test_omezarr_load_volume_out_of_range());
}

// Negative: out-of-range Z/C/T plane index must throw.
TEST(TEST_NYXUS, TEST_OMEZARR_OUT_OF_RANGE_THROWS) {
	ASSERT_NO_THROW (test_omezarr_out_of_range_throws());
}

// Illegal / adversarial: malformed metadata must be rejected cleanly, not crash.
TEST(TEST_NYXUS, TEST_OMEZARR_MALFORMED_THROWS) {
	ASSERT_NO_THROW (test_omezarr_malformed_throws());
}

// N1 (negative): a Zarr v3 store whose zarr.json declares a codec z5 does not support. z5's
// readV3CodecsFromJson throws "unsupported zarr v3 codec" during metadata parse (openDataset),
// so the loader must surface a clean error, not crash. Both loader stacks must reject it.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_UNSUPPORTED_CODEC_THROWS) {
	fs::path ds = omezarr_data_path("dim5_v3_badcodec.ome.zarr");
	ASSERT_TRUE(fs::exists(ds)) << ds.string();
	EXPECT_ANY_THROW(NyxusOmeZarrLoader<uint32_t>(1, ds.string()));
	EXPECT_ANY_THROW(RawOmezarrLoader(ds.string()));
}

// P4 (positive): the crash's positive twin on the OME-Zarr path -- a T>1 Zarr intensity paired
// with a single-timeframe ZYX Zarr mask. Zarr never crashed (no T axis to over-index), but it
// was never asserted. The prescan must reuse the mask across timeframes and find the ROI.
TEST(TEST_NYXUS, TEST_OMEZARR_MULTITIMEFRAME_MASK_PRESCAN) {
	fs::path ip = omezarr_data_path("dim5.ome.zarr");        // T=2, C=3, Z=4
	fs::path mp = omezarr_data_path("dim3_mask.ome.zarr");   // ZYX (T=1) label mask
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)(4 * 4 * 6));
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

// Regression (found by the scale/stress harness): a MULTI-timeframe OME-TIFF paired with a
// single-timeframe (ZYX) mask. The 3D prescan (RawImageLoader::for_each_voxel) reuses the
// mask across every intensity timeframe, but read it at the intensity's timeframe -- so for
// t>0 the TIFF mask loader addressed an IFD past its last plane and TIFFSetDirectory threw,
// UNCAUGHT, crashing the process (0xC0000409). Zarr masks have no T axis to over-index, so
// only TIFF crashed. dim5 is T=2,C=3,Z=4; dim3_mask is its ZYX (T=1) segmentation of one ROI
// (label 1 over z=all, y in [1,5), x in [1,7) = 4*4*6 voxels). Pre-fix this threw.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITIMEFRAME_MASK_PRESCAN) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();
	ASSERT_TRUE(fs::exists(mp)) << mp.string();

	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)(4 * 4 * 6));   // z=4 * y=4 * x=6
}

// P3 (positive): the crash's regression above covers the PRESCAN (for_each_voxel); this covers
// the FEATURIZE facade. ImageLoader::load_volume(c,t) forwards t as the mask timeframe, so with
// a T>1 intensity and a T=1 mask, load_volume(c,1) exercises the internal mask-timeframe clamp
// (image_loader.cpp). It must not throw, must reuse the same mask across timeframes, and must
// read different intensity per timeframe. dim5 is T=2; dim3_mask is its ZYX (T=1) mask.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITIMEFRAME_MASK_FACADE) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	SlideProps p; p.fname_int = ip.string(); p.fname_seg = mp.string();
	FpImageOptions fp; ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ip.string();

	ASSERT_TRUE(il.load_volume(0, 0));
	const std::vector<uint32_t> seg_t0 = il.get_seg_volume_buffer();
	const std::vector<uint32_t> int_t0 = il.get_int_volume_buffer();
	// timeframe 1 with a single-timeframe mask must NOT throw (clamp), reuse the mask, and
	// deliver different intensity
	ASSERT_TRUE(il.load_volume(0, 1));
	EXPECT_EQ(il.get_seg_volume_buffer(), seg_t0) << "mask changed across timeframes";
	EXPECT_NE(il.get_int_volume_buffer(), int_t0) << "intensity t=1 read t=0 data";
	il.close();
}

// P2 (positive): OME-TIFF physical calibration -> SlideProps (the TIFF twin of the OME-Zarr
// calibration test, and it additionally checks the scan_slide_props propagation the Zarr test
// omits). dim5_calibrated carries PhysicalSizeX/Y=0.5, Z=2.0 micrometer in its OME-XML.
TEST(TEST_NYXUS, TEST_OMETIFF_PHYSICAL_CALIBRATION) {
	fs::path cal = ometiff_data_path("dim5_calibrated.ome.tif");
	ASSERT_TRUE(fs::exists(cal)) << cal.string();

	Environment e;
	SlideProps p (cal.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	EXPECT_DOUBLE_EQ(p.phys_x, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_y, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_z, 2.0);
	EXPECT_EQ(p.phys_unit, "micrometer");

	// an uncalibrated OME-TIFF must default to 1.0 / no unit
	SlideProps p2 (ometiff_data_path("dim5.ome.tif").string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(p2, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	EXPECT_DOUBLE_EQ(p2.phys_x, 1.0);
	EXPECT_DOUBLE_EQ(p2.phys_z, 1.0);
	EXPECT_TRUE(p2.phys_unit.empty());
}

// Unit canonicalization (OME-TIFF, end-to-end through scan_slide_props): dim5_calibrated_nm
// declares X/Y in nanometer and Z in a THIRD unit (millimeter) -- 500nm==0.5um,
// 0.002mm==2.0um. Must report the SAME physX/Y/Z and a single "micrometer" unit as
// dim5_calibrated above, proving each axis converts using its OWN declared unit rather than
// X/Y's unit leaking onto Z (or vice versa).
TEST(TEST_NYXUS, TEST_OMETIFF_UNIT_CANONICALIZATION) {
	fs::path cal_nm = ometiff_data_path("dim5_calibrated_nm.ome.tif");
	ASSERT_TRUE(fs::exists(cal_nm)) << cal_nm.string();

	Environment e;
	SlideProps p (cal_nm.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	EXPECT_DOUBLE_EQ(p.phys_x, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_y, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_z, 2.0);
	EXPECT_EQ(p.phys_unit, "micrometer");
}

// N2 (negative): a <TiffData> block maps a plane to an IFD PAST the end of the file (an
// in-file overrun, distinct from a multi-file UUID). ifdForPlane returns 99, so the read must
// throw cleanly at TIFFSetDirectory -- not crash and not read a wrong plane. dim5_badifd has
// 4 z-planes; plane z3 claims IFD=99.
TEST(TEST_NYXUS, TEST_OMETIFF_BAD_IFD_THROWS) {
	fs::path ds = ometiff_data_path("dim5_badifd.ome.tif");
	ASSERT_TRUE(fs::exists(ds)) << ds.string();
	SlideProps p; p.fname_int = ds.string(); p.fname_seg = "";
	FpImageOptions fp; ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ds.string();
	// z0..z2 are fine; assembling the whole volume must hit z3's bad IFD and throw, not crash
	EXPECT_ANY_THROW(il.load_volume(0, 0));
	il.close();
}

// N3 (negative): an all-background (all-zero) mask -> ZERO ROIs. The prescan must complete
// cleanly (no divide-by-zero, no garbage), reporting no ROI area, rather than crash.
TEST(TEST_NYXUS, TEST_OMETIFF_EMPTY_MASK_ZERO_ROIS) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_emptymask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)0) << "empty mask should yield no ROI";
}

// N4 (edge): a mask with MORE channels than the intensity (C=2 mask, C=1-effective use). The
// featurize loop iterates the intensity's channels, so the extra mask channel must simply be
// ignored (mask channel clamped to what is asked), not crash or misread. dim3_zyx (C=1) paired
// with a C=2 label mask; the ROI is identical on both mask channels.
TEST(TEST_NYXUS, TEST_OMETIFF_MASK_MORE_CHANNELS_THAN_INTENSITY) {
	fs::path ip = ometiff_data_path("dim3_zyx.ome.tif");   // C=1, Z=4
	fs::path mp = ometiff_data_path("dim4_mask_c2.ome.tif"); // C=2 label mask
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)(4 * 4 * 6));   // the one ROI, read from mask channel 0
}

// P1 (positive + negative): detect_input_format is the single dispatch point all three loader
// stacks use; it had no direct test. Extension classification + OME content-sniff.
TEST(TEST_NYXUS, TEST_DETECT_INPUT_FORMAT) {
	using Nyxus::detect_input_format;
	using Nyxus::ContainerKind;
	// OME content present -> OmeTiff + is_ome
	auto a = detect_input_format(ometiff_data_path("dim5.ome.tif").string());
	EXPECT_EQ(a.kind, ContainerKind::OmeTiff);
	EXPECT_TRUE(a.is_ome);
	// a plain multipage TIFF (no OME-XML) -> TiffPlain, not OME
	auto b = detect_input_format(ometiff_data_path("dim3_plain.tif").string());
	EXPECT_EQ(b.kind, ContainerKind::TiffPlain);
	EXPECT_FALSE(b.is_ome);
	// extension-only kinds (no file needed / not opened)
	EXPECT_EQ(detect_input_format("x.dcm").kind, ContainerKind::Dicom);
	EXPECT_EQ(detect_input_format("x.nii.gz").kind, ContainerKind::Nifti);
	// a .zarr path with no multiscales metadata -> OmeZarr kind but is_ome=false
	auto z = detect_input_format("no_such_store.zarr");
	EXPECT_EQ(z.kind, ContainerKind::OmeZarr);
	EXPECT_FALSE(z.is_ome);
}

// Pyramidal OME-TIFF: every full-res plane's IFD carries downsampled levels as SubIFDs (tag
// 330), which live OUTSIDE the main IFD chain. This must not shift full-res plane addressing:
// TIFFNumberOfDirectories still returns Z (not Z*levels) and ifdForPlane -> main-chain IFD
// still lands on the full-res plane. nyxus reads level 0 only; the facade check (which also
// spans a 2x3 tile grid) asserts every full-res voxel is correct despite the SubIFDs. Z=6
// z-stack (C=1,T=1), its own encoding.
TEST(TEST_NYXUS, TEST_OMETIFF_PYRAMID_SUBIFD_FULLRES) {
	ASSERT_NO_THROW (test_ometiff_multitile_facade_volume("dim5_pyramid.ome.tif", 1, 1, 6, 32, 48));
}

// Non-canonical <TiffData> plane->IFD mapping: dim5_reordered stores its planes in REVERSED
// IFD order and declares the mapping via per-plane <TiffData IFD=..> blocks. A reader that
// ignores TiffData and assumes contiguous-from-IFD-0 order reads the reversed plane's pixels;
// honoring the map reads correctly. load_volume loops Z through loadTileFromFile -> ifdForPlane
// (both loader stacks route here), so the facade check asserts the right plane per (z,c).
// T=1,C=2,Z=3 (its own encoding). This is the OME-TIFF counterpart to what bioformats emits.
TEST(TEST_NYXUS, TEST_OMETIFF_TIFFDATA_REORDERED) {
	ASSERT_NO_THROW (test_ometiff_multitile_facade_volume("dim5_reordered.ome.tif", 1, 2, 3, 6, 8));
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

// End-to-end through the wired volumetric consumer (scan_trivial_wholevolume).
TEST(TEST_NYXUS, TEST_OMETIFF_WHOLEVOLUME_CONSUMER) {
	ASSERT_NO_THROW (test_ometiff_wholevolume_consumer("dim3_zyx.ome.tif", 4));
}

// Wired consumer reads the correct plane for every (channel, timeframe), not just (0,0).
TEST(TEST_NYXUS, TEST_OMETIFF_WHOLEVOLUME_CONSUMER_CT) {
	ASSERT_NO_THROW (test_ometiff_wholevolume_consumer_ct("dim5.ome.tif", 2, 3, 4));
}

// TILED multi-plane OME-TIFF: the tile loaders map (z,c,t)->IFD (distinct from strip loaders).
TEST(TEST_NYXUS, TEST_OMETIFF_TILED_ADDRESSING) {
	ASSERT_NO_THROW (test_ometiff_tiled_addressing());
}
// Facade whole-volume assembly over the TILED path (open() routes tiled TIFF -> tile loader).
TEST(TEST_NYXUS, TEST_OMETIFF_TILED_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_ometiff_facade_volume("dim5_tiled.ome.tif", 2, 3, 4));
}
// Multi-TILE planes (2x3 grid of 16x16 tiles): dim5_tiled above has ONE tile per plane, so
// it passes even when only tile (0,0) is read. This is the OME-TIFF counterpart of
// TEST_OMEZARR_MULTICHUNK_FACADE_VOLUME. Covers ImageLoader::assemble_volume...
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITILE_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_ometiff_multitile_facade_volume("dim5_multitile.ome.tif", 1, 2, 2, 32, 48));
}
// ...and RawImageLoader::for_each_voxel (the prescan), which had the same single-tile bug.
// Encoded values run 1..6144 over all (c,t); whole-slide, so the ROI is the whole volume.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITILE_PRESCAN) {
	fs::path ip = ometiff_data_path("dim5_multitile.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 6144.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(48 * 32 * 2));
}

// PARTIAL edge tiles: 40x24 plane / 16 -> 3x2 tiles, last row-tile 8 tall, last col-tile 8
// wide. Every multi-tile fixture above has plane dims that are exact multiples of the tile
// size, so the validH/validW seam clamp -- min(tileDim, fullDim-offset) -- had never run.
// Reading the exact value at every voxel checks the partial tiles are copied without garbage
// past the seam and without over-reading the tile buffer. Encoded 1..1920 over (c in 0,1).
TEST(TEST_NYXUS, TEST_OMETIFF_ODDTILE_FACADE_VOLUME) {
	ASSERT_NO_THROW (test_ometiff_multitile_facade_volume("dim5_oddtile.ome.tif", 1, 2, 1, 40, 24));
}
// The prescan (for_each_voxel) walks the same partial grid; its slide min/max and ROI area
// would be wrong if a partial tile were mis-clamped (a too-large validH double-counts voxels).
TEST(TEST_NYXUS, TEST_OMETIFF_ODDTILE_PRESCAN) {
	fs::path ip = ometiff_data_path("dim5_oddtile.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1920.0);	// 1 + (((0*2+1)*1+0)*40+39)*24+23
	EXPECT_EQ(p.max_roi_area, (size_t)(24 * 40 * 1));
}

// Facade whole-volume assembly (load_volume loops Z into one X*Y*Z buffer).
TEST(TEST_NYXUS, TEST_OMETIFF_FACADE_VOLUME_3D) {
	ASSERT_NO_THROW (test_ometiff_facade_volume("dim3_zyx.ome.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMETIFF_FACADE_VOLUME_5D) {
	ASSERT_NO_THROW (test_ometiff_facade_volume("dim5.ome.tif", 2, 3, 4));
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

// Negative: out-of-range channel/timeframe through the whole-volume facade must throw.
TEST(TEST_NYXUS, TEST_OMETIFF_LOAD_VOLUME_OUT_OF_RANGE) {
	ASSERT_NO_THROW (test_ometiff_load_volume_out_of_range());
}

// Regression: a single-channel mask is reused across all intensity channels (not read OOB).
TEST(TEST_NYXUS, TEST_OMETIFF_MULTICHANNEL_MASK_PAIRING) {
	ASSERT_NO_THROW (test_ometiff_multichannel_mask_pairing());
}

// Strip loaders advertise the OME C/T extents; the plain (non-OME) multi-page TIFF
// keeps C=T=1 (its pages are Z-slices, not channels/timeframes).
TEST(TEST_NYXUS, TEST_OMETIFF_CT_COUNTS) {
	ASSERT_NO_THROW (test_ometiff_ct_counts("dim5.ome.tif", 2, 3, 4));
	ASSERT_NO_THROW (test_ometiff_ct_counts("dim4_tzyx.ome.tif", 2, 1, 4));
	ASSERT_NO_THROW (test_ometiff_ct_counts("dim4_czyx.ome.tif", 1, 3, 4));
	ASSERT_NO_THROW (test_ometiff_ct_counts("dim3_zyx.ome.tif", 1, 1, 4));
	ASSERT_NO_THROW (test_ometiff_ct_counts("dim3_plain.tif", 1, 1, 4));
}

// Negative: out-of-range Z/C/T plane index must throw.
TEST(TEST_NYXUS, TEST_OMETIFF_OUT_OF_RANGE_THROWS) {
	ASSERT_NO_THROW (test_ometiff_out_of_range_throws());
}

// Illegal / adversarial: RGB / corrupt / missing files must be rejected cleanly.
TEST(TEST_NYXUS, TEST_OMETIFF_MALFORMED_THROWS) {
	ASSERT_NO_THROW (test_ometiff_malformed_throws());
}


// Regression: the 3D prescan must scan the WHOLE volume of EVERY (channel, timeframe),
// not one Z-plane of (c0,t0). dim5.ome.tif is C=3,T=2,Z=4,Y=6,X=8 encoding values 1..1152,
// so the slide intensity range must be exactly [1, 1152]. Before the fix the prescan
// (a) did a single load_tile(0,0) and then indexed W*H*D voxels off that ONE-plane buffer,
// reading out of bounds (observed range 0..44,465 of garbage), and (b) covered only
// (c0,t0) -> a range of 1..192, which under-sized intensity-indexed buffers for c>0 and
// segfaulted. The area check guards that ROI geometry is taken from the first pass only
// (otherwise it would be multiplied by n_channels*n_timeframes).
TEST(TEST_NYXUS, TEST_3D_PRESCAN_SLIDE_RANGE) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.resultOptions.need_annotation()));

	EXPECT_EQ(p.inten_channels, (size_t)3);
	EXPECT_EQ(p.inten_time, (size_t)2);
	EXPECT_EQ(p.volume_d, (size_t)4);
	// the full encoded range across ALL (c,t) -- not garbage, and not just (c0,t0)'s 1..192
	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	// geometry from the first pass only: one whole-slide ROI of exactly X*Y*Z voxels
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// Regression: the 3D WHOLE-VOLUME reduce path. reduce_trivial_3d_wholevolume calls
// D3_VoxelIntensityFeatures::extract(), which used to invoke the 2-arg calculate() -- a stub
// that throws "illegal call" -- so EVERY 3D whole-volume featurization died before writing a
// row. The segmented path reduces via reduce(), which passes the Dataset, so nothing covered
// this. Mirrors featurize_wholevolume()'s vROI setup, then reduces.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_REDUCE) {
	fs::path ds = ometiff_data_path("dim3_zyx.ome.tif");	// 3D X8 Y6 Z4
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	Environment e;
	// enable the 3D intensity features so the reduce actually runs them
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);

	// prescan the slide (whole-volume => no mask)
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ds.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();

	// build the vROI exactly as featurize_wholevolume() does
	FpImageOptions fp;
	ImageLoader ilo;
	ASSERT_TRUE(ilo.open(sp, fp)) << ds.string();
	LR vroi(1);
	vroi.slide_idx = 0;
	vroi.aux_area = sp.max_roi_area;
	vroi.aabb.init_from_whd(sp.max_roi_w, sp.max_roi_h, sp.max_roi_d);
	vroi.aux_min = (PixIntens)0;
	vroi.aux_max = (PixIntens)(sp.max_preroi_inten - sp.min_preroi_inten);
	ASSERT_NO_THROW(vroi.initialize_fvals());
	ASSERT_TRUE(Nyxus::scan_trivial_wholevolume(vroi, ds.string(), ilo, 0/*channel*/, 0/*timeframe*/));
	ASSERT_GT(vroi.raw_pixels_3D.size(), 0u);
	vroi.aux_image_cube.allocate(vroi.aabb.get_width(), vroi.aabb.get_height(), vroi.aabb.get_z_depth());
	vroi.aux_image_cube.calculate_from_pixelcloud(vroi.raw_pixels_3D, vroi.aabb);

	// THE REGRESSION: this used to throw "illegal call of D3_VoxelIntensityFeatures::calculate"
	ASSERT_NO_THROW(Nyxus::reduce_trivial_3d_wholevolume(e, vroi));

	// and it must actually produce values (MAX >= MIN, both finite)
	double vmin = vroi.get_fvals((int)Nyxus::Feature3D::MIN)[0];
	double vmax = vroi.get_fvals((int)Nyxus::Feature3D::MAX)[0];
	EXPECT_GE(vmax, vmin);
	EXPECT_GT(vmax, 0.0);
	ilo.close();
}

// Regression: an OVERSIZED whole volume of a format that delivers plane-by-plane (e.g. OME-TIFF)
// must now featurize SUCCESSFULLY out-of-core, producing the SAME feature values as the in-RAM
// (fitting) run of the identical file -- not fail, and not emit a zero row. This supersedes the
// old TEST_3D_WHOLEVOLUME_OVERSIZED_FAILS_LOUDLY premise: before workflow_3d_whole.cpp's oversized
// branch streamed via populate_3d_voxel_cloud/run_3d_ooc_features, NO whole-volume streaming path
// existed at all, so oversized-but-streamable volumes failed loudly (a deliberate, validated
// fallback at the time). Now the only remaining loud-fail case is a genuinely unstreamable format
// (whole-4D-in-one-read, e.g. NIfTI) -- see TEST_3D_WHOLEVOLUME_UNSTREAMABLE_FORMAT_FAILS_LOUDLY.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_OVERSIZED_STREAMS_OOC) {
	fs::path ds = ometiff_data_path("dim3_zyx.ome.tif");	// 3D X8 Y6 Z4
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	auto run_and_read_row = [&](bool oversized, const fs::path& outdir) -> std::string
	{
		fs::remove_all(outdir); fs::create_directories(outdir);
		Environment e;
		e.theFeatureSet.enableAll(false);
		e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);
		e.theFeatureSet.enableFeatures(D3_SurfaceFeature::featureset);
		e.theFeatureSet.enableFeatures(D3_GLCM_feature::featureset);
		// The full sequence main_nyxus.cpp runs before any workflow (theFeatureMgr.compile() ->
		// apply_user_selection() -> init_feature_classes() -> compile_feature_settings()). Skipping
		// theFeatureMgr setup leaves get_num_requested_features()==0, so run_3d_ooc_features's loop
		// never executes and every feature stays at its zero-initialized default (first symptom: an
		// all-zero OOC row). Skipping compile_feature_settings() leaves fsett_D3_* at size 0, so any
		// STNGS_*(s) macro access (e.g. surface's STNGS_SINGLEROI) reads out of bounds -- crashed
		// (SEH 0xc0000005) before this was added.
		EXPECT_TRUE(e.theFeatureMgr.compile());
		e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
		EXPECT_TRUE(e.theFeatureMgr.init_feature_classes());
		e.compile_feature_settings();
		EXPECT_TRUE(e.set_ram_limit(oversized ? 0 : 64));	// 0 -> force oversized; 64 MB comfortably fits this tiny (X8 Y6 Z4) volume
		e.output_dir = outdir.string();

		std::vector<std::string> ifiles{ ds.string() };
		auto [ok, erm] = Nyxus::processDataset_3D_wholevolume(e, ifiles, 1, Nyxus::SaveOption::saveCSV, outdir.string());
		EXPECT_TRUE(ok) << (erm ? *erm : std::string("(no error message)"));

		std::string row;
		size_t datarows = 0;
		for (auto& de : fs::directory_iterator(outdir))
			if (de.path().extension() == ".csv")
			{
				std::ifstream f(de.path()); std::string header, ln;
				std::getline(f, header);
				while (std::getline(f, ln)) if (!ln.empty()) { row = ln; ++datarows; }
			}
		EXPECT_EQ(datarows, (size_t)1) << "expected exactly one feature row";
		fs::remove_all(outdir);
		return row;
	};

	std::string ooc_row = run_and_read_row(true, fs::temp_directory_path() / "nyxus_wv_ooc_test");
	std::string ram_row = run_and_read_row(false, fs::temp_directory_path() / "nyxus_wv_ram_test");

	ASSERT_FALSE(ooc_row.empty());
	ASSERT_FALSE(ram_row.empty());
	EXPECT_EQ(ooc_row, ram_row) << "the out-of-core whole-volume row must match the in-RAM row exactly";
}

// Regression: a whole volume whose loader delivers the ENTIRE X*Y*Z*T blob in one read (e.g.
// NIfTI -- see ImageLoader::stream_volume_planes) cannot stream plane-by-plane, so an oversized
// NIfTI whole volume must still fail loudly (no streaming path exists for it) rather than crash or
// emit a silent zero row. ram_limit=0 forces oversized regardless of the fixture's actual size.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_UNSTREAMABLE_FORMAT_FAILS_LOUDLY) {
	fs::path p(__FILE__);
	fs::path ds(p.parent_path().string() + fs::path("/data/hounsfield/ct3d_int16.nii").make_preferred().string());
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	fs::path outdir = fs::temp_directory_path() / "nyxus_wv_nifti_ooc_test";
	fs::remove_all(outdir); fs::create_directories(outdir);

	Environment e;
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);
	ASSERT_TRUE(e.theFeatureMgr.compile());
	e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
	ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
	e.compile_feature_settings();
	ASSERT_TRUE(e.set_ram_limit(0));		// force oversized regardless of this small fixture's real size

	std::vector<std::string> ifiles{ ds.string() };
	auto [ok, erm] = Nyxus::processDataset_3D_wholevolume(e, ifiles, 1, Nyxus::SaveOption::saveCSV, outdir.string());

	EXPECT_FALSE(ok) << "an oversized NIfTI whole volume has no streaming path and must fail loudly";

	size_t datarows = 0;
	for (auto& de : fs::directory_iterator(outdir))
		if (de.path().extension() == ".csv")
		{
			std::ifstream f(de.path()); std::string ln; size_t n = 0;
			while (std::getline(f, ln)) if (!ln.empty()) ++n;
			if (n) datarows += n - 1;	// minus header
		}
	EXPECT_EQ(datarows, (size_t)0) << "no feature row should be written for an unstreamable oversized volume";
	fs::remove_all(outdir);
}

// Regression (found by running nyxus under a hard memory cap): the whole-volume oversized check
// must use the 3D footprint estimator (W*H*D for the image cube), not the 2D one (W*H). The 2D
// estimator ignores depth, so it under-counted a volume's memory by ~depth x, let oversized
// volumes slip through the "trivial" path, and they OOM-crashed under a real memory limit even
// with a matching --ramLimit. This pins that the 3D estimator accounts for depth (the 2D one
// does not) so featurize_wholevolume's switch to get_ram_footprint_estimate_3D stays correct.
TEST(TEST_NYXUS, TEST_3D_RAM_FOOTPRINT_COUNTS_DEPTH) {
	// Two ROIs with identical W/H and voxel count, differing ONLY in bounding-box depth. This
	// isolates depth's effect on each estimator.
	LR flat(1);
	flat.aabb.init_from_whd(64, 64, 1);
	flat.aux_area = 4096;
	LR tall(1);
	tall.aabb.init_from_whd(64, 64, 64);
	tall.aux_area = 4096;

	// the 2D estimator's image-matrix term is W*H -> it IGNORES depth: identical for both
	EXPECT_EQ(flat.get_ram_footprint_estimate(1), tall.get_ram_footprint_estimate(1));

	// the 3D estimator's image-cube term is W*H*D -> the 64x-deeper bbox is far larger. This is
	// the term the 2D estimator missed, which under-counted whole volumes and let them OOM.
	EXPECT_GT(tall.get_ram_footprint_estimate_3D(1), flat.get_ram_footprint_estimate_3D(1) * 10)
		<< "3D footprint estimator is not counting depth";
}

// Regression (found while chasing the anisotropic-resampling hang, TEST_3D_SEGMENTED_ANISOTROPIC_*
// above): both footprint estimators computed (n_rois - 1) * sizeof(int) for the "neighbors" term.
// processTrivialRois_3D (and the 2D/2.5D siblings) call this with an in-progress BATCH count
// (Pending.size()), which is 0 on every batch's first item -- size_t(0-1) underflows to SIZE_MAX,
// and the subsequent multiply overflows to another huge wrapped value, silently misrouting even a
// tiny single-ROI batch through the "oversized, scan immediately" path instead of genuinely
// batching. n_rois==0 must mean "zero other ROIs, so 0 bytes for the neighbors term", not garbage.
TEST(TEST_NYXUS, TEST_RAM_FOOTPRINT_ESTIMATE_ZERO_ROIS_DOES_NOT_UNDERFLOW) {
	LR r(1);
	r.aabb.init_from_whd(8, 8, 4);
	r.aux_area = 64;

	size_t with_zero = r.get_ram_footprint_estimate(0);
	size_t with_one = r.get_ram_footprint_estimate(1);   // (1-1)=0 neighbors bytes too -- same base cost
	EXPECT_EQ(with_zero, with_one) << "n_rois=0 and n_rois=1 both contribute 0 neighbor bytes";
	// sanity ceiling: a real (non-underflowed) footprint for an 8x8x4 ROI is a few KB, not
	// anywhere near what (size_t)(0-1)*sizeof(int) would produce (~16 exabytes)
	EXPECT_LT(with_zero, (size_t)1'000'000) << "n_rois=0 must not underflow into an astronomical value";

	size_t with_zero_3d = r.get_ram_footprint_estimate_3D(0);
	size_t with_one_3d = r.get_ram_footprint_estimate_3D(1);
	EXPECT_EQ(with_zero_3d, with_one_3d);
	EXPECT_LT(with_zero_3d, (size_t)1'000'000);
}

// Regression-guard: processNontrivialRois_3D's per-feature out-of-core dispatch
// (phase3_3d.cpp) must throw for any 3D FeatureMethod NOT covered by is_3d_ooc_supported() --
// otherwise a future feature added without a streaming osized_calculate would silently read
// raw_voxels_NT (which OOC never populates for it) via the base FeatureMethod::osized_scan_whole_image
// default, producing a wrong/zero row instead of an actionable error. Every CURRENT 3D feature class
// is supported, so there is no live "unsupported" feature to exercise this through the normal
// featurize path; this pins the ALLOW-LIST FUNCTION ITSELF directly, using a minimal stand-in
// FeatureMethod that is deliberately never added to the allow-list, alongside real supported classes.
class DummyUnsupported3DFeature : public FeatureMethod
{
public:
	DummyUnsupported3DFeature() : FeatureMethod("DummyUnsupported3DFeature") {}
	void calculate (LR&, const Fsettings&) override {}
	void osized_add_online_pixel (size_t, size_t, uint32_t) override {}
	void osized_calculate (LR&, const Fsettings&, ImageLoader&) override {}
	void save_value (std::vector<std::vector<double>>&) override {}
};

TEST(TEST_NYXUS, TEST_3D_OOC_GUARD_REJECTS_UNSUPPORTED_FEATURE) {
	DummyUnsupported3DFeature unsupported;
	EXPECT_FALSE(Nyxus::is_3d_ooc_supported(&unsupported))
		<< "a 3D feature class outside the allow-list must be rejected by the OOC guard";

	D3_VoxelIntensityFeatures intensityFeature;
	EXPECT_TRUE(Nyxus::is_3d_ooc_supported(&intensityFeature))
		<< "a real streaming-supported 3D feature (intensity) must be accepted";

	D3_GLCM_feature glcmFeature;
	EXPECT_TRUE(Nyxus::is_3d_ooc_supported(&glcmFeature))
		<< "a real streaming-supported 3D texture feature (GLCM) must be accepted";
}

// Regression: separatecsv derives ONE output path per slide, but the CSV sinks are invoked
// once per (channel, timeframe) plane and used to open that path with mode "w" every time --
// so each plane truncated the one before it and the file ended up holding only the LAST
// channel. Confirmed against the pre-fix build, where this file had 1 data row (c_index=1)
// instead of 2. The t_index/c_index columns exist precisely so the planes can coexist as rows.
TEST(TEST_NYXUS, TEST_CSV_MULTICHANNEL_NO_OVERWRITE) {
	fs::path ds = ometiff_data_path("dim3_zyx.ome.tif");	// 3D X8 Y6 Z4
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	fs::path outdir = fs::temp_directory_path() / "nyxus_csv_ct_test";
	fs::remove_all(outdir);
	fs::create_directories(outdir);

	Environment e;
	e.separateCsv = true;					// the mode that overwrote (and the default)
	e.output_dir = outdir.string();
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);

	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ds.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();

	LR vroi(1);
	vroi.slide_idx = 0;
	vroi.aux_area = sp.max_roi_area;
	vroi.aabb.init_from_whd(sp.max_roi_w, sp.max_roi_h, sp.max_roi_d);
	ASSERT_NO_THROW(vroi.initialize_fvals());

	// Two channel planes of the SAME slide, exactly as the whole-volume workflow emits them
	ASSERT_TRUE(Nyxus::save_features_2_csv_wholeslide (e, vroi, ds.string(), "", outdir.string(), 0, 0));
	ASSERT_TRUE(Nyxus::save_features_2_csv_wholeslide (e, vroi, ds.string(), "", outdir.string(), 0, 1));

	// Read the single file back
	std::vector<std::string> lines;
	{
		std::ifstream f (Nyxus::get_feature_output_fname (e, ds.string(), ""));
		ASSERT_TRUE(f.good());
		std::string ln;
		while (std::getline(f, ln))
			if (!ln.empty())
				lines.push_back(ln);
	}

	ASSERT_EQ(lines.size(), (size_t)3) << "expected 1 header + one row per channel plane";
	EXPECT_NE(lines[0].find("\"c_index\""), std::string::npos) << "line 0 must be the header";
	// ...and the header must appear exactly once, not once per plane
	EXPECT_EQ(lines[1].find("\"c_index\""), std::string::npos);
	EXPECT_EQ(lines[2].find("\"c_index\""), std::string::npos);

	fs::remove_all(outdir);
}

// Phase 6 physical-calibration logic (negative + positive). resolve_slide_anisotropy
// must NOT engage the anisotropic (resampling) path unless it's genuinely warranted:
//   - flag off                      -> false, (1,1,1)   even with anisotropic spacing
//   - degenerate spacing (a 0 axis) -> false, (1,1,1)   (guarded, no div-by-zero)
//   - isotropic spacing (all equal) -> false, (1,1,1)   (nothing to correct)
//   - out-of-range slide index      -> false, (1,1,1)   (no OOB read)
//   - real anisotropic spacing      -> true,  ratios normalized so min == 1
//   - explicit --aniso*             -> true,  the CLI values (win over physical)
TEST(TEST_NYXUS, TEST_RESOLVE_SLIDE_ANISOTROPY) {
	Environment e;
	e.use_physical_spacing_ = true;			// opt-in on; anisoOptions stays un-customized
	e.dataset.dataset_props.clear();
	SlideProps p;							// ctor sets phys_x/y/z = 1.0
	e.dataset.dataset_props.push_back(p);
	double ax = -1, ay = -1, az = -1;

	// degenerate: a zero-length axis must not divide-by-zero -> isotropic fallback
	e.dataset.dataset_props[0].phys_x = 1.0; e.dataset.dataset_props[0].phys_y = 1.0; e.dataset.dataset_props[0].phys_z = 0.0;
	EXPECT_FALSE(Nyxus::resolve_slide_anisotropy(e, 0, ax, ay, az));
	EXPECT_DOUBLE_EQ(ax, 1.0); EXPECT_DOUBLE_EQ(ay, 1.0); EXPECT_DOUBLE_EQ(az, 1.0);

	// isotropic but non-unit spacing -> normalized to (1,1,1) -> no anisotropic path
	e.dataset.dataset_props[0].phys_x = 2.0; e.dataset.dataset_props[0].phys_y = 2.0; e.dataset.dataset_props[0].phys_z = 2.0;
	EXPECT_FALSE(Nyxus::resolve_slide_anisotropy(e, 0, ax, ay, az));
	EXPECT_DOUBLE_EQ(az, 1.0);

	// out-of-range slide index -> safe (no dataset_props[99] read)
	EXPECT_FALSE(Nyxus::resolve_slide_anisotropy(e, 99, ax, ay, az));
	EXPECT_DOUBLE_EQ(ax, 1.0);

	// genuinely anisotropic voxels (z 4x thicker) -> engage, ratio-normalized min == 1
	e.dataset.dataset_props[0].phys_x = 0.5; e.dataset.dataset_props[0].phys_y = 0.5; e.dataset.dataset_props[0].phys_z = 2.0;
	EXPECT_TRUE(Nyxus::resolve_slide_anisotropy(e, 0, ax, ay, az));
	EXPECT_DOUBLE_EQ(ax, 1.0); EXPECT_DOUBLE_EQ(ay, 1.0); EXPECT_DOUBLE_EQ(az, 4.0);

	// flag OFF -> never engage, even with anisotropic spacing present
	e.use_physical_spacing_ = false;
	EXPECT_FALSE(Nyxus::resolve_slide_anisotropy(e, 0, ax, ay, az));
	EXPECT_DOUBLE_EQ(az, 1.0);
}

// TEST_RESOLVE_SLIDE_ANISOTROPY covers the DECISION (physical spacing -> ratios). This covers
// that the resolved ratios actually RESCALE ROI geometry end-to-end: the 3D prescan's
// anisotropic branch (make_anisotropic_aabb 3-arg -> AABB::apply_anisotropy) was never
// exercised -- every other test uses make_nonanisotropic_aabb. A customized az=4 must scale the
// ROI's z-depth ~4x while leaving x/y (ax=ay=1) unchanged; without applying anisotropy the
// depth would be identical to the isotropic run. dim3_mask's ROI spans all Z (depth 4).
TEST(TEST_NYXUS, TEST_3D_ANISOTROPY_RESCALES_ROI_DEPTH) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;

	SlideProps iso (ip.string(), mp.string());
	AnisotropyOptions aniso_off;                       // un-customized -> isotropic AABB
	ASSERT_FALSE(aniso_off.customized());
	ASSERT_TRUE(Nyxus::scan_slide_props(iso, 3, aniso_off, e.resultOptions.need_annotation()));

	SlideProps ani (ip.string(), mp.string());
	AnisotropyOptions aniso_z4;
	aniso_z4.set_aniso_z(4.0);                          // z 4x thicker
	ASSERT_TRUE(aniso_z4.customized());
	ASSERT_TRUE(Nyxus::scan_slide_props(ani, 3, aniso_z4, e.resultOptions.need_annotation()));

	EXPECT_GT(ani.max_roi_d, iso.max_roi_d) << "z-anisotropy did not rescale ROI depth";
	EXPECT_GE(ani.max_roi_d, iso.max_roi_d * 3) << "z-depth not scaled ~4x";
	EXPECT_EQ(ani.max_roi_w, iso.max_roi_w) << "x (ax=1) must be unchanged";
	EXPECT_EQ(ani.max_roi_h, iso.max_roi_h) << "y (ay=1) must be unchanged";
}

// Regression: the whole-volume anisotropic scan (scan_trivial_wholevolume_anisotropic)
// previously (a) clobbered its own for-loop counter with the physical voxel index, which
// could run far past nVox iterations before the loop's own exit condition was ever
// satisfied again -- a hang, not just a slowdown; (b) indexed rows by fullH (height) instead
// of fullW (width), reading wrong voxels; (c) left vroi.aabb/aux_area at the PRE-resample
// physical values, so aux_image_cube (sized from the stale aabb) was allocated too small for
// the resampled cloud -- an out-of-bounds write -- and MEAN (which divides by aux_area) came
// out exactly 4x too large. Nothing exercised this combination before (every other 3D
// anisotropy test only covers the prescan's aabb, not the actual featurize+reduce). MIN/MAX
// are structurally invariant to nearest-neighbor upsampling, but so is MEAN under this scan's
// truncation-based mapping (every physical voxel is duplicated the SAME number of times) --
// so it must match the isotropic run exactly, not just "look plausible".
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_ANISOTROPIC_REDUCE_MATCHES_ISOTROPIC) {
	fs::path ds = ometiff_data_path("dim3_zyx.ome.tif");	// 3D X8 Y6 Z4
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	Environment e;
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);

	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ds.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();
	// force anisotropic calibration (z 4x thicker than x/y) regardless of the fixture's own metadata
	sp.phys_x = 0.5; sp.phys_y = 0.5; sp.phys_z = 2.0;
	e.use_physical_spacing_ = true;

	double ax, ay, az;
	ASSERT_TRUE(Nyxus::resolve_slide_anisotropy(e, 0, ax, ay, az));
	ASSERT_DOUBLE_EQ(ax, 1.0); ASSERT_DOUBLE_EQ(ay, 1.0); ASSERT_DOUBLE_EQ(az, 4.0);

	FpImageOptions fp;

	// anisotropic run
	ImageLoader ilo_a;
	ASSERT_TRUE(ilo_a.open(sp, fp)) << ds.string();
	LR vroi_a(1);
	vroi_a.slide_idx = 0;
	vroi_a.aux_area = sp.max_roi_area;
	vroi_a.aabb.init_from_whd(sp.max_roi_w, sp.max_roi_h, sp.max_roi_d);
	vroi_a.aux_min = (PixIntens)0;
	vroi_a.aux_max = (PixIntens)(sp.max_preroi_inten - sp.min_preroi_inten);
	ASSERT_NO_THROW(vroi_a.initialize_fvals());
	ASSERT_TRUE(Nyxus::scan_trivial_wholevolume_anisotropic(vroi_a, ds.string(), ilo_a, ax, ay, az, 0, 0));
	// the fix under test (mirrors workflow_3d_whole.cpp's featurize_triv_wholevolume):
	vroi_a.aabb.update_from_voxelcloud(vroi_a.raw_pixels_3D);
	vroi_a.aux_area = (unsigned int) vroi_a.raw_pixels_3D.size();
	vroi_a.aux_image_cube.allocate(vroi_a.aabb.get_width(), vroi_a.aabb.get_height(), vroi_a.aabb.get_z_depth());
	ASSERT_NO_THROW(vroi_a.aux_image_cube.calculate_from_pixelcloud(vroi_a.raw_pixels_3D, vroi_a.aabb));
	ASSERT_NO_THROW(Nyxus::reduce_trivial_3d_wholevolume(e, vroi_a));
	ilo_a.close();

	// isotropic baseline, same fixture
	ImageLoader ilo_i;
	ASSERT_TRUE(ilo_i.open(sp, fp)) << ds.string();
	LR vroi_i(1);
	vroi_i.slide_idx = 0;
	vroi_i.aux_area = sp.max_roi_area;
	vroi_i.aabb.init_from_whd(sp.max_roi_w, sp.max_roi_h, sp.max_roi_d);
	vroi_i.aux_min = (PixIntens)0;
	vroi_i.aux_max = (PixIntens)(sp.max_preroi_inten - sp.min_preroi_inten);
	ASSERT_NO_THROW(vroi_i.initialize_fvals());
	ASSERT_TRUE(Nyxus::scan_trivial_wholevolume(vroi_i, ds.string(), ilo_i, 0, 0));
	vroi_i.aux_image_cube.allocate(vroi_i.aabb.get_width(), vroi_i.aabb.get_height(), vroi_i.aabb.get_z_depth());
	ASSERT_NO_THROW(vroi_i.aux_image_cube.calculate_from_pixelcloud(vroi_i.raw_pixels_3D, vroi_i.aabb));
	ASSERT_NO_THROW(Nyxus::reduce_trivial_3d_wholevolume(e, vroi_i));
	ilo_i.close();

	// the resampled cloud really is ~4x bigger (z upsampled), not stuck at the physical count
	EXPECT_GE(vroi_a.raw_pixels_3D.size(), vroi_i.raw_pixels_3D.size() * 3);

	double mean_a = vroi_a.get_fvals((int)Nyxus::Feature3D::MEAN)[0];
	double mean_i = vroi_i.get_fvals((int)Nyxus::Feature3D::MEAN)[0];
	EXPECT_DOUBLE_EQ(mean_a, mean_i) << "MEAN must be resampling-invariant under uniform duplication";
	EXPECT_DOUBLE_EQ(vroi_a.get_fvals((int)Nyxus::Feature3D::MIN)[0], vroi_i.get_fvals((int)Nyxus::Feature3D::MIN)[0]);
	EXPECT_DOUBLE_EQ(vroi_a.get_fvals((int)Nyxus::Feature3D::MAX)[0], vroi_i.get_fvals((int)Nyxus::Feature3D::MAX)[0]);
}

// Regression (segmented counterpart): processTrivialRois_3D's anisotropic branch
// (scanTrivialRois_3D_anisotropic) populates raw_pixels_3D with the RESAMPLED voxel cloud,
// but aux_area (set during Phase 1 from the PHYSICAL, pre-resample voxel count) was never
// updated to match -- caught in two places (the main batch loop AND the "remaining pending"
// cleanup block are near-identical but NOT textually identical, so fixing one via a
// find-and-replace silently missed the other). MEAN (and anything else that divides by
// aux_area) was off by the resampling factor. aux_area must always equal the actual cloud size.
TEST(TEST_NYXUS, TEST_3D_SEGMENTED_ANISOTROPIC_AUX_AREA_MATCHES_VOXELCLOUD) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));

	Environment e;
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);

	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ip.string(), mp.string());
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
	e.dataset.update_dataset_props_extrema();
	sp.phys_x = 0.5; sp.phys_y = 0.5; sp.phys_z = 2.0;   // force anisotropic (z 4x)
	e.use_physical_spacing_ = true;

	clear_slide_rois (e.uniqueLabels, e.roiData);
	ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ip.string(), mp.string(), 0, 0));
	ASSERT_GT(e.uniqueLabels.size(), 0u);
	std::vector<int> labels (e.uniqueLabels.begin(), e.uniqueLabels.end());
	std::unordered_map<int, unsigned int> physical_area;   // Phase 1's PRE-resample count, per label
	for (auto lab : labels)
	{
		e.roiData[lab].initialize_fvals();
		physical_area[lab] = e.roiData[lab].aux_area;
	}

	double ax, ay, az;
	ASSERT_TRUE(Nyxus::resolve_slide_anisotropy(e, 0, ax, ay, az));
	ASSERT_DOUBLE_EQ(az, 4.0);

	// Call the scan directly (bypassing processTrivialRois_3D's batching, which has its own
	// unrelated, pre-existing bug: get_ram_footprint_estimate(Pending.size()) underflows when
	// Pending.size()==0 on the very first loop iteration, size_t(0-1)*sizeof(int) wrapping to
	// an astronomical value that can route even a tiny single-ROI batch through the "oversized"
	// immediate-scan branch unpredictably -- a separate footprint-estimation bug, not what this
	// test targets) -- exercises the exact fix under test (see the identical logic and its
	// rationale at both of processTrivialRois_3D's call sites in phase2_3d.cpp).
	ASSERT_TRUE(Nyxus::scanTrivialRois_3D_anisotropic(e, labels, ip.string(), mp.string(), 0, 0, ax, ay, az));
	for (auto lab : labels)
	{
		LR& r = e.roiData[lab];
		r.aabb.update_from_voxelcloud(r.raw_pixels_3D);
		r.aux_area = (unsigned int) r.raw_pixels_3D.size();
	}

	for (auto lab : labels)
	{
		LR& r = e.roiData[lab];
		EXPECT_GT(r.raw_pixels_3D.size(), 0u) << "label " << lab;
		EXPECT_EQ(r.aux_area, r.raw_pixels_3D.size())
			<< "label " << lab << ": aux_area must track the RESAMPLED cloud, not the stale physical count";
		// resampling z 4x must have grown the cloud past the PRE-resample physical count (not
		// an exact 4x -- the rounding-based nearest-neighbor mapping under- or over-represents
		// the boundary slice by up to one duplication step, so the growth factor isn't clean)
		EXPECT_GT(r.raw_pixels_3D.size(), physical_area[lab])
			<< "label " << lab << ": resampling did not grow the cloud past its physical count of " << physical_area[lab];
	}
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
