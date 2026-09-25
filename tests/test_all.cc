#define NOMINMAX	// keep Windows min/max macros from breaking dcmtk's OFvariant (DICOM tests)
#include <gtest/gtest.h>
#include <fstream>		// reading a written CSV back in TEST_CSV_MULTICHANNEL_NO_OVERWRITE_MECHANICS
#include "test_2d_gabor_mechanics.h"
#include "test_2d_gabor_skimage.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/feature_method.h"		// TEST_3D_OOC_GUARD_REJECTS_UNSUPPORTED_FEATURE_MECHANICS
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/features/3d_gldm.h"		// OOC-vs-RAM equality coverage (all texture families)
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/features/3d_glszm.h"
#include "../src/nyx/features/3d_gldzm.h"
#include "../src/nyx/features/3d_ngldm.h"
#include "../src/nyx/features/3d_ngtdm.h"
#include "../src/nyx/ome/format_detect.h"		// detect_container_family
#include "test_2d_contour_analytic.h"
#include "test_ome_meta_mechanics.h"		// native-OME metadata parsers / OmeAxes descriptor
#include "test_ometiff_mechanics.h"	// OME-TIFF native (z,c,t)->IFD read (core; no USE_Z5)
#include "test_2d_firstorder_common.h"
#include "test_2d_firstorder_regression.h"
#include "test_2d_firstorder_matlab.h"
#include "test_2d_intensity_histogram_regression.h"
#include "test_2d_intensity_degenerate_roi_mechanics.h"
#include "test_3d_intensity_degenerate_roi_mechanics.h"
#include "test_2d_radial_invariant.h"
#include "test_2d_radial_mechanics.h"
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
#include "test_2d_morphology_invariant.h"
#include "test_2d_morphology_mechanics.h"
#include "test_2d_moments_skimage.h"
#include "test_2d_moments_regression.h"
#include "test_2d_zernike_analytic.h"
#include "test_2d_zernike_invariant.h"
#include "test_2d_zernike_mechanics.h"
#include "test_2d_zernike_regression.h"
#include "test_2d_neighbor_common.h"
#include "test_2d_neighbor_regression.h"
#include "test_2d_neighbor_cellprofiler.h"
#include "test_2d_neighbor_analytic.h"
#include "test_2d_neighbor_invariant.h"
#include "test_initialization_mechanics.h"
#include "test_feature_manager_mechanics.h"
#include "test_2d_glcm_ibsi.h"
#include "test_2d_glcm_pyradiomics.h"
#include "test_2d_glcm_mirp.h"
#include "test_2d_gldm_ibsi.h"
#include "test_2d_gldm_pyradiomics.h"
#include "test_2d_glrlm_ibsi.h"
#include "test_2d_glrlm_pyradiomics.h"
#include "test_2d_glrlm_mirp.h"
#include "test_2d_gldzm_ibsi.h"
#include "test_2d_gldzm_mirp.h"
#include "test_2d_gldzm_regression.h"
#include "test_2d_glszm_ibsi.h"
#include "test_2d_glszm_mirp.h"
#include "test_2d_firstorder_ibsi.h"
#include "test_2d_firstorder_pyradiomics.h"
#include "test_2d_ngldm_ibsi.h"
#include "test_2d_ngldm_mirp.h"
#include "test_2d_ngldm_regression.h"
#include "test_2d_ngtdm_ibsi.h"
#include "test_2d_ngtdm_mirp.h"
#include "test_2d_ngtdm_mechanics.h"
#include "test_2d_glcm_regression.h"
#include "test_2d_gldm_regression.h"
#include "test_2d_gldm_mechanics.h"
#include "test_2d_glrlm_regression.h"
#include "test_2d_glszm_regression.h"
#include "test_2d_ngtdm_regression.h"
#include "test_roi_blacklist_mechanics.h"
#include "test_2d_nested_roi_mechanics.h"
#include "test_2d_tiff_loader_mechanics.h"
#include "test_imq_regression.h"
#include "test_imq_opencv.h"
#include "test_imq_cellprofiler.h"
#include "test_3d_nifti_mechanics.h"
#include "test_3d_layouta_mechanics.h"	// the layoutA (per-Z slice files) path
#include "test_io_plumbing_mechanics.h"	// guards shared by both dimensions
#include "test_3d_ooc_guards_mechanics.h"	// guards of the 3D out-of-core paths
#include "test_2d_wholeslide_mechanics.h"	// refusal points of the 2D whole-slide workflow
#include "test_omezarr_mechanics.h"
#include "test_2d_omezarr_mechanics.h"
#include "test_3d_morphology_regression.h"
#include "test_3d_morphology_mechanics.h"
#include "test_3d_morphology_matlab.h"
#include "test_3d_morphology_mirp.h"
#include "test_3d_gldzm_common.h"
#include "test_3d_gldzm_mechanics.h"
#include "test_3d_gldzm_mirp.h"
#include "test_3d_gldzm_regression.h"
#include "test_3d_ngldm_mirp.h"
#include "test_3d_ngldm_regression.h"
#include "test_3d_firstorder_pyradiomics.h"
#include "test_3d_firstorder_regression.h"
#include "test_3d_firstorder_matlab.h"
#include "test_3d_glcm_pyradiomics.h"
#include "test_3d_glcm_regression.h"
#include "test_3d_gldm_pyradiomics.h"
#include "test_3d_gldm_regression.h"
#include "test_3d_ngtdm_pyradiomics.h"
#include "test_3d_ngtdm_regression.h"
#include "test_3d_ngtdm_mechanics.h"
#include "test_3d_glrlm_pyradiomics.h"
#include "test_3d_glrlm_regression.h"
#include "test_3d_glszm_pyradiomics.h"
#include "test_3d_glszm_regression.h"
#include "test_3d_glszm_mechanics.h"
#include "test_3d_coverage_common.h"
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
	ASSERT_NO_THROW(test_3d_firstorder_p10_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_P90_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_p90_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ENERGY_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_energy_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ENTROPY_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_entropy_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_INTERQUARTILE_RANGE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_interquartile_range_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_KURTOSIS_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_kurtosis_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MAX_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_max_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MEAN_ABSOLUTE_DEVIATION_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_mean_absolute_deviation_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MEAN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_mean_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MEDIAN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_median_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MIN_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_min_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_RANGE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_range_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ROBUST_MEAN_ABSOLUTE_DEVIATION_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_robust_mean_absolute_deviation_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ROOT_MEAN_SQUARED_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_root_mean_squared_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_SKEWNESS_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_skewness_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_UNIFORMITY_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_uniformity_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_VARIANCE_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_firstorder_variance_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MATLAB) {
	ASSERT_NO_THROW(test_3d_firstorder_matlab());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_COVERED_IMAGE_INTENSITY_RANGE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_firstorder_covered_image_intensity_range_regression());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_MEDIAN_ABSOLUTE_DEVIATION_REGRESSION) {
	ASSERT_NO_THROW(test_3d_firstorder_median_absolute_deviation_regression());
}

TEST(TEST_NYXUS, TEST_3D_FIRSTORDER_ROBUST_MEAN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_firstorder_robust_mean_regression());
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

TEST(TEST_NYXUS, TEST_3D_NGTDM_BUSYNESS_R2_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_busyness_r2_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_COARSENESS_R2_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_coarseness_r2_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_COMPLEXITY_R2_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_complexity_r2_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_CONTRAST_R2_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_contrast_r2_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_STRENGTH_R2_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_strength_r2_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_MATRIX_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_matrix_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_MATRIX_R2_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_matrix_r2_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_DOCMATRIX_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_docmatrix_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_DUMP_PYRADIOMICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_dump_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_COARSENESS_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngtdm_coarseness_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_CONTRAST_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngtdm_contrast_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_BUSYNESS_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngtdm_busyness_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_COMPLEXITY_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngtdm_complexity_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_STRENGTH_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngtdm_strength_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_DUMP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngtdm_dump_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGTDM_DEFAULT_RADIUS_MECHANICS) {
	ASSERT_NO_THROW(test_3d_ngtdm_default_radius_mechanics());
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

TEST(TEST_NYXUS, TEST_3D_GLSZM_MATRIX_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_glszm_matrix_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SMALLMATRIX_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_glszm_smallmatrix_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_IBSI_GAPPED_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_glszm_ibsi_gapped_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_DUMP_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_glszm_dump_pyradiomics());
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

TEST(TEST_NYXUS, TEST_3D_GLSZM_SAE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_sae_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LAE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_lae_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_lglze_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_HGLZE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_hglze_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SALGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_salgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SAHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_sahgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LALGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_lalgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_LAHGLE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_lahgle_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_GLN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_gln_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_GLNN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_glnn_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SZN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_szn_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_SZNN_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_sznn_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_ZP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_zp_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_GLV_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_glv_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_ZV_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_zv_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_ZE_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_ze_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_DUMP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_dump_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_DEFAULT_GREYDEPTH_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_default_greydepth_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_CONSTANT_ROI_REGRESSION) {
	ASSERT_NO_THROW(test_3d_glszm_constant_roi_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_DEFAULT_GREYDEPTH_MECHANICS) {
	ASSERT_NO_THROW(test_3d_glszm_default_greydepth_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_GLSZM_IBSI_EQUALS_NO_BINNING_MECHANICS) {
	ASSERT_NO_THROW(test_3d_glszm_ibsi_equals_no_binning_mechanics());
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

TEST(TEST_NYXUS, TEST_3D_GLDM_MATRIX_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_matrix_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_smallmatrix_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_DUMP_PYRADIOMICS) {
	ASSERT_NO_THROW (test_3d_gldm_dump_pyradiomics());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_DE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_de_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_DN_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_dn_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_DNN_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_dnn_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_DV_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_dv_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_GLN_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_gln_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_GLV_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_glv_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_HGLE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_hgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LDE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_lde_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LDHGLE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_ldhgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LDLGLE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_ldlgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LGLE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_lgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_SDE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_sde_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_SDHGLE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_sdhgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_SDLGLE_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_sdlgle_regression()); }

TEST(TEST_NYXUS, TEST_3D_GLDM_DUMP_REGRESSION) {
	ASSERT_NO_THROW (test_3d_gldm_dump_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_CONSTANT_ROI_REGRESSION) {
	ASSERT_NO_THROW (test_3d_gldm_constant_roi_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDM_DE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_de_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_DN_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_dn_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_DNN_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_dnn_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_DV_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_dv_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_GLN_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_gln_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_GLV_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_glv_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_HGLE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_hgle_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LDE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_lde_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LDHGLE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_ldhgle_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LDLGLE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_ldlgle_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_LGLE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_lgle_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_SDE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_sde_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_SDHGLE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_sdhgle_nobinning_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLDM_SDLGLE_NOBINNING_REGRESSION) { ASSERT_NO_THROW (test_3d_gldm_sdlgle_nobinning_regression()); }

TEST(TEST_NYXUS, TEST_3D_GLDM_DUMP_NOBINNING_REGRESSION) {
	ASSERT_NO_THROW (test_3d_gldm_dump_nobinning_regression());
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

// The three volume features have separate MATLAB and MIRP assertions. The five PCA axis features
// are asserted against MIRP.

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_VOXEL_VOLUME_MATLAB) {
	ASSERT_NO_THROW(test_3d_morphology_voxel_volume_matlab());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_VOLUME_CONVEX_HULL_MATLAB) {
	ASSERT_NO_THROW(test_3d_morphology_volume_convex_hull_matlab());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_MESH_VOLUME_MATLAB) {
	ASSERT_NO_THROW(test_3d_morphology_mesh_volume_matlab());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_VOXEL_VOLUME_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_voxel_volume_mirp());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_VOLUME_CONVEX_HULL_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_volume_convex_hull_mirp());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_MESH_VOLUME_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_mesh_volume_mirp());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_MAJOR_AXIS_LEN_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_major_axis_len_mirp());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_MINOR_AXIS_LEN_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_minor_axis_len_mirp());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_LEAST_AXIS_LEN_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_least_axis_len_mirp());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_ELONGATION_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_elongation_mirp());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_FLATNESS_MIRP) {
	ASSERT_NO_THROW(test_3d_morphology_flatness_mirp());
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

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_DUMP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_morphology_dump_regression());
}

TEST(TEST_NYXUS, TEST_3D_MORPHOLOGY_COVMATRIX_AND_EIGENVALS_MECHANICS) {
	ASSERT_NO_THROW(test_3d_morphology_covmatrix_and_eigenvals_mechanics());
}


//***** 3D GLDZM mechanics *****

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZERO_LEVEL_VOXELS_ARE_ZONED_MECHANICS) {
	ASSERT_NO_THROW(test_3d_gldzm_zero_level_voxels_are_zoned_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZERO_LEVEL_VOXELS_ARE_ZONED_RADIOMICS_MECHANICS) {
	ASSERT_NO_THROW(test_3d_gldzm_zero_level_voxels_are_zoned_radiomics_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_NO_BINNING_SPELLINGS_AGREE_MECHANICS) {
	ASSERT_NO_THROW(test_3d_gldzm_no_binning_spellings_agree_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_RADIOMICS_BINNING_IS_IDENTITY_HERE_MECHANICS) {
	ASSERT_NO_THROW(test_3d_gldzm_radiomics_binning_is_identity_here_mechanics());
}


//***** 3D GLDZM vs mirp *****

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_sde_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_lde_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LGLZE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_lglze_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_HGLZE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_hglze_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDLGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_sdlgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDHGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_sdhgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDLGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_ldlgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDHGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_ldhgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLNU_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_glnu_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLNUN_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_glnun_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDNU_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zdnu_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDNUN_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zdnun_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZP_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zp_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLV_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_glv_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDV_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zdv_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDE_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zde_mirp());
}

//***** 3D GLDZM vs mirp, radiomics binning point *****

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_sde_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_lde_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LGLZE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_lglze_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_HGLZE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_hglze_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDLGLE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_sdlgle_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_SDHGLE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_sdhgle_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDLGLE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_ldlgle_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_LDHGLE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_ldhgle_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLNU_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_glnu_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLNUN_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_glnun_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDNU_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zdnu_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDNUN_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zdnun_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZP_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zp_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLV_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_glv_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDV_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zdv_radiomics_mirp());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDE_RADIOMICS_MIRP) {
	ASSERT_NO_THROW(test_3d_gldzm_zde_radiomics_mirp());
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

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDM_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zdm_regression());
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

TEST(TEST_NYXUS, TEST_3D_GLDZM_DUMP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_dump_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLM_COMPAT_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_glm_compat_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDM_COMPAT_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zdm_compat_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_GLM_RADIOMICS_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_glm_radiomics_regression());
}

TEST(TEST_NYXUS, TEST_3D_GLDZM_ZDM_RADIOMICS_REGRESSION) {
	ASSERT_NO_THROW(test_3d_gldzm_zdm_radiomics_regression());
}

//***** 3D NGLDM vetted vs MIRP, at the recipe where both tools share a grey-level ladder *****

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_lde_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hde_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LGLCE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_lglce_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HGLCE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hglce_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDLGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_ldlgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDHGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_ldhgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDLGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hdlgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDHGLE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hdhgle_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLNU_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_glnu_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLNUN_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_glnun_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCNU_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcnu_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCNUN_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcnun_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLV_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_glv_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCV_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcv_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCENT_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcent_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCENE_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcene_mirp());
}

//***** 3D NGLDM vetted vs MIRP at IBSI=true, where the raw intensity is the grey level *****

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_lde_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hde_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LGLCE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_lglce_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HGLCE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hglce_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDLGLE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_ldlgle_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_LDHGLE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_ldhgle_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDLGLE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hdlgle_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_HDHGLE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_hdhgle_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLNU_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_glnu_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLNUN_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_glnun_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCNU_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcnu_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCNUN_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcnun_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLV_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_glv_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCV_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcv_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCENT_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcent_ibsi_mirp());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCENE_IBSI_MIRP) {
	ASSERT_NO_THROW(test_3d_ngldm_dcene_ibsi_mirp());
}


//***** 3D NGLDM regression *****

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCP_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcp_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_GLM_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_glm_regression());
}

TEST(TEST_NYXUS, TEST_3D_NGLDM_DCM_REGRESSION) {
	ASSERT_NO_THROW(test_3d_ngldm_dcm_regression());
}

//***** Gabor (vetted vs scikit-image, plus the GPU path's plumbing guard) *****

TEST(TEST_NYXUS, TEST_2D_GABOR_CPP_STATIC_DEFAULTS_SKIMAGE){
    test_2d_gabor_cpp_static_defaults_skimage();
}

TEST(TEST_NYXUS, TEST_2D_GABOR_PYTHON_RAW_DEFAULTS_SKIMAGE){
    test_2d_gabor_python_raw_defaults_skimage();
}

TEST(TEST_NYXUS, TEST_2D_GABOR_GPU_RUNS_MECHANICS){
    test_2d_gabor_gpu_runs_mechanics();
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

TEST(TEST_NYXUS, TEST_FEATURE_MANAGER_MECHANICS) {
	test_feature_manager_mechanics();
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

TEST(TEST_NYXUS, TEST_2D_INTENSITY_ZERO_VALUED_ROI_RATIOS_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_intensity_zero_valued_roi_ratios_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_NONZERO_ROI_RATIOS_UNAFFECTED_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_intensity_nonzero_roi_ratios_unaffected_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_INTENSITY_ZERO_VALUED_ROI_RATIOS_MECHANICS)
{
	ASSERT_NO_THROW(test_3d_intensity_zero_valued_roi_ratios_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_INTENSITY_NONZERO_ROI_RATIOS_UNAFFECTED_MECHANICS)
{
	ASSERT_NO_THROW(test_3d_intensity_nonzero_roi_ratios_unaffected_mechanics());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_OFFSET_NEGATIVE_MIN_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_offset_negative_min_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_IDENTITY_NONNEGATIVE_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_identity_nonnegative_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_QUANTIZED_FLOAT_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_quantized_float_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_PRESERVE_HU_FLOAT_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_preserve_hu_float_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_QUANTIZED_WINDOW_CLAMPS_ABOVE_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_quantized_window_clamps_above_analytic());
}

TEST(TEST_NYXUS, TEST_HU_FPIMAGE_OPTIONS_REJECT_MALFORMED_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_fpimage_options_reject_malformed_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_NONFINITE_SLIDE_RANGE_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_nonfinite_slide_range_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_NONFINITE_PIXEL_QUANTIZED_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_nonfinite_pixel_quantized_analytic());
}

TEST(TEST_NYXUS, TEST_HU_SCANNED_RANGE_DEGENERATE_SLIDE_SETTLES_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_scanned_range_degenerate_slide_settles_analytic());
}

TEST(TEST_NYXUS, TEST_HU_SCANNED_RANGE_PASSTHROUGH_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_scanned_range_passthrough_analytic());
}

TEST(TEST_NYXUS, TEST_2D_INTENSITY_EMPTY_ROI_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_intensity_empty_roi_mechanics());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_OFFSET_HAS_NO_UPPER_CLAMP_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_offset_has_no_upper_clamp_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_ZARR_FLOAT_QUANTIZED_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_zarr_float_quantized_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_ZARR_SIGNED_OFFSET_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_zarr_signed_offset_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_ZARR_UNSIGNED_IDENTITY_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_zarr_unsigned_identity_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_CONSTANT_FLOAT_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_constant_float_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_CONSTANT_NEGATIVE_FLOAT_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_constant_negative_float_analytic());
}

TEST(TEST_NYXUS, TEST_HU_GREY_LEVEL_CAST_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_grey_level_cast_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_OFFSET_SATURATES_ABOVE_GREY_RANGE_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_offset_saturates_above_grey_range_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_QUANTIZED_SATURATES_ABOVE_GREY_RANGE_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_quantized_saturates_above_grey_range_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_STORED_SMALL_SLOPE_KEEPS_EVERY_LEVEL_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_stored_small_slope_keeps_every_level_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_STORED_UNIT_SLOPE_MATCHES_OFFSET_MAP_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_stored_unit_slope_matches_offset_map_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_STORED_SIGNED_AND_FRACTIONAL_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_stored_signed_and_fractional_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_STORED_NONNEGATIVE_SHIFT_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_stored_nonnegative_shift_analytic());
}

TEST(TEST_NYXUS, TEST_HU_DOMAIN_MAP_STORED_FALLS_BACK_TO_OFFSET_ANALYTIC)
{
	ASSERT_NO_THROW(test_hu_domain_map_stored_falls_back_to_offset_analytic());
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

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_DICOM_FRACTIONAL_SLOPE_ROUNDS_UNDER_PRESERVE_HU_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_dicom_fractional_slope_rounds_under_preserve_hu_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_LOADER_DICOM_FRACTIONAL_SLOPE_TRUNCATES_WITHOUT_ROUND_OFFSET_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_loader_dicom_fractional_slope_truncates_without_round_offset_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_HU_DICOM_FRACTIONAL_SLOPE_LOAD_PATH_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_hu_dicom_fractional_slope_load_path_mechanics());
}
#endif

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_ENTROPY_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_entropy_regression());
}

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MEDIAN_ABSOLUTE_DEVIATION_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_firstorder_median_absolute_deviation_regression());
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

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ROI_RADIUS_DISK_CLOSED_FORM_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_morphology_roi_radius_disk_closed_form_analytic());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ROI_RADIUS_DISKS_SKIMAGE)
{
	ASSERT_NO_THROW(test_2d_morphology_roi_radius_disks_skimage());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_EDT_EQUALS_EXHAUSTIVE_SCAN_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_morphology_edt_equals_exhaustive_scan_invariant());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ROI_RADIUS_ENGINES_AGREE_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_morphology_roi_radius_engines_agree_invariant());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ROI_RADIUS_ENGINE_CHOICE_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_morphology_roi_radius_engine_choice_invariant());
}

TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_ROI_RADIUS_PATHS_AGREE_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_morphology_roi_radius_paths_agree_invariant());
}

TEST(TEST_NYXUS, TEST_2D_EDT_NO_SITES_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_edt_no_sites_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_EDT_EMPTY_RASTER_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_edt_empty_raster_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_EDT_SINGLE_CELL_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_edt_single_cell_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_EDT_SINGLE_SITE_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_edt_single_site_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_EDT_BOUNDARY_SITES_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_edt_boundary_sites_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_ROI_RADIUS_EMPTY_INPUTS_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_roi_radius_empty_inputs_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_ROI_RADIUS_SINGLE_PIXEL_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_roi_radius_single_pixel_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_ROI_RADIUS_CONTOUR_OUTSIDE_BBOX_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_roi_radius_contour_outside_bbox_mechanics());
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

TEST(TEST_NYXUS, TEST_2D_RADIAL_BIN_CONVENTIONS_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_radial_bin_conventions_regression());
}

TEST(TEST_NYXUS, TEST_2D_RADIAL_FRAC_AT_D_IS_A_PARTITION_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_radial_frac_at_d_is_a_partition_invariant());
}

TEST(TEST_NYXUS, TEST_2D_RADIAL_EMPTY_BINS_ARE_ZERO_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_radial_empty_bins_are_zero_invariant());
}

TEST(TEST_NYXUS, TEST_2D_RADIAL_CV_IS_WITHIN_ITS_BOUND_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_radial_cv_is_within_its_bound_invariant());
}

TEST(TEST_NYXUS, TEST_2D_RADIAL_CONTOUR_FRAME_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_radial_contour_frame_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_RADIAL_CENTER_AND_RADIUS_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_radial_center_and_radius_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_RADIAL_BIN_INDEX_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_radial_bin_index_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_MOMENTS_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_zernike_moments_regression());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_MOMENTS_ANALYTIC)
{
	ASSERT_NO_THROW(test_2d_zernike_moments_analytic());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_ZEROTH_MOMENT_IS_ONE_OVER_PI_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_zernike_zeroth_moment_is_one_over_pi_invariant());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_FIRST_MOMENT_ABOUT_THE_CENTROID_VANISHES_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_zernike_first_moment_about_the_centroid_vanishes_invariant());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_MAGNITUDES_ARE_WITHIN_THEIR_BOUND_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_zernike_magnitudes_are_within_their_bound_invariant());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_INDEX_SET_MATCHES_THE_DECLARED_COUNT_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_zernike_index_set_matches_the_declared_count_invariant());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_GEOMETRY_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_zernike_geometry_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_ZERNIKE_EVERY_PIXEL_IS_INSIDE_THE_UNIT_DISK_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_zernike_every_pixel_is_inside_the_unit_disk_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_PERCENT_TOUCHING_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_neighbor_percent_touching_regression());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_PERCENT_TOUCHING_BOUNDED_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_neighbor_percent_touching_bounded_invariant());
}

TEST(TEST_NYXUS, TEST_2D_NEIGHBOR_PERCENT_TOUCHING_ENCLOSED_INVARIANT)
{
	ASSERT_NO_THROW(test_2d_neighbor_percent_touching_enclosed_invariant());
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

//***** 2D NGTDM vs mirp *****

TEST(TEST_NYXUS, TEST_2D_NGTDM_COARSENESS_MIRP)
{
	ASSERT_NO_THROW(test_2d_ngtdm_coarseness_mirp());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_CONTRAST_MIRP)
{
	ASSERT_NO_THROW(test_2d_ngtdm_contrast_mirp());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_BUSYNESS_MIRP)
{
	ASSERT_NO_THROW(test_2d_ngtdm_busyness_mirp());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_COMPLEXITY_MIRP)
{
	ASSERT_NO_THROW(test_2d_ngtdm_complexity_mirp());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_STRENGTH_MIRP)
{
	ASSERT_NO_THROW(test_2d_ngtdm_strength_mirp());
}

//***** 2D NGTDM mechanics *****

TEST(TEST_NYXUS, TEST_2D_NGTDM_IBSI_MODE_IGNORES_N_LEVELS_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_ngtdm_ibsi_mode_ignores_n_levels_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_NGTDM_SLICE_HELPER_RESTORES_N_LEVELS_MECHANICS)
{
	ASSERT_NO_THROW(test_2d_ngtdm_slice_helper_restores_n_levels_mechanics());
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

//***** PyRadiomics tests of 2D GLDM *****

TEST(TEST_NYXUS, TEST_2D_GLDM_SDE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_sde_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_lde_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_GLN_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_gln_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DN_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_dn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DNN_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_dnn_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_GLV_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_glv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DV_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_dv_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_DE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_de_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LGLE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_lgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_HGLE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_hgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_SDLGLE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_sdlgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_SDHGLE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_sdhgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDLGLE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_ldlgle_pyradiomics());
}

TEST(TEST_NYXUS, TEST_2D_GLDM_LDHGLE_PYRADIOMICS)
{
	ASSERT_NO_THROW(test_2d_gldm_ldhgle_pyradiomics());
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

//***** 2D GLSZM vs mirp *****

TEST(TEST_NYXUS, TEST_2D_GLSZM_SAE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_sae_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LAE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_lae_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LGLZE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_lglze_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_HGLZE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_hglze_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SALGLE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_salgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SAHGLE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_sahgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LALGLE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_lalgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_LAHGLE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_lahgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLN_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_gln_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLNN_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_glnn_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SZN_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_szn_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_SZNN_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_sznn_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZP_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_zp_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_GLV_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_glv_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZV_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_zv_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLSZM_ZE_MIRP) {
	ASSERT_NO_THROW(test_2d_glszm_ze_mirp());
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

TEST(TEST_NYXUS, TEST_2D_FIRSTORDER_MATLAB)
{
	ASSERT_NO_THROW(test_2d_firstorder_matlab());
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

//***** mirp tests of 2D GLDZM *****

TEST(TEST_NYXUS, TEST_2D_GLDZM_SDE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_sde_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LDE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_lde_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LGLZE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_lglze_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_HGLZE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_hglze_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_SDLGLE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_sdlgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_SDHGLE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_sdhgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LDLGLE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_ldlgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_LDHGLE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_ldhgle_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLNU_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_glnu_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLNUN_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_glnun_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDNU_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdnu_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDNUN_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdnun_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZP_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_zp_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLV_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_glv_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDV_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdv_mirp());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDE_MIRP)
{
	ASSERT_NO_THROW(test_2d_gldzm_zde_mirp());
}

//***** 2D GLDZM drift guards *****

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLM_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_gldzm_glm_regression());
}

TEST(TEST_NYXUS, TEST_2D_GLDZM_ZDM_REGRESSION)
{
	ASSERT_NO_THROW(test_2d_gldzm_zdm_regression());
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

TEST(TEST_NYXUS, TEST_2D_GLDZM_GLV_IBSI)
{
	ASSERT_NO_THROW(test_2d_gldzm_glv_ibsi());
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

TEST(TEST_NYXUS, TEST_IMQ_MIN_SATURATION_CONSTANT_ROI_REGRESSION)
{
	ASSERT_NO_THROW(test_imq_min_saturation_constant_roi_regression());
}

TEST(TEST_NYXUS, TEST_IMQ_MAX_SATURATION_CONSTANT_ROI_REGRESSION)
{
	ASSERT_NO_THROW(test_imq_max_saturation_constant_roi_regression());
}

TEST(TEST_NYXUS, TEST_IMQ_MIN_SATURATION_NARROW_MASK_REGRESSION)
{
	ASSERT_NO_THROW(test_imq_min_saturation_narrow_mask_regression());
}

TEST(TEST_NYXUS, TEST_IMQ_MAX_SATURATION_NARROW_MASK_REGRESSION)
{
	ASSERT_NO_THROW(test_imq_max_saturation_narrow_mask_regression());
}

TEST(TEST_NYXUS, TEST_IMQ_POWER_SPECTRUM_SLOPE_LARGE_ROI_REGRESSION)
{
	ASSERT_NO_THROW(test_imq_power_spectrum_slope_large_roi_regression());
}

//***** 3D i/o ***** 

TEST(TEST_NYXUS, TEST_3D_NIFTI_LOADER_MECHANICS) {
	ASSERT_NO_THROW (test_3d_nifti_loader_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_NIFTI_DATA_ACCESS_CONSISTENCY_MECHANICS) {
	ASSERT_NO_THROW (test_3d_nifti_data_access_consistency_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_NIFTI_FACADE_STREAM_EQUIVALENCE_MECHANICS) {
	ASSERT_NO_THROW (test_3d_nifti_facade_stream_equivalence_mechanics());
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

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_SIGNED_SOURCE_DOMAIN_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_signed_source_domain_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_FLOAT_SOURCE_DOMAIN_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_float_source_domain_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_UINT32_ACCESSOR_CLAMPS_NEGATIVE_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_uint32_accessor_clamps_negative_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_UINT32_ACCESSOR_UNSIGNED_EXACT_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_uint32_accessor_unsigned_exact_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_RAW_UINT32_ACCESSOR_NONFINITE_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_raw_uint32_accessor_nonfinite_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_TILELOADER_SIGNED_OFFSET_MAP_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_tileloader_signed_offset_map_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_TILELOADER_FLOAT_QUANTIZED_MAP_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_tileloader_float_quantized_map_mechanics());
}

TEST(TEST_NYXUS, TEST_2D_OMEZARR_TILELOADER_UNSIGNED_UNAFFECTED_MECHANICS) {
	ASSERT_NO_THROW (test_2d_omezarr_tileloader_unsigned_unaffected_mechanics());
}

TEST(TEST_NYXUS, TEST_OMEZARR_5D_CHANNEL_TIME_ADDRESSING_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim5.ome.zarr", 2, 3, 4));
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_5D_CHANNEL_TIME_ADDRESSING_MECHANICS) {
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim5.ome.zarr", 2, 3, 4));
}

// All 6 legal orderings of {t,c,z}: passes only if the loader honors the NGFF
// 'axes' metadata instead of assuming a fixed [T,C,Z,Y,X] order.
TEST(TEST_NYXUS, TEST_OMEZARR_ALL_5D_PERMUTATIONS_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_all_5d_permutations_mechanics());
}

// 4D (rank-4): time-only and channel-only.
TEST(TEST_NYXUS, TEST_OMEZARR_4D_TZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim4_tzyx.ome.zarr", 2, 1, 4));
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim4_tzyx.ome.zarr", 2, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_4D_CZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim4_czyx.ome.zarr", 1, 3, 4));
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim4_czyx.ome.zarr", 1, 3, 4));
}

// End-to-end through the wired volumetric consumer (scan_trivial_wholevolume).
TEST(TEST_NYXUS, TEST_OMEZARR_WHOLEVOLUME_CONSUMER_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_wholevolume_consumer("dim3_zyx.ome.zarr", 4));
}

// Facade whole-volume assembly (the streamed Z-planes stacked into one X*Y*Z buffer).
// Wired consumer reads the correct plane for every (channel, timeframe), not just (0,0).
TEST(TEST_NYXUS, TEST_OMEZARR_WHOLEVOLUME_CONSUMER_CT_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_wholevolume_consumer_ct("dim5.ome.zarr", 2, 3, 4));
}

TEST(TEST_NYXUS, TEST_OMEZARR_FACADE_VOLUME_3D_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_facade_volume("dim3_zyx.ome.zarr", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_FACADE_VOLUME_5D_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_facade_volume("dim5.ome.zarr", 2, 3, 4));
}

// Lower-rank stores: a fixed shape[2..4] loader would crash; the axis-role loader
// reads 3D (ZYX) and 2D (YX) correctly.
TEST(TEST_NYXUS, TEST_OMEZARR_3D_ZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim3_zyx.ome.zarr", 1, 1, 4));
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_3D_ZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim3_zyx.ome.zarr", 1, 1, 4));
}

TEST(TEST_NYXUS, TEST_OMEZARR_2D_YX_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim2_yx.ome.zarr", 1, 1, 1));
}

// OME-Zarr 0.5 (Zarr v3): zarr.json metadata + 'ome'-wrapped NGFF + 0/c/... chunk keys,
// read through the z5 Dataset API (v2/v3-agnostic). Same coordinate encoding as the v2
// stores, so the addressing / facade / CT-count helpers apply unchanged.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim5_v3.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim5_v3.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_facade_volume("dim5_v3.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_CT_COUNTS_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim5_v3.ome.zarr", 2, 3, 4));
}
// Zstd-compressed Zarr v3 (the zarr 3.x / real-world default codec). z5 decodes the
// bytes+zstd v3 codec pipeline only when the build found libzstd and its header, so this
// case is compiled in on the same condition CMake uses to enable the codec; a build
// without zstd cannot read the fixture and must not be failed for it.
#ifdef WITH_ZSTD
TEST(TEST_NYXUS, TEST_OMEZARR_V3_ZSTD_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim5_v3_zstd.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim5_v3_zstd.ome.zarr", 2, 3, 4));
}
#endif

// Sharded Zarr v3 (the ``sharding_indexed`` codec) -- how large v3 stores, including Axle's,
// actually lay out data: many inner chunks packed into one shard object per (t,c). z5 3.x
// reads it via ShardedDataset, chosen automatically when zarr.json carries a shard shape; the
// nyxus loader is UNCHANGED because it reads through readSubarray, which unpacks the inner
// chunks from the shard transparently. The fixture's inner chunk is (z,y,x)=(1,3,4) so each
// 6x8 plane is a 2x2 grid of inner chunks living inside one shard -- the read must assemble
// across inner-chunk boundaries within a shard. Same 1..1152 TCZYX encoding as the other v3.
//
// Coverage is the whole-volume facade + prescan, NOT assert_omezarr_addressing: with sharding,
// tileWidth/Height report the INNER chunk (4x3), so a single loadTileFromFile(0,0,...) reads
// only the top-left inner chunk, not the whole plane -- the addressing helper's one-tile-per-
// plane assumption. facade_volume assembles the full inner-chunk grid and checks every voxel,
// which is the correct coverage for a multi-chunk store (same reason the multichunk v2 fixture
// uses it). It drives both loadTileFromFile (abstract stack) and readSubarray under the hood.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_SHARDED_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_facade_volume("dim5_v3_sharded.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_SHARDED_CT_COUNTS_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim5_v3_sharded.ome.zarr", 2, 3, 4));
}
// Prescan over the sharded store (raw loader's readSubarray, driven through the inner-chunk
// tile grid by for_each_voxel): whole-slide, so the ROI is the whole X*Y*Z volume.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_SHARDED_PRESCAN_MECHANICS) {
	fs::path ip = omezarr_data_path("dim5_v3_sharded.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// Blosc-compressed Zarr v3 (common in real v3 stores alongside zstd). z5 decodes the
// bytes+blosc v3 codec pipeline when built WITH_BLOSC (already required for OME-Zarr).
TEST(TEST_NYXUS, TEST_OMEZARR_V3_BLOSC_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim5_v3_blosc.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim5_v3_blosc.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_V3_BLOSC_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_facade_volume("dim5_v3_blosc.ome.zarr", 2, 3, 4));
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

TEST(TEST_NYXUS, TEST_OMEZARR_V3_MULTISHARD_FACADE_VOLUME_MECHANICS) {
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
	      std::vector<uint32_t> vol_i1, vol_s1;
	      ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, c, t, vol_i1, vol_s1));
	      const std::vector<uint32_t>& vol = vol_i1;
	      ASSERT_EQ(vol.size(), (size_t)X * Y * Z);
	      for (int z = 0; z < Z; ++z)
	        for (int y = 0; y < Y; ++y)
	          for (int x = 0; x < X; ++x)
	            ASSERT_EQ(vol[(size_t)z * X * Y + (size_t)y * X + x], dim5_multishard_enc(x, y, z, c, t))
	                << "multishard vol (x" << x << " y" << y << " z" << z << " c" << c << " t" << t << ")";
	  }
	il.close();
}

TEST(TEST_NYXUS, TEST_OMEZARR_V3_MULTISHARD_CT_COUNTS_MECHANICS) {
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
TEST(TEST_NYXUS, TEST_OMEZARR_V3_MULTISHARD_PRESCAN_MECHANICS) {
	fs::path ip = omezarr_data_path("dim5_v3_multishard.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

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

// Exercises omezarr.h's NyxusOmeZarrLoader via ImageLoader::stream_volume_planes -- the
// path that read zero past the first Z-plane of a chunk before the fix.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_ZCHUNKED_FACADE_VOLUME_MECHANICS) {
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
	      std::vector<uint32_t> vol_i2, vol_s2;
	      ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, c, t, vol_i2, vol_s2));
	      const std::vector<uint32_t>& vol = vol_i2;
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
TEST(TEST_NYXUS, TEST_OMEZARR_V3_ZCHUNKED_PRESCAN_MECHANICS) {
	fs::path ip = omezarr_data_path("dim5_v3_zchunked.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, dim5_zchunked_enc(7, 5, 6, 1, 0));	// last voxel, last channel
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 7));
}

// No 'axes' metadata -> the loader falls back to legacy 5D TCZYX and still reads.
TEST(TEST_NYXUS, TEST_OMEZARR_NOAXES_FALLBACK_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_addressing("dim5_noaxes.ome.zarr", 2, 3, 4));
}

TEST(TEST_NYXUS, TEST_RAW_OMEZARR_NOAXES_FALLBACK_MECHANICS) {
	ASSERT_NO_THROW (assert_raw_omezarr_addressing("dim5_noaxes.ome.zarr", 2, 3, 4));
}

// Loaders advertise the real C/T extents (numberChannels/fullTimestamps), which
// is what activates the pipeline's channel/timeframe iteration. dim5_noaxes proves
// the positional fallback reports counts too.
TEST(TEST_NYXUS, TEST_OMEZARR_CT_COUNTS_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim5.ome.zarr", 2, 3, 4));
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim4_tzyx.ome.zarr", 2, 1, 4));
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim4_czyx.ome.zarr", 1, 3, 4));
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim3_zyx.ome.zarr", 1, 1, 4));
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim2_yx.ome.zarr", 1, 1, 1));
	ASSERT_NO_THROW (assert_omezarr_ct_counts("dim5_noaxes.ome.zarr", 2, 3, 4));
}

// Physical calibration: loaders surface coordinateTransformations scale + unit.
TEST(TEST_NYXUS, TEST_OMEZARR_PHYSICAL_CALIBRATION_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_physical_calibration_mechanics());
}

// Unit canonicalization: a nanometer-declared store must report the same physX/Y/Z and
// "micrometer" as the equivalent micrometer-declared store above -- proves conversion, not
// just passthrough of the raw unit string.
TEST(TEST_NYXUS, TEST_OMEZARR_UNIT_CANONICALIZATION_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_unit_canonicalization_mechanics());
}

// Multi-CHUNK plane: real OME-Zarr splits each Y/X plane across a chunk grid (typically
// 512x512), and dim5_multichunk uses 3x4 chunks over the 6x8 plane. The volumetric read
// must walk the whole tile grid: fetching only chunk (0,0) returns wrong data past the
// first chunk (and over-reads its buffer). Every other fixture is one-chunk-per-plane,
// which is why this went unnoticed. Covers ImageLoader::assemble_tile_layer...
TEST(TEST_NYXUS, TEST_OMEZARR_MULTICHUNK_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_facade_volume("dim5_multichunk.ome.zarr", 2, 3, 4));
}
// ...and RawImageLoader::for_each_voxel (the prescan), which walks the same tile grid:
// the encoded values are 1..1152 over all (c,t), and the ROI is the whole X*Y*Z volume.
TEST(TEST_NYXUS, TEST_OMEZARR_MULTICHUNK_PRESCAN_MECHANICS) {
	fs::path ip = omezarr_data_path("dim5_multichunk.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// PARTIAL edge chunks: chunk (4,5) does not divide the 6x8 plane, so the last row-chunk is 2
// tall and the last col-chunk is 3 wide. dim5_multichunk above (3x4 over 6x8) tiles exactly,
// so the validH/validW seam clamp never ran on the OME-Zarr path either. Asserting the exact
// value at every voxel is the seam check; same 1..1152 TCZYX encoding as dim5_multichunk.
TEST(TEST_NYXUS, TEST_OMEZARR_ODDCHUNK_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_omezarr_facade_volume("dim5_oddchunk.ome.zarr", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_OMEZARR_ODDCHUNK_PRESCAN_MECHANICS) {
	fs::path ip = omezarr_data_path("dim5_oddchunk.ome.zarr");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// Negative: out-of-range channel/timeframe through the whole-volume facade must throw.
TEST(TEST_NYXUS, TEST_OMEZARR_STREAM_VOLUME_OUT_OF_RANGE_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_stream_volume_out_of_range_mechanics());
}

// Negative: out-of-range Z/C/T plane index must throw.
TEST(TEST_NYXUS, TEST_OMEZARR_OUT_OF_RANGE_THROWS_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_out_of_range_throws_mechanics());
}

// Illegal / adversarial: malformed metadata must be rejected cleanly, not crash.
TEST(TEST_NYXUS, TEST_OMEZARR_MALFORMED_THROWS_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_malformed_throws_mechanics());
}

// Nested v2 chunk keys ('/' separator) + blosc -- the layout bioformats2raw writes by default.
TEST(TEST_NYXUS, TEST_OMEZARR_NESTED_CHUNK_KEYS_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_nested_chunk_keys_mechanics());
}

// The two refusals real converter output runs into must name the cause: a big-endian array and
// a bioformats2raw store root, both of which hold good data.
TEST(TEST_NYXUS, TEST_OMEZARR_DIAGNOSED_REFUSALS_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_diagnosed_refusals_mechanics());
}

// A chunk grid uneven along Z, Y and X, through the facade and the prescan.
TEST(TEST_NYXUS, TEST_OMEZARR_CHUNKED_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_chunked_facade_volume_mechanics());
}
TEST(TEST_NYXUS, TEST_OMEZARR_CHUNKED_PRESCAN_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_chunked_prescan_mechanics());
}

// A Z chunk that spans the whole volume has no bounded streaming path.
TEST(TEST_NYXUS, TEST_OMEZARR_WHOLE_Z_CHUNK_IS_UNSTREAMABLE_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_whole_z_chunk_is_unstreamable_mechanics());
}

// The prescan's ROI geometry follows --use-physical-spacing, as every other pass does.
TEST(TEST_NYXUS, TEST_OMEZARR_PHYSICAL_SPACING_PRESCAN_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_physical_spacing_prescan_mechanics());
}

// More than one channel or timepoint is refused in 2D rather than featurized at C=0, T=0 alone.
TEST(TEST_NYXUS, TEST_OMEZARR_MULTICHANNEL_TIMEPOINT_REFUSED_MECHANICS) {
	ASSERT_NO_THROW (test_omezarr_multichannel_timepoint_refused_mechanics());
}

// N1 (negative): a Zarr v3 store whose zarr.json declares a codec z5 does not support. z5's
// readV3CodecsFromJson throws "unsupported zarr v3 codec" during metadata parse (openDataset),
// so the loader must surface a clean error, not crash. Both loader stacks must reject it.
TEST(TEST_NYXUS, TEST_OMEZARR_V3_UNSUPPORTED_CODEC_THROWS_MECHANICS) {
	fs::path ds = omezarr_data_path("dim5_v3_badcodec.ome.zarr");
	ASSERT_TRUE(fs::exists(ds)) << ds.string();
	EXPECT_ANY_THROW(NyxusOmeZarrLoader<uint32_t>(1, ds.string()));
	EXPECT_ANY_THROW(RawOmezarrLoader(ds.string()));
}

// P4 (positive): the crash's positive twin on the OME-Zarr path -- a T>1 Zarr intensity paired
// with a single-timeframe ZYX Zarr mask. Zarr never crashed (no T axis to over-index), but it
// was never asserted. The prescan must reuse the mask across timeframes and find the ROI.
TEST(TEST_NYXUS, TEST_OMEZARR_MULTITIMEFRAME_MASK_PRESCAN_MECHANICS) {
	fs::path ip = omezarr_data_path("dim5.ome.zarr");        // T=2, C=3, Z=4
	fs::path mp = omezarr_data_path("dim3_mask.ome.zarr");   // ZYX (T=1) label mask
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)(4 * 4 * 6));
}

#endif // OMEZARR_SUPPORT


//***** OME-TIFF native (z,c,t) -> IFD read (core; runs in every build) *****

TEST(TEST_NYXUS, TEST_OMETIFF_5D_CHANNEL_TIME_ADDRESSING_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_addressing("dim5.ome.tif", 2, 3, 4));
}
TEST(TEST_NYXUS, TEST_RAW_OMETIFF_5D_CHANNEL_TIME_ADDRESSING_MECHANICS) {
	ASSERT_NO_THROW (assert_raw_ometiff_addressing("dim5.ome.tif", 2, 3, 4));
}

// All 6 legal DimensionOrder values: passes only if ifdForPlane honors DimensionOrder.
TEST(TEST_NYXUS, TEST_OMETIFF_ALL_5D_PERMUTATIONS_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_all_5d_permutations_mechanics());
}

// A multi-timeframe OME-TIFF paired with a single-timeframe (ZYX) mask: the 3D prescan
// (RawImageLoader::for_each_voxel) reuses that one mask plane across every intensity
// timeframe, reading it at the frame it pairs with rather than at the intensity's own.
// dim5 is T=2,C=3,Z=4; dim3_mask is its ZYX (T=1) segmentation of one ROI (label 1 over
// z=all, y in [1,5), x in [1,7) = 4*4*6 voxels). What this discriminates: a prescan that
// addresses the mask at the intensity's timeframe asks a TIFF mask for an IFD past its last
// plane, and TIFFSetDirectory throws uncaught, taking the process down (0xC0000409). A Zarr
// mask has no T axis to over-index, so only TIFF shows it.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITIMEFRAME_MASK_PRESCAN_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();
	ASSERT_TRUE(fs::exists(mp)) << mp.string();

	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)(4 * 4 * 6));   // z=4 * y=4 * x=6
}

// P3 (positive): the crash's regression above covers the PRESCAN (for_each_voxel); this covers
// the FEATURIZE facade. ImageLoader::stream_volume_planes(c,t) forwards t as the mask timeframe, so with
// a T>1 intensity and a T=1 mask, streaming (c,1) exercises the internal mask-timeframe clamp
// (image_loader.cpp). It must not throw, must reuse the same mask across timeframes, and must
// read different intensity per timeframe. dim5 is T=2; dim3_mask is its ZYX (T=1) mask.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITIMEFRAME_MASK_FACADE_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	SlideProps p; p.fname_int = ip.string(); p.fname_seg = mp.string();
	FpImageOptions fp; ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ip.string();

	std::vector<uint32_t> vol_i3, vol_s3;
	ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, 0, vol_i3, vol_s3));
	const std::vector<uint32_t> seg_t0 = vol_s3;
	const std::vector<uint32_t> int_t0 = vol_i3;
	// timeframe 1 with a single-timeframe mask must NOT throw (clamp), reuse the mask, and
	// deliver different intensity
	std::vector<uint32_t> vol_i4, vol_s4;
	ASSERT_NO_THROW(Nyxus::assemble_streamed_volume(il, 0, 1, vol_i4, vol_s4));
	EXPECT_EQ(vol_s4, seg_t0) << "mask changed across timeframes";
	EXPECT_NE(vol_i4, int_t0) << "intensity t=1 read t=0 data";
	il.close();
}

// P2 (positive): OME-TIFF physical calibration -> SlideProps (the TIFF twin of the OME-Zarr
// calibration test, and it additionally checks the scan_slide_props propagation the Zarr test
// omits). dim5_calibrated carries PhysicalSizeX/Y=0.5, Z=2.0 micrometer in its OME-XML.
TEST(TEST_NYXUS, TEST_OMETIFF_PHYSICAL_CALIBRATION_MECHANICS) {
	fs::path cal = ometiff_data_path("dim5_calibrated.ome.tif");
	ASSERT_TRUE(fs::exists(cal)) << cal.string();

	Environment e;
	SlideProps p (cal.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_DOUBLE_EQ(p.phys_x, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_y, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_z, 2.0);
	EXPECT_EQ(p.phys_unit, "micrometer");

	// an uncalibrated OME-TIFF must default to 1.0 / no unit
	SlideProps p2 (ometiff_data_path("dim5.ome.tif").string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(p2, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_DOUBLE_EQ(p2.phys_x, 1.0);
	EXPECT_DOUBLE_EQ(p2.phys_z, 1.0);
	EXPECT_TRUE(p2.phys_unit.empty());
}

// Unit canonicalization (OME-TIFF, end-to-end through scan_slide_props): dim5_calibrated_nm
// declares X/Y in nanometer and Z in a THIRD unit (millimeter) -- 500nm==0.5um,
// 0.002mm==2.0um. Must report the SAME physX/Y/Z and a single "micrometer" unit as
// dim5_calibrated above, proving each axis converts using its OWN declared unit rather than
// X/Y's unit leaking onto Z (or vice versa).
TEST(TEST_NYXUS, TEST_OMETIFF_UNIT_CANONICALIZATION_MECHANICS) {
	fs::path cal_nm = ometiff_data_path("dim5_calibrated_nm.ome.tif");
	ASSERT_TRUE(fs::exists(cal_nm)) << cal_nm.string();

	Environment e;
	SlideProps p (cal_nm.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_DOUBLE_EQ(p.phys_x, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_y, 0.5);
	EXPECT_DOUBLE_EQ(p.phys_z, 2.0);
	EXPECT_EQ(p.phys_unit, "micrometer");
}

// N2 (negative): a <TiffData> block maps a plane to an IFD PAST the end of the file (an
// in-file overrun, distinct from a multi-file UUID). ifdForPlane returns 99, so the read must
// throw cleanly at TIFFSetDirectory -- not crash and not read a wrong plane. dim5_badifd has
// 4 z-planes; plane z3 claims IFD=99.
TEST(TEST_NYXUS, TEST_OMETIFF_BAD_IFD_THROWS_MECHANICS) {
	fs::path ds = ometiff_data_path("dim5_badifd.ome.tif");
	ASSERT_TRUE(fs::exists(ds)) << ds.string();
	SlideProps p; p.fname_int = ds.string(); p.fname_seg = "";
	FpImageOptions fp; ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ds.string();
	// z0..z2 are fine; assembling the whole volume must hit z3's bad IFD and throw, not crash
	EXPECT_ANY_THROW(il.stream_volume_planes(0, 0, [](size_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&) {}));
	il.close();
}

// N3 (negative): an all-background (all-zero) mask -> ZERO ROIs. The prescan must complete
// cleanly (no divide-by-zero, no garbage), reporting no ROI area, rather than crash.
TEST(TEST_NYXUS, TEST_OMETIFF_EMPTY_MASK_ZERO_ROIS_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_emptymask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)0) << "empty mask should yield no ROI";
}

// N4 (edge): a mask with MORE channels than the intensity (C=2 mask, C=1-effective use). The
// featurize loop iterates the intensity's channels, so the extra mask channel must simply be
// ignored (mask channel clamped to what is asked), not crash or misread. dim3_zyx (C=1) paired
// with a C=2 label mask; the ROI is identical on both mask channels.
TEST(TEST_NYXUS, TEST_OMETIFF_MASK_MORE_CHANNELS_THAN_INTENSITY_MECHANICS) {
	fs::path ip = ometiff_data_path("dim3_zyx.ome.tif");   // C=1, Z=4
	fs::path mp = ometiff_data_path("dim4_mask_c2.ome.tif"); // C=2 label mask
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;
	SlideProps p (ip.string(), mp.string());
	bool ok = false;
	ASSERT_NO_THROW(ok = Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_TRUE(ok);
	EXPECT_EQ(p.max_roi_area, (size_t)(4 * 4 * 6));   // the one ROI, read from mask channel 0
}

// detect_container_family is the single dispatch point all three loader stacks use. It
// classifies by big-extension alone, so it needs no file on disk; every TIFF flavor is one kind
// because one TIFF loader reads plain and OME-TIFF alike.
TEST(TEST_NYXUS, TEST_DETECT_CONTAINER_FAMILY_MECHANICS) {
	using Nyxus::detect_container_family;
	using Nyxus::ContainerKind;
	EXPECT_EQ(detect_container_family(ometiff_data_path("dim5.ome.tif").string()), ContainerKind::Tiff);
	EXPECT_EQ(detect_container_family(ometiff_data_path("dim3_plain.tif").string()), ContainerKind::Tiff);
	EXPECT_EQ(detect_container_family("x.TIFF"), ContainerKind::Tiff);
	EXPECT_EQ(detect_container_family("x.dcm"), ContainerKind::Dicom);
	EXPECT_EQ(detect_container_family("x.dicom"), ContainerKind::Dicom);
	EXPECT_EQ(detect_container_family("x.nii"), ContainerKind::Nifti);
	EXPECT_EQ(detect_container_family("x.nii.gz"), ContainerKind::Nifti);
	EXPECT_EQ(detect_container_family("no_such_store.zarr"), ContainerKind::OmeZarr);
	EXPECT_EQ(detect_container_family("no_such_store.ome.zarr"), ContainerKind::OmeZarr);
}

// Pyramidal OME-TIFF: every full-res plane's IFD carries downsampled levels as SubIFDs (tag
// 330), which live OUTSIDE the main IFD chain. This must not shift full-res plane addressing:
// TIFFNumberOfDirectories still returns Z (not Z*levels) and ifdForPlane -> main-chain IFD
// still lands on the full-res plane. nyxus reads level 0 only; the facade check (which also
// spans a 2x3 tile grid) asserts every full-res voxel is correct despite the SubIFDs. Z=6
// z-stack (C=1,T=1), its own encoding.
TEST(TEST_NYXUS, TEST_OMETIFF_PYRAMID_SUBIFD_FULLRES_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_multitile_facade_volume("dim5_pyramid.ome.tif", 1, 1, 6, 32, 48));
}

// Non-canonical <TiffData> plane->IFD mapping: dim5_reordered stores its planes in REVERSED
// IFD order and declares the mapping via per-plane <TiffData IFD=..> blocks. A reader that
// ignores TiffData and assumes contiguous-from-IFD-0 order reads the reversed plane's pixels;
// honoring the map reads correctly. The stream loops Z through loadTileFromFile -> ifdForPlane
// (both loader stacks route here), so the facade check asserts the right plane per (z,c).
// T=1,C=2,Z=3 (its own encoding). This is the OME-TIFF counterpart to what bioformats emits.
TEST(TEST_NYXUS, TEST_OMETIFF_TIFFDATA_REORDERED_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_multitile_facade_volume("dim5_reordered.ome.tif", 1, 2, 3, 6, 8));
}

// 4D (rank-4): time-only and channel-only.
TEST(TEST_NYXUS, TEST_OMETIFF_4D_TZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_addressing("dim4_tzyx.ome.tif", 2, 1, 4));
	ASSERT_NO_THROW (assert_raw_ometiff_addressing("dim4_tzyx.ome.tif", 2, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMETIFF_4D_CZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_addressing("dim4_czyx.ome.tif", 1, 3, 4));
	ASSERT_NO_THROW (assert_raw_ometiff_addressing("dim4_czyx.ome.tif", 1, 3, 4));
}

// End-to-end through the wired volumetric consumer (scan_trivial_wholevolume).
TEST(TEST_NYXUS, TEST_OMETIFF_WHOLEVOLUME_CONSUMER_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_wholevolume_consumer("dim3_zyx.ome.tif", 4));
}

// Wired consumer reads the correct plane for every (channel, timeframe), not just (0,0).
TEST(TEST_NYXUS, TEST_OMETIFF_WHOLEVOLUME_CONSUMER_CT_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_wholevolume_consumer_ct("dim5.ome.tif", 2, 3, 4));
}

// TILED multi-plane OME-TIFF: the tile loaders map (z,c,t)->IFD (distinct from strip loaders).
TEST(TEST_NYXUS, TEST_OMETIFF_TILED_ADDRESSING_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_tiled_addressing_mechanics());
}
// Facade whole-volume assembly over the TILED path (open() routes tiled TIFF -> tile loader).
TEST(TEST_NYXUS, TEST_OMETIFF_TILED_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_facade_volume("dim5_tiled.ome.tif", 2, 3, 4));
}
// Multi-TILE planes (2x3 grid of 16x16 tiles): dim5_tiled above has ONE tile per plane, so
// it passes even when only tile (0,0) is read. This is the OME-TIFF counterpart of
// TEST_OMEZARR_MULTICHUNK_FACADE_VOLUME_MECHANICS. Covers ImageLoader::assemble_tile_layer...
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITILE_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_multitile_facade_volume("dim5_multitile.ome.tif", 1, 2, 2, 32, 48));
}
// ...and RawImageLoader::for_each_voxel (the prescan), which had the same single-tile bug.
// Encoded values run 1..6144 over all (c,t); whole-slide, so the ROI is the whole volume.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTITILE_PRESCAN_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5_multitile.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 6144.0);
	EXPECT_EQ(p.max_roi_area, (size_t)(48 * 32 * 2));
}

// PARTIAL edge tiles: 40x24 plane / 16 -> 3x2 tiles, last row-tile 8 tall, last col-tile 8
// wide. Every multi-tile fixture above has plane dims that are exact multiples of the tile
// size, so the validH/validW seam clamp -- min(tileDim, fullDim-offset) -- had never run.
// Reading the exact value at every voxel checks the partial tiles are copied without garbage
// past the seam and without over-reading the tile buffer. Encoded 1..1920 over (c in 0,1).
TEST(TEST_NYXUS, TEST_OMETIFF_ODDTILE_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_multitile_facade_volume("dim5_oddtile.ome.tif", 1, 2, 1, 40, 24));
}
// The prescan (for_each_voxel) walks the same partial grid; its slide min/max and ROI area
// would be wrong if a partial tile were mis-clamped (a too-large validH double-counts voxels).
TEST(TEST_NYXUS, TEST_OMETIFF_ODDTILE_PRESCAN_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5_oddtile.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1920.0);	// 1 + (((0*2+1)*1+0)*40+39)*24+23
	EXPECT_EQ(p.max_roi_area, (size_t)(24 * 40 * 1));
}

// Facade whole-volume assembly (the streamed Z-planes stacked into one X*Y*Z buffer).
TEST(TEST_NYXUS, TEST_OMETIFF_FACADE_VOLUME_3D_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_facade_volume("dim3_zyx.ome.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMETIFF_FACADE_VOLUME_5D_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_facade_volume("dim5.ome.tif", 2, 3, 4));
}

// Lower-rank OME-TIFF.
TEST(TEST_NYXUS, TEST_OMETIFF_3D_ZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_addressing("dim3_zyx.ome.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_RAW_OMETIFF_3D_ZYX_MECHANICS) {
	ASSERT_NO_THROW (assert_raw_ometiff_addressing("dim3_zyx.ome.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_OMETIFF_2D_YX_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_addressing("dim2_yx.ome.tif", 1, 1, 1));
}

// Plain multi-page TIFF (no OME-XML): the legacy page=Z fallback must still work.
TEST(TEST_NYXUS, TEST_OMETIFF_PLAIN_MULTIPAGE_FALLBACK_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_addressing("dim3_plain.tif", 1, 1, 4));
}
TEST(TEST_NYXUS, TEST_RAW_OMETIFF_PLAIN_MULTIPAGE_FALLBACK_MECHANICS) {
	ASSERT_NO_THROW (assert_raw_ometiff_addressing("dim3_plain.tif", 1, 1, 4));
}

// Negative: out-of-range channel/timeframe through the whole-volume facade must throw.
TEST(TEST_NYXUS, TEST_OMETIFF_STREAM_VOLUME_OUT_OF_RANGE_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_stream_volume_out_of_range_mechanics());
}

// A single-channel mask pairs with every intensity channel, and is read within its own extent.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTICHANNEL_MASK_PAIRING_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_multichannel_mask_pairing_mechanics());
}

// Strip loaders advertise the OME C/T extents; the plain (non-OME) multi-page TIFF
// keeps C=T=1 (its pages are Z-slices, not channels/timeframes).
TEST(TEST_NYXUS, TEST_OMETIFF_CT_COUNTS_MECHANICS) {
	ASSERT_NO_THROW (assert_ometiff_ct_counts("dim5.ome.tif", 2, 3, 4));
	ASSERT_NO_THROW (assert_ometiff_ct_counts("dim4_tzyx.ome.tif", 2, 1, 4));
	ASSERT_NO_THROW (assert_ometiff_ct_counts("dim4_czyx.ome.tif", 1, 3, 4));
	ASSERT_NO_THROW (assert_ometiff_ct_counts("dim3_zyx.ome.tif", 1, 1, 4));
	ASSERT_NO_THROW (assert_ometiff_ct_counts("dim3_plain.tif", 1, 1, 4));
}

// Negative: out-of-range Z/C/T plane index must throw.
TEST(TEST_NYXUS, TEST_OMETIFF_OUT_OF_RANGE_THROWS_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_out_of_range_throws_mechanics());
}

// Illegal / adversarial: RGB / corrupt / missing files must be rejected cleanly.
TEST(TEST_NYXUS, TEST_OMETIFF_MALFORMED_THROWS_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_malformed_throws_mechanics());
}

// Planes spanning a 3x2 tile grid with partial edge tiles, through the facade and the prescan.
TEST(TEST_NYXUS, TEST_OMETIFF_ODDTILE_ZSTACK_FACADE_VOLUME_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_oddtile_zstack_facade_volume_mechanics());
}
TEST(TEST_NYXUS, TEST_OMETIFF_ODDTILE_ZSTACK_PRESCAN_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_oddtile_zstack_prescan_mechanics());
}

// The 3D prescan covers every Z-plane of a multi-page OME-TIFF.
TEST(TEST_NYXUS, TEST_OMETIFF_ZSTACK_PRESCAN_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_zstack_prescan_mechanics());
}

// More than one channel or timepoint is refused in 2D rather than featurized at C=0, T=0 alone.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTICHANNEL_TIMEPOINT_REFUSED_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_multichannel_timepoint_refused_mechanics());
}

// The streamed volume of (c, t) pairs mask frames N:N and 1:N by one rule.
TEST(TEST_NYXUS, TEST_OMETIFF_MASK_FRAME_PAIRING_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_mask_frame_pairing_mechanics());
}

// Page numbers past 65535 select their own page, not page (n mod 65536).
TEST(TEST_NYXUS, TEST_OMETIFF_DIRECTORY_BEYOND_16BIT_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_directory_beyond_16bit_mechanics());
}

// The 3D prescan measures every (channel, timeframe) plane, and every frame of an N:N mask.
TEST(TEST_NYXUS, TEST_OMETIFF_SEGMENTED_TIMESERIES_PRESCAN_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_segmented_timeseries_prescan_mechanics());
}

// open() starts a pair at plane (0,0), whatever plane the previous pair was read at.
TEST(TEST_NYXUS, TEST_OMETIFF_REOPEN_RESETS_PLANE_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_reopen_resets_plane_mechanics());
}

// A strip TIFF larger than one tile is read tile by tile, by the raw loader and both prescans.
TEST(TEST_NYXUS, TEST_OMETIFF_PLAIN_STRIP_BEYOND_ONE_TILE_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_plain_strip_beyond_one_tile_mechanics());
}

// A plain TIFF's depth is its run of full-size directories, tiled or not.
TEST(TEST_NYXUS, TEST_OMETIFF_PLAIN_DEPTH_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_plain_depth_mechanics());
}

// Half-float TIFF samples are read at their own 2-byte width, in strips and in tiles.
TEST(TEST_NYXUS, TEST_OMETIFF_PLAIN_HALF_FLOAT_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_plain_half_float_mechanics());
}

// A multi-file OME-TIFF is refused; one whose blocks name its own UUID is read.
TEST(TEST_NYXUS, TEST_OMETIFF_MULTI_FILE_REFUSED_MECHANICS) {
	ASSERT_NO_THROW (test_ometiff_multi_file_refused_mechanics());
}


// The 3D prescan scans the whole volume of every (channel, timeframe), and takes each ROI's
// geometry from the first pass alone. dim5.ome.tif is C=3,T=2,Z=4,Y=6,X=8 encoding values
// 1..1152, so the slide intensity range is exactly [1, 1152] and the ROI area is one volume's
// worth. What this discriminates: a prescan that loads a single tile and then indexes W*H*D
// voxels off that one-plane buffer reads out of bounds (a range of 0..44,465 of garbage); one
// that covers only (c0,t0) reports 1..192, which under-sizes every intensity-indexed buffer
// for c>0; and one that accumulates geometry across passes multiplies the area by
// n_channels*n_timeframes.
TEST(TEST_NYXUS, TEST_3D_PRESCAN_SLIDE_RANGE_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	ASSERT_TRUE(fs::exists(ip)) << ip.string();

	Environment e;
	SlideProps p (ip.string(), "");		// whole-slide: no mask
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_EQ(p.inten_channels, (size_t)3);
	EXPECT_EQ(p.inten_time, (size_t)2);
	EXPECT_EQ(p.volume_d, (size_t)4);
	// the full encoded range across ALL (c,t) -- not garbage, and not just (c0,t0)'s 1..192
	EXPECT_DOUBLE_EQ(p.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(p.max_preroi_inten, 1152.0);
	// geometry from the first pass only: one whole-slide ROI of exactly X*Y*Z voxels
	EXPECT_EQ(p.max_roi_area, (size_t)(8 * 6 * 4));
}

// The 3D whole-volume reduce path: reduce_trivial_3d_wholevolume reaches
// D3_VoxelIntensityFeatures through a call that carries the Dataset, as the segmented path's
// reduce() does. What this discriminates: routing it through the 2-arg calculate() instead
// hits a stub that throws "illegal call", so every 3D whole-volume featurization dies before
// writing a row -- and no segmented test covers it. This mirrors featurize_wholevolume()'s
// vROI setup, then reduces.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_REDUCE_MECHANICS) {
	fs::path ds = ometiff_data_path("dim3_zyx.ome.tif");	// 3D X8 Y6 Z4
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	Environment e;
	// enable the 3D intensity features so the reduce actually runs them
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);

	// prescan the slide (whole-volume => no mask)
	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ds.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
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

	// WHAT THIS DISCRIMINATES: the 2-arg calculate() throws "illegal call of D3_VoxelIntensityFeatures::calculate"
	ASSERT_NO_THROW(Nyxus::reduce_trivial_3d_wholevolume(e, vroi));

	// and it must actually produce values (MAX >= MIN, both finite)
	double vmin = vroi.get_fvals((int)Nyxus::Feature3D::MIN)[0];
	double vmax = vroi.get_fvals((int)Nyxus::Feature3D::MAX)[0];
	EXPECT_GE(vmax, vmin);
	EXPECT_GT(vmax, 0.0);
	ilo.close();
}

// An OVERSIZED whole volume (here a multi-page OME-TIFF) featurizes out-of-core through
// workflow_3d_whole.cpp's oversized branch (populate_3d_voxel_cloud/run_3d_ooc_features) and
// must produce the SAME feature values as the in-RAM (fitting) run of the identical file -- not
// fail, and not emit a zero row. A whole-4D-in-one-read format (NIfTI) has no bounded streaming
// path -- see TEST_3D_WHOLEVOLUME_UNSTREAMABLE_FORMAT_FAILS_LOUDLY_MECHANICS.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_OVERSIZED_STREAMS_OOC_MECHANICS) {
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
		// Cover every streaming 3D texture family so the out-of-core paths are held byte-exact
		// against the in-RAM path (not just intensity/surface/GLCM)
		e.theFeatureSet.enableFeatures(D3_GLDM_feature::featureset);
		e.theFeatureSet.enableFeatures(D3_GLRLM_feature::featureset);
		e.theFeatureSet.enableFeatures(D3_GLSZM_feature::featureset);
		e.theFeatureSet.enableFeatures(D3_GLDZM_feature::featureset);
		e.theFeatureSet.enableFeatures(D3_NGLDM_feature::featureset);
		e.theFeatureSet.enableFeatures(D3_NGTDM_feature::featureset);
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

// The whole-volume ROI takes the extent the prescan recorded, which is already on the resampled
// grid: scan_slide_props resolves the same spacing and scales each ROI's box before recording
// max_roi_w/h/d. What this discriminates: scaling that box a second time squares the resampling
// (with --aniso-z 4, a Z extent of 14 becomes 54), and the oversized check built on it then sends
// a volume that fits comfortably in RAM down the out-of-core path or rejects it outright.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_ANISOTROPIC_PRESCAN_BOX_MECHANICS) {
	// a volume big enough that the two footprints below straddle a whole megabyte, the granularity
	// Environment::set_ram_limit takes
	const uint32_t W = 200, H = 200, D = 8;
	fs::path ds = fs::temp_directory_path() / "nyxus_wv_aniso_box.tif";
	{
		TIFF* t = TIFFOpen(ds.string().c_str(), "w");
		ASSERT_NE(t, nullptr) << ds.string();
		std::vector<uint16_t> row(W);
		for (uint32_t z = 0; z < D; ++z)
		{
			TIFFSetField(t, TIFFTAG_IMAGEWIDTH, W);
			TIFFSetField(t, TIFFTAG_IMAGELENGTH, H);
			TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 16);
			TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
			TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1u);
			for (uint32_t y = 0; y < H; ++y)
			{
				for (uint32_t x = 0; x < W; ++x)
					row[x] = (uint16_t)(1 + (x % 97) + (y % 89) + z);
				ASSERT_EQ(TIFFWriteScanline(t, row.data(), y, 0), 1);
			}
			ASSERT_EQ(TIFFWriteDirectory(t), 1);
		}
		TIFFClose(t);
	}

	Environment e;
	AnisotropyOptions aniso_z4;
	aniso_z4.set_aniso_z(4.0);
	ASSERT_TRUE(aniso_z4.customized());

	SlideProps p (ds.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(p, 3, aniso_z4, false, e.fpimageOptions, e.resultOptions.need_annotation()));
	EXPECT_EQ(p.max_roi_w, (size_t) W);
	EXPECT_EQ(p.max_roi_h, (size_t) H);
	EXPECT_GT(p.max_roi_d, (size_t) D) << "the prescan records the resampled Z extent";

	// the footprint of the box the prescan's extent gives, and of that box resampled a second
	// time -- the RAM limit goes between them, so only a doubly-resampled box reads as oversized
	LR once(1), twice(1);
	Nyxus::init_wholevolume_vroi (p, 0, once);
	Nyxus::init_wholevolume_vroi (p, 0, twice);
	twice.aabb.apply_anisotropy (1.0, 1.0, 4.0);
	const size_t f_once = once.get_ram_footprint_estimate_3D (1),
		f_twice = twice.get_ram_footprint_estimate_3D (1);
	ASSERT_LT(f_once, f_twice);

	fs::path outdir = fs::temp_directory_path() / "nyxus_wv_aniso_box_test";
	fs::remove_all(outdir); fs::create_directories(outdir);

	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);
	ASSERT_TRUE(e.theFeatureMgr.compile());
	e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
	ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
	e.compile_feature_settings();
	e.anisoOptions = aniso_z4;
	e.output_dir = outdir.string();
	e.set_verbosity_level(1);
	const size_t limit_mb = ((f_once + f_twice) / 2) / (1024 * 1024);
	ASSERT_GT(limit_mb * 1024 * 1024, f_once);
	ASSERT_LT(limit_mb * 1024 * 1024, f_twice) << "the two footprints must straddle a megabyte boundary";
	ASSERT_TRUE(e.set_ram_limit (limit_mb));

	std::vector<std::string> ifiles{ ds.string() };
	testing::internal::CaptureStdout();
	auto [ok, erm] = Nyxus::processDataset_3D_wholevolume(e, ifiles, 1, Nyxus::SaveOption::saveCSV, outdir.string());
	std::string out = testing::internal::GetCapturedStdout();

	EXPECT_TRUE(ok) << (erm ? *erm : std::string("(no error message)"));
	EXPECT_EQ(out.find("oversized whole volume"), std::string::npos)
		<< "this volume fits the limit; only a box resampled a second time reads as oversized";
	fs::remove_all(outdir);
	std::error_code ec;
	fs::remove(ds, ec);
}

// The out-of-core voxel cloud is resampled on a non-cubic grid, like the in-RAM scan: both must
// hold the same voxels at the same virtual coordinates, and the ROI's extent and voxel count must
// describe that resampled cloud. What this discriminates: a streaming pass that ignores the
// spacing writes the physical cloud, so one slide mixes resampled in-RAM ROIs with unresampled
// oversized ones.
TEST(TEST_NYXUS, TEST_3D_OOC_ANISOTROPIC_CLOUD_MATCHES_IN_RAM_MECHANICS) {
	fs::path ds = ometiff_data_path("dim3_zyx.ome.tif");	// 3D X8 Y6 Z4
	ASSERT_TRUE(fs::exists(ds)) << ds.string();
	const double ax = 1.0, ay = 1.0, az = 2.0;

	SlideProps p;
	p.fname_int = ds.string();
	p.fname_seg = "";
	FpImageOptions fp;
	ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ds.string();

	// in-RAM: the resampled cloud phase 2 caches
	LR ram(1);
	ASSERT_TRUE(Nyxus::scan_trivial_wholevolume_anisotropic(ram, ds.string(), il, ax, ay, az, 0, 0));

	// out-of-core: the same volume streamed to the disk-backed cloud
	LR ooc(1);
	ooc.aabb.init_from_whd (8, 6, 4);
	ooc.aux_area = 8 * 6 * 4;
	ASSERT_NO_THROW(Nyxus::populate_3d_voxel_cloud(il, ooc, 0, 0, /*wholevolume=*/ true, ax, ay, az, ds.string(), ""));

	ASSERT_EQ(ooc.raw_voxels_NT.size(), ram.raw_pixels_3D.size());
	EXPECT_EQ(ooc.raw_voxels_NT.size(), (size_t)(8 * 6 * 8));	// Z resampled 2x
	EXPECT_EQ(ooc.aux_area, (unsigned int) ram.raw_pixels_3D.size());
	EXPECT_EQ(ooc.aabb.get_z_depth(), (StatsInt)8);
	EXPECT_EQ(ooc.aabb.get_width(), (StatsInt)8);

	for (size_t i = 0; i < ram.raw_pixels_3D.size(); i++)
	{
		const Pixel3& a = ram.raw_pixels_3D[i];
		const Pixel3 b = ooc.raw_voxels_NT.get_at(i);
		ASSERT_EQ(b.x, a.x); ASSERT_EQ(b.y, a.y); ASSERT_EQ(b.z, a.z);
		ASSERT_EQ(b.inten, a.inten) << "voxel " << i << " at (" << a.x << "," << a.y << "," << a.z << ")";
	}
	ooc.raw_voxels_NT.clear();
	il.close();
}

// A whole volume whose loader delivers the ENTIRE X*Y*Z*T blob in one read (NIfTI) cannot be
// streamed within a bounded footprint -- a tile layer of it is the whole cube -- so an oversized
// one must fail loudly rather than allocate the volume out-of-core and OOM, and must write no row.
// ram_limit=0 forces oversized regardless of the fixture's actual size.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_UNSTREAMABLE_FORMAT_FAILS_LOUDLY_MECHANICS) {
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
	e.output_dir = outdir.string();

	std::vector<std::string> ifiles{ ds.string() };
	auto [ok, erm] = Nyxus::processDataset_3D_wholevolume(e, ifiles, 1, Nyxus::SaveOption::saveCSV, outdir.string());

	EXPECT_FALSE(ok) << "an oversized whole-4D-in-one-read volume has no bounded streaming path and must fail loudly";

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

// Whether one read delivers the whole volume is a question about the Z and T axes together: the
// NIfTI loader hands back the entire X*Y*Z*T blob, so a time series of single-plane volumes is as
// unstreamable as a deep one. What this discriminates: a guard that compares depths alone -- or
// short-circuits on a depth of 1 -- reads such a file as bounded, and the out-of-core path then
// allocates every time frame at once, which is the allocation it exists to avoid.
TEST(TEST_NYXUS, TEST_3D_STREAMS_BOUNDED_COUNTS_TIME_AXIS_MECHANICS) {
	const int64_t W = 8, H = 6, Z = 1, T = 5;
	fs::path ds = fs::temp_directory_path() / "nyxus_nifti_1z_5t.nii";
	std::error_code ec;
	fs::remove(ds, ec);
	{
		int64_t dims[8] = { 4, W, H, Z, T, 1, 1, 1 };	// ndim, nx, ny, nz, nt, ...
		nifti_image* nim = nifti_make_new_nim (dims, DT_UINT16, 1);
		ASSERT_NE(nim, nullptr);
		uint16_t* v = (uint16_t*) nim->data;
		for (int64_t i = 0; i < nim->nvox; i++)
			v[i] = (uint16_t) (1 + (i % 251));
		ASSERT_EQ(nifti_set_filenames (nim, ds.string().c_str(), 0, 1), 0);
		nifti_image_write (nim);
		nifti_image_free (nim);
	}
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	SlideProps p;
	p.fname_int = ds.string();
	p.fname_seg = "";
	FpImageOptions fp;
	ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ds.string();

	EXPECT_EQ(il.get_full_depth(), (size_t) Z) << "the fixture is a single Z-plane";
	EXPECT_EQ(il.get_inten_time(), (size_t) T);
	EXPECT_FALSE(il.streams_bounded())
		<< "one read of this file delivers all " << (Z * T) << " planes, which is the whole volume";

	// and the number a refusal reports is that read, not the frame it belongs to: this loader
	// hands back every frame at once, so it is Z*T, which is what the message must say
	size_t planes = 0;
	bool of_mask = true;
	ASSERT_TRUE(il.unstreamable_read (planes, of_mask));
	EXPECT_EQ(planes, (size_t) (Z * T)) << "one read of this file is every plane it holds";
	EXPECT_FALSE(of_mask) << "the intensity loader is the unstreamable one here";

	il.close();
	fs::remove(ds, ec);
}

// streams_bounded() fails on either half of the pair, and the refusal has to name the half it
// describes: the file to re-chunk is the one whose read is unbounded, and pointing at the other
// sends the reader to a file that is fine. A mask can be that half -- ImageLoader::open() requires
// the pair to agree on width, height, depth and tile geometry, but not on the length of the time
// series, so a single-frame intensity volume pairs with a mask that carries several.
TEST(TEST_NYXUS, TEST_3D_UNSTREAMABLE_READ_NAMES_THE_MASK_MECHANICS) {
	const int64_t W = 8, H = 6;
	fs::path ints = fs::temp_directory_path() / "nyxus_nifti_mask_int.nii",
		segs = fs::temp_directory_path() / "nyxus_nifti_mask_seg.nii";
	std::error_code ec;
	fs::remove(ints, ec); fs::remove(segs, ec);

	// one plane, one frame: a read of it is a single plane, so it streams
	// one plane, five frames: a read of it is every frame at once, so it does not
	auto write_nii = [](const fs::path& f, int64_t nt, uint16_t base)
	{
		int64_t dims[8] = { 4, W, H, 1, nt, 1, 1, 1 };
		nifti_image* nim = nifti_make_new_nim (dims, DT_UINT16, 1);
		ASSERT_NE(nim, nullptr);
		uint16_t* v = (uint16_t*) nim->data;
		for (int64_t i = 0; i < nim->nvox; i++)
			v[i] = (uint16_t) (base + (i % 7));
		ASSERT_EQ(nifti_set_filenames (nim, f.string().c_str(), 0, 1), 0);
		nifti_image_write (nim);
		nifti_image_free (nim);
	};
	write_nii (ints, 1, 1);
	write_nii (segs, 5, 1);
	ASSERT_TRUE(fs::exists(ints)) << ints.string();
	ASSERT_TRUE(fs::exists(segs)) << segs.string();

	SlideProps p;
	p.fname_int = ints.string();
	p.fname_seg = segs.string();
	FpImageOptions fp;
	ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ints.string() << " / " << segs.string();

	size_t planes = 0;
	bool of_mask = false;
	ASSERT_TRUE(il.unstreamable_read (planes, of_mask))
		<< "the mask delivers all 5 of its frames in one read";
	EXPECT_TRUE(of_mask) << "the intensity volume is a single plane and streams; the mask is the one that does not";
	EXPECT_EQ(planes, (size_t) 5) << "and the count describes the mask's read, not the intensity's";
	EXPECT_FALSE(il.streams_bounded());

	il.close();
	fs::remove(ints, ec); fs::remove(segs, ec);
}

// A tile grid walk bounds its row by the number of tile ROWS and its column by the number of tile
// COLUMNS, which is the order load_tile (row, col) takes and the order the loaders' accessors
// report. What this discriminates: a walk that takes the two counts the other way round asks for
// tiles that do not exist on any grid that is not square -- here 3 columns and 5 rows -- and
// load_tile refuses them, so the whole-slide scan returns a truncated cloud or no cloud at all.
// A square grid hides it completely, which is why this fixture is deliberately oblong.
TEST(TEST_NYXUS, TEST_2D_WHOLESLIDE_NONSQUARE_TILE_GRID_MECHANICS) {
	const uint32_t W = 48, H = 80, TILE = 16;	// 3 tiles across, 5 down
	fs::path ds = fs::temp_directory_path() / "nyxus_nonsquare_tiles.tif";
	std::error_code ec;
	fs::remove(ds, ec);

	auto enc = [](uint32_t x, uint32_t y) { return (uint16_t) (1 + (y * W + x) % 4096); };
	write_tiled_plane_u16 (ds, W, H, TILE, enc);
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	SlideProps p;
	p.fname_int = ds.string();
	p.fname_seg = "";
	FpImageOptions fp;
	ImageLoader il;
	ASSERT_TRUE(il.open(p, fp)) << ds.string();

	// the grid this rests on
	ASSERT_EQ(il.get_num_tiles_hor(), (size_t) 3) << "tiles across a row";
	ASSERT_EQ(il.get_num_tiles_vert(), (size_t) 5) << "tiles down a column";

	LR vroi (1);
	ASSERT_TRUE(Nyxus::scan_trivial_wholeslide (vroi, ds.string(), il))
		<< "every tile of the grid must be readable";
	EXPECT_EQ(vroi.raw_pixels.size(), (size_t) (W * H)) << "the cloud holds the whole slide";

	// and the pixels are where they belong, corners included
	std::map<std::pair<size_t, size_t>, uint32_t> got;
	for (auto& px : vroi.raw_pixels)
		got[{ (size_t) px.x, (size_t) px.y }] = px.inten;
	for (auto xy : { std::make_pair(0u, 0u), std::make_pair(W - 1, 0u),
					 std::make_pair(0u, H - 1), std::make_pair(W - 1, H - 1),
					 std::make_pair(17u, 33u) })
	{
		auto it = got.find({ xy.first, xy.second });
		ASSERT_NE(it, got.end()) << "missing pixel (" << xy.first << "," << xy.second << ")";
		EXPECT_EQ(it->second, (uint32_t) enc (xy.first, xy.second))
			<< "at (" << xy.first << "," << xy.second << ")";
	}


	// the prescan walks the same grid through the raw loader stack, whose tile counts must agree
	// with the tile loaders' -- the two report the same two numbers, and a walk written against
	// one has to hold against the other
	Environment e;
	SlideProps sp (ds.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props (sp, 2, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions,
		e.resultOptions.need_annotation())) << "the prescan must read every tile of the grid too";
	EXPECT_DOUBLE_EQ(sp.min_preroi_inten, 1.0);
	EXPECT_DOUBLE_EQ(sp.max_preroi_inten, (double) ((H - 1) * W + (W - 1) + 1))
		<< "the largest value sits in the far corner, which only a complete walk reaches";
	il.close();
	fs::remove(ds, ec);
}

// The 2.5D (layoutA) path: both passes over a stack of per-Z slice files.
TEST(TEST_NYXUS, TEST_3D_LAYOUTA_SEGMENTED_PASSES_MECHANICS) {
	ASSERT_NO_THROW (test_3d_layouta_segmented_passes_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_LAYOUTA_NONSQUARE_TILE_GRID_MECHANICS) {
	ASSERT_NO_THROW (test_3d_layouta_nonsquare_tile_grid_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_LAYOUTA_UNREADABLE_PLANE_FAILS_THE_PASS_MECHANICS) {
	ASSERT_NO_THROW (test_3d_layouta_unreadable_plane_fails_the_pass_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_LAYOUTA_ANISOTROPIC_TILE_INDEX_MECHANICS) {
	ASSERT_NO_THROW (test_3d_layouta_anisotropic_tile_index_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_LAYOUTA_ANISOTROPIC_RESAMPLING_MECHANICS) {
	ASSERT_NO_THROW (test_3d_layouta_anisotropic_resampling_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_LAYOUTA_ANISOTROPIC_CLOUD_FITS_ITS_AABB_MECHANICS) {
	ASSERT_NO_THROW (test_3d_layouta_anisotropic_cloud_fits_its_aabb_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_LAYOUTA_ANISOTROPIC_VANISHED_ROI_IS_REFUSED_MECHANICS) {
	ASSERT_NO_THROW (test_3d_layouta_anisotropic_vanished_roi_is_refused_mechanics());
}



// Guards and invariants the value-parity tests cannot reach.
TEST(TEST_NYXUS, TEST_HISTOGRAM_UNIQUES_RESET_MECHANICS) {
	ASSERT_NO_THROW (test_histogram_uniques_reset_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_OOC_VOLUME_GUARDS_MECHANICS) {
	ASSERT_NO_THROW (test_3d_ooc_volume_guards_mechanics());
}
TEST(TEST_NYXUS, TEST_TIFF_SAMPLE_UNSUPPORTED_THROWS_MECHANICS) {
	ASSERT_NO_THROW (test_tiff_sample_unsupported_throws_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_VOXEL_CLOUD_SEEK_BEYOND_2GB_MECHANICS) {
	ASSERT_NO_THROW (test_3d_voxel_cloud_seek_beyond_2gb_mechanics());
}
TEST(TEST_NYXUS, TEST_HISTOGRAM_IS_BOUNDED_BY_LEVELS_MECHANICS) {
	ASSERT_NO_THROW (test_histogram_is_bounded_by_levels_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_OOC_REASON_EMPTY_WHEN_BOUNDED_MECHANICS) {
	ASSERT_NO_THROW (test_3d_ooc_reason_empty_when_bounded_mechanics());
}
TEST(TEST_NYXUS, TEST_3D_WV_THREAD_UNOPENABLE_SLIDE_MECHANICS) {
	ASSERT_NO_THROW (test_3d_wv_thread_unopenable_slide_mechanics());
}
TEST(TEST_NYXUS, TEST_2D_WSI_THREAD_UNOPENABLE_SLIDE_MECHANICS) {
	ASSERT_NO_THROW (test_2d_wsi_thread_unopenable_slide_mechanics());
}

TEST(TEST_NYXUS, TEST_3D_TRIVIAL_ROIS_SCAN_FAILURE_PROPAGATES_MECHANICS) {
	ASSERT_NO_THROW (test_3d_trivial_rois_scan_failure_propagates_mechanics());
}







// An oversized segmented ROI whose loader cannot be streamed must fail the run, not skip the ROI:
// its features are still at the zeros initialize_fvals() left, so carrying on writes a row that
// reports measurements nobody made, with nyxus exiting 0. The Python binding raises here; this is
// the CLI path, which has only the return value to say so. ram_limit=0 forces every ROI oversized.
TEST(TEST_NYXUS, TEST_3D_SEGMENTED_UNSTREAMABLE_FORMAT_FAILS_THE_RUN_MECHANICS) {
	fs::path p(__FILE__);
	fs::path dir = p.parent_path() / "data" / "nifti";
	fs::path ints = dir / "compat_int" / "compat_int_mri.nii",
		segs = dir / "compat_seg" / "compat_seg_liver.nii";
	if (! fs::exists(ints) || ! fs::exists(segs))
		GTEST_SKIP() << "NIfTI compat fixtures not present: " << ints.string();

	fs::path outdir = fs::temp_directory_path() / "nyxus_seg_nifti_ooc_test";
	fs::remove_all(outdir); fs::create_directories(outdir);

	Environment e;
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);
	ASSERT_TRUE(e.theFeatureMgr.compile());
	e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
	ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
	e.compile_feature_settings();
	ASSERT_TRUE(e.set_ram_limit(0));	// every ROI reads as oversized, whatever its real footprint
	e.output_dir = outdir.string();

	std::vector<Nyxus::Imgfile3D_layoutA> ifiles{ Nyxus::Imgfile3D_layoutA (ints.string()) },
		mfiles{ Nyxus::Imgfile3D_layoutA (segs.string()) };
	int rc = Nyxus::processDataset_3D_segmented (e, ifiles, mfiles, 1, Nyxus::SaveOption::saveCSV, outdir.string());

	EXPECT_NE(rc, 0) << "an oversized ROI that could not be streamed must fail the run";

	size_t datarows = 0;
	for (auto& de : fs::directory_iterator(outdir))
		if (de.path().extension() == ".csv")
		{
			std::ifstream f(de.path()); std::string ln; size_t n = 0;
			while (std::getline(f, ln)) if (!ln.empty()) ++n;
			if (n) datarows += n - 1;	// minus header
		}
	EXPECT_EQ(datarows, (size_t) 0) << "and must write no all-zero row for the ROI it could not measure";
	fs::remove_all(outdir);
}
// Regression (found by running nyxus under a hard memory cap): the whole-volume oversized check
// must use the 3D footprint estimator (W*H*D for the image cube), not the 2D one (W*H). The 2D
// estimator ignores depth, so it under-counted a volume's memory by ~depth x, let oversized
// volumes slip through the "trivial" path, and they OOM-crashed under a real memory limit even
// with a matching --ramLimit. This pins that the 3D estimator accounts for depth (the 2D one
// does not) so featurize_wholevolume's switch to get_ram_footprint_estimate_3D stays correct.
TEST(TEST_NYXUS, TEST_3D_RAM_FOOTPRINT_COUNTS_DEPTH_MECHANICS) {
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
TEST(TEST_NYXUS, TEST_RAM_FOOTPRINT_ESTIMATE_ZERO_ROIS_DOES_NOT_UNDERFLOW_MECHANICS) {
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

TEST(TEST_NYXUS, TEST_3D_OOC_GUARD_REJECTS_UNSUPPORTED_FEATURE_MECHANICS) {
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

// separatecsv derives ONE output path per slide, while the CSV sinks are invoked once per
// (channel, timeframe) plane, so every plane of a slide appends to that one file and the
// t_index/c_index columns tell them apart. What this discriminates: a sink that opens the
// path with mode "w" per plane truncates the file each time, leaving only the last plane
// -- one data row where this fixture's two channels should give two.
TEST(TEST_NYXUS, TEST_CSV_MULTICHANNEL_NO_OVERWRITE_MECHANICS) {
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
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
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

// Physical-calibration logic (negative + positive). resolve_slide_anisotropy
// must NOT engage the anisotropic (resampling) path unless it's genuinely warranted:
//   - flag off                      -> false, (1,1,1)   even with anisotropic spacing
//   - degenerate spacing (a 0 axis) -> false, (1,1,1)   (guarded, no div-by-zero)
//   - isotropic spacing (all equal) -> false, (1,1,1)   (nothing to correct)
//   - out-of-range slide index      -> false, (1,1,1)   (no OOB read)
//   - real anisotropic spacing      -> true,  ratios normalized so min == 1
//   - explicit --aniso*             -> true,  the CLI values (win over physical)
TEST(TEST_NYXUS, TEST_RESOLVE_SLIDE_ANISOTROPY_MECHANICS) {
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

// TEST_RESOLVE_SLIDE_ANISOTROPY_MECHANICS covers the DECISION (physical spacing -> ratios). This covers
// that the resolved ratios actually RESCALE ROI geometry end-to-end: the 3D prescan's
// anisotropic branch (make_anisotropic_aabb 3-arg -> AABB::apply_anisotropy) was never
// exercised -- every other test uses make_nonanisotropic_aabb. A customized az=4 must scale the
// ROI's z-depth ~4x while leaving x/y (ax=ay=1) unchanged; without applying anisotropy the
// depth would be identical to the isotropic run. dim3_mask's ROI spans all Z (depth 4).
TEST(TEST_NYXUS, TEST_3D_ANISOTROPY_RESCALES_ROI_DEPTH_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));
	Environment e;

	SlideProps iso (ip.string(), mp.string());
	AnisotropyOptions aniso_off;                       // un-customized -> isotropic AABB
	ASSERT_FALSE(aniso_off.customized());
	ASSERT_TRUE(Nyxus::scan_slide_props(iso, 3, aniso_off, false, e.fpimageOptions, e.resultOptions.need_annotation()));

	SlideProps ani (ip.string(), mp.string());
	AnisotropyOptions aniso_z4;
	aniso_z4.set_aniso_z(4.0);                          // z 4x thicker
	ASSERT_TRUE(aniso_z4.customized());
	ASSERT_TRUE(Nyxus::scan_slide_props(ani, 3, aniso_z4, false, e.fpimageOptions, e.resultOptions.need_annotation()));

	EXPECT_GT(ani.max_roi_d, iso.max_roi_d) << "z-anisotropy did not rescale ROI depth";
	EXPECT_GE(ani.max_roi_d, iso.max_roi_d * 3) << "z-depth not scaled ~4x";
	EXPECT_EQ(ani.max_roi_w, iso.max_roi_w) << "x (ax=1) must be unchanged";
	EXPECT_EQ(ani.max_roi_h, iso.max_roi_h) << "y (ay=1) must be unchanged";
}

// The whole-volume anisotropic scan (scan_trivial_wholevolume_anisotropic) walks the virtual
// grid with its own loop counter, indexes rows by fullW, and leaves vroi.aabb and aux_area
// describing the resampled cloud it just built. What this discriminates: a scan that clobbers
// its counter with the physical voxel index can run far past nVox before its exit condition
// holds again -- a hang, not a slowdown; one that indexes rows by fullH reads the wrong
// voxels; and one that leaves the pre-resample physical geometry in place sizes
// aux_image_cube too small for the resampled cloud (an out-of-bounds write) and divides MEAN
// by the wrong voxel count, putting it out by exactly the resampling factor. Every other 3D
// anisotropy test covers only the prescan's aabb, not the featurize-and-reduce this drives.
// MIN/MAX are structurally invariant to nearest-neighbour upsampling, and so is MEAN under
// this scan's truncation mapping (every physical voxel is duplicated the same number of
// times) -- so the run must match the isotropic one exactly, not merely look plausible.
TEST(TEST_NYXUS, TEST_3D_WHOLEVOLUME_ANISOTROPIC_REDUCE_MATCHES_ISOTROPIC_MECHANICS) {
	fs::path ds = ometiff_data_path("dim3_zyx.ome.tif");	// 3D X8 Y6 Z4
	ASSERT_TRUE(fs::exists(ds)) << ds.string();

	Environment e;
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);

	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ds.string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
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
TEST(TEST_NYXUS, TEST_3D_SEGMENTED_ANISOTROPIC_AUX_AREA_MATCHES_VOXELCLOUD_MECHANICS) {
	fs::path ip = ometiff_data_path("dim5.ome.tif");
	fs::path mp = ometiff_data_path("dim3_mask.ome.tif");
	ASSERT_TRUE(fs::exists(ip) && fs::exists(mp));

	Environment e;
	e.theFeatureSet.enableAll(false);
	e.theFeatureSet.enableFeatures(D3_VoxelIntensityFeatures::featureset);

	e.dataset.dataset_props.reserve(1);
	SlideProps& sp = e.dataset.dataset_props.emplace_back(ip.string(), mp.string());
	ASSERT_TRUE(Nyxus::scan_slide_props(sp, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
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

// Nested ROI on plain TIFF: the nested table is built from the feature CSVs' own column layout.
TEST(TEST_NYXUS, TEST_2D_NESTED_ROI_CSV_MECHANICS) {
	ASSERT_NO_THROW (test_2d_nested_roi_csv_mechanics());
}

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

// 3D GLCM "grey64" drift guards: named individually per feature rather than swept via TEST_P.
// History: tests/vetting/audit/glcm_3d_golden_regen.md, "grey64 table and the retired Wave-9 sweep".
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
// 3D GLRLM drift guards on the ut_ segmented phantom. The header forward-declares
// get_3d_segmented_phantom() rather than defining it, which is what makes it includable here.

TEST(TEST_NYXUS, TEST_3D_GLRLM_SRE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_sre_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lre_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LGLRE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lglre_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_HGLRE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_hglre_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRLGLE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_srlgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRHGLE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_srhgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRLGLE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lrlgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRHGLE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lrhgle_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLN_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_gln_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLNN_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_glnn_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RLN_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rln_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RLNN_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rlnn_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RP_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rp_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLV_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_glv_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RV_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rv_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RE_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_re_regression()); }

// The grey64 profile (GLRLM_GREYDEPTH=+64): the 13 angled values of each base feature and the
// mean its *_AVE twin stores, one named test per feature.
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLN_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_gln_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLNN_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_glnn_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLV_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_glv_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_HGLRE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_hglre_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LGLRE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lglre_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lre_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRHGLE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lrhgle_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRLGLE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lrlgle_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_re_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RLN_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rln_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RLNN_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rlnn_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RP_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rp_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RV_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rv_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_sre_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRHGLE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_srhgle_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRLGLE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_srlgle_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLN_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_gln_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLNN_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_glnn_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_GLV_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_glv_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_HGLRE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_hglre_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LGLRE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lglre_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lre_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRHGLE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lrhgle_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_LRLGLE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_lrlgle_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_re_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RLN_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rln_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RLNN_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rlnn_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RP_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rp_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_RV_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_rv_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_sre_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRHGLE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_srhgle_ave_grey64_regression()); }
TEST(TEST_NYXUS, TEST_3D_GLRLM_SRLGLE_AVE_GREY64_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_srlgle_ave_grey64_regression()); }

TEST(TEST_NYXUS, TEST_3D_GLRLM_DUMP_REGRESSION) { ASSERT_NO_THROW(test_3d_glrlm_dump_regression()); }

TEST(TEST_NYXUS, TEST_3D_NGLDM_DUMP_REGRESSION) { ASSERT_NO_THROW(test_3d_ngldm_dump_regression()); }
TEST(TEST_NYXUS, TEST_3D_NGLDM_DUMP_IBSI_MIRP) { ASSERT_NO_THROW(test_3d_ngldm_dump_ibsi_mirp()); }
