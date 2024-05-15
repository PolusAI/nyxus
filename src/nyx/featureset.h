#pragma once

#include <map>
#include <vector>
#include <iostream>
namespace Nyxus
{
	/// @brief Feature codes (2D)
	enum class Feature2D
	{
		// Intensity
		COV = 0,	// coefficient of variation
		COVERED_IMAGE_INTENSITY_RANGE,
		ENERGY,
		ENTROPY,
		EXCESS_KURTOSIS,
		HYPERFLATNESS,
		HYPERSKEWNESS,
		INTEGRATED_INTENSITY,
		INTERQUARTILE_RANGE,
		KURTOSIS,
		MAX,
		MEAN,
		MEAN_ABSOLUTE_DEVIATION,
		MEDIAN,
		MEDIAN_ABSOLUTE_DEVIATION,
		MIN,
		MODE,
		P01, P10, P25, P75, P90, P99,
		QCOD,	// quantile coefficient of dispersion
		RANGE,
		ROBUST_MEAN,
		ROBUST_MEAN_ABSOLUTE_DEVIATION,
		ROOT_MEAN_SQUARED,
		SKEWNESS,
		STANDARD_DEVIATION,
		STANDARD_DEVIATION_BIASED,
		STANDARD_ERROR,
		VARIANCE,
		VARIANCE_BIASED,
		UNIFORMITY,
		UNIFORMITY_PIU,

		// Morphology:
		AREA_PIXELS_COUNT,
		AREA_UM2,
		CENTROID_X,
		CENTROID_Y,
		WEIGHTED_CENTROID_Y,
		WEIGHTED_CENTROID_X,
		MASS_DISPLACEMENT,
		COMPACTNESS,
		BBOX_YMIN,
		BBOX_XMIN,
		BBOX_HEIGHT,
		BBOX_WIDTH,
		DIAMETER_EQUAL_AREA,
		EXTENT,
		ASPECT_RATIO,
		// -- Legendre inertia ellipse
		MAJOR_AXIS_LENGTH,
		MINOR_AXIS_LENGTH,
		// -- ellipticity related
		ECCENTRICITY,
		ELONGATION,
		ORIENTATION,
		ROUNDNESS,

		// --contour related
		PERIMETER,
		DIAMETER_EQUAL_PERIMETER,
		EDGE_MEAN_INTENSITY,
		EDGE_STDDEV_INTENSITY,
		EDGE_MAX_INTENSITY,
		EDGE_MIN_INTENSITY,
		EDGE_INTEGRATED_INTENSITY,
		CIRCULARITY,

		// -- convex hull related
		CONVEX_HULL_AREA,
		SOLIDITY,

		// -- erosions
		EROSIONS_2_VANISH,
		EROSIONS_2_VANISH_COMPLEMENT,

		// -- fractal dimension
		FRACT_DIM_BOXCOUNT,
		FRACT_DIM_PERIMETER,

		// Caliper:
		MIN_FERET_DIAMETER,
		MAX_FERET_DIAMETER,
		MIN_FERET_ANGLE,
		MAX_FERET_ANGLE,

		STAT_FERET_DIAM_MIN,
		STAT_FERET_DIAM_MAX,
		STAT_FERET_DIAM_MEAN,
		STAT_FERET_DIAM_MEDIAN,
		STAT_FERET_DIAM_STDDEV,
		STAT_FERET_DIAM_MODE,

		STAT_MARTIN_DIAM_MIN,
		STAT_MARTIN_DIAM_MAX,
		STAT_MARTIN_DIAM_MEAN,
		STAT_MARTIN_DIAM_MEDIAN,
		STAT_MARTIN_DIAM_STDDEV,
		STAT_MARTIN_DIAM_MODE,

		STAT_NASSENSTEIN_DIAM_MIN,
		STAT_NASSENSTEIN_DIAM_MAX,
		STAT_NASSENSTEIN_DIAM_MEAN,
		STAT_NASSENSTEIN_DIAM_MEDIAN,
		STAT_NASSENSTEIN_DIAM_STDDEV,
		STAT_NASSENSTEIN_DIAM_MODE,

		// -- Chords
		MAXCHORDS_MAX,
		MAXCHORDS_MAX_ANG,
		MAXCHORDS_MIN,
		MAXCHORDS_MIN_ANG,
		MAXCHORDS_MEDIAN,
		MAXCHORDS_MEAN,
		MAXCHORDS_MODE,
		MAXCHORDS_STDDEV,
		ALLCHORDS_MAX,
		ALLCHORDS_MAX_ANG,
		ALLCHORDS_MIN,
		ALLCHORDS_MIN_ANG,
		ALLCHORDS_MEDIAN,
		ALLCHORDS_MEAN,
		ALLCHORDS_MODE,
		ALLCHORDS_STDDEV,

		EULER_NUMBER,

		EXTREMA_P1_X, EXTREMA_P1_Y,
		EXTREMA_P2_X, EXTREMA_P2_Y,
		EXTREMA_P3_X, EXTREMA_P3_Y,
		EXTREMA_P4_X, EXTREMA_P4_Y,
		EXTREMA_P5_X, EXTREMA_P5_Y,
		EXTREMA_P6_X, EXTREMA_P6_Y,
		EXTREMA_P7_X, EXTREMA_P7_Y,
		EXTREMA_P8_X, EXTREMA_P8_Y,

		// -- polygonal representation
		POLYGONALITY_AVE,
		HEXAGONALITY_AVE,
		HEXAGONALITY_STDDEV,

		DIAMETER_MIN_ENCLOSING_CIRCLE,
		DIAMETER_CIRCUMSCRIBING_CIRCLE,
		DIAMETER_INSCRIBING_CIRCLE,

		GEODETIC_LENGTH,
		THICKNESS,

		// -- ROI radius features
		ROI_RADIUS_MEAN,
		ROI_RADIUS_MAX,
		ROI_RADIUS_MEDIAN,

		// Neighbor features
		NUM_NEIGHBORS,
		PERCENT_TOUCHING,
		CLOSEST_NEIGHBOR1_DIST,
		CLOSEST_NEIGHBOR1_ANG,
		CLOSEST_NEIGHBOR2_DIST,
		CLOSEST_NEIGHBOR2_ANG,
		ANG_BW_NEIGHBORS_MEAN,
		ANG_BW_NEIGHBORS_STDDEV,
		ANG_BW_NEIGHBORS_MODE,

		// GLCM:
		GLCM_ASM,			// Angular second moment, IBSI # 8ZQL
		GLCM_ACOR,			// Autocorrelation, IBSI # QWB0
		GLCM_CLUPROM,		// Cluster prominence, IBSI # AE86
		GLCM_CLUSHADE,		// Cluster shade, IBSI # 7NFM
		GLCM_CLUTEND,		// Cluster tendency, IBSI # DG8W
		GLCM_CONTRAST,		// Contrast, IBSI # ACUI
		GLCM_CORRELATION,	// Correlation, IBSI # NI2N
		GLCM_DIFAVE,		// Difference average, IBSI # TF7R
		GLCM_DIFENTRO,		// Difference entropy, IBSI # NTRS
		GLCM_DIFVAR,		// Difference variance, IBSI # D3YU
		GLCM_DIS,			// Dissimilarity, IBSI # 8S9J
		GLCM_ENERGY,		// Energy
		GLCM_ENTROPY,		// Entropy
		GLCM_HOM1,			// Homogeneity-1 (PyR)
		GLCM_HOM2,			// Homogeneity-2 (PyR)
		GLCM_ID,			// Inv diff, IBSI # IB1Z
		GLCM_IDN,			// Inv diff normalized, IBSI # NDRX
		GLCM_IDM,			// Inv diff mom, IBSI # WF0Z
		GLCM_IDMN,			// Inv diff mom normalized, IBSI # 1QCO
		GLCM_INFOMEAS1,		// Information measure of correlation 1, IBSI # R8DG
		GLCM_INFOMEAS2,		// Information measure of correlation 2, IBSI # JN9H
		GLCM_IV,			// Inv variance, IBSI # E8JP
		GLCM_JAVE,			// Joint average, IBSI # 60VM
		GLCM_JE,			// Joint entropy, IBSI # TU9B
		GLCM_JMAX,			// Joint max (aka PyR max probability), IBSI # GYBY
		GLCM_JVAR,			// Joint var (aka PyR Sum of Squares), IBSI # UR99
		GLCM_SUMAVERAGE,	// Sum average, IBSI # ZGXS
		GLCM_SUMENTROPY,	// Sum entropy, IBSI # P6QZ
		GLCM_SUMVARIANCE,	// Sum variance, IBSI # OEEB
		GLCM_VARIANCE,
		// -- averages --
		GLCM_ASM_AVE,
		GLCM_ACOR_AVE,
		GLCM_CLUPROM_AVE,
		GLCM_CLUSHADE_AVE,
		GLCM_CLUTEND_AVE,
		GLCM_CONTRAST_AVE,
		GLCM_CORRELATION_AVE,
		GLCM_DIFAVE_AVE,
		GLCM_DIFENTRO_AVE,
		GLCM_DIFVAR_AVE,
		GLCM_DIS_AVE,
		GLCM_ENERGY_AVE,
		GLCM_ENTROPY_AVE,
		GLCM_HOM1_AVE,
		GLCM_ID_AVE,
		GLCM_IDN_AVE,
		GLCM_IDM_AVE,
		GLCM_IDMN_AVE,
		GLCM_IV_AVE,
		GLCM_JAVE_AVE,
		GLCM_JE_AVE,
		GLCM_INFOMEAS1_AVE,
		GLCM_INFOMEAS2_AVE,
		GLCM_VARIANCE_AVE,
		GLCM_JMAX_AVE,
		GLCM_JVAR_AVE,
		GLCM_SUMAVERAGE_AVE,
		GLCM_SUMENTROPY_AVE,
		GLCM_SUMVARIANCE_AVE,

		// GLRLM:
		GLRLM_SRE,	// Short Run Emphasis 
		GLRLM_LRE,	// Long Run Emphasis 
		GLRLM_GLN,	// Gray Level Non-Uniformity 
		GLRLM_GLNN,	// Gray Level Non-Uniformity Normalized 
		GLRLM_RLN,	// Run Length Non-Uniformity
		GLRLM_RLNN,	// Run Length Non-Uniformity Normalized 
		GLRLM_RP,	// Run Percentage
		GLRLM_GLV,	// Gray Level Variance 
		GLRLM_RV,	// Run Variance 
		GLRLM_RE,	// Run Entropy 
		GLRLM_LGLRE,	// Low Gray Level Run Emphasis 
		GLRLM_HGLRE,	// High Gray Level Run Emphasis 
		GLRLM_SRLGLE,	// Short Run Low Gray Level Emphasis 
		GLRLM_SRHGLE,	// Short Run High Gray Level Emphasis 
		GLRLM_LRLGLE,	// Long Run Low Gray Level Emphasis 
		GLRLM_LRHGLE,	// Long Run High Gray Level Emphasis 
		// -- averages --
		GLRLM_SRE_AVE,
		GLRLM_LRE_AVE,
		GLRLM_GLN_AVE,
		GLRLM_GLNN_AVE,
		GLRLM_RLN_AVE,
		GLRLM_RLNN_AVE,
		GLRLM_RP_AVE,
		GLRLM_GLV_AVE,
		GLRLM_RV_AVE,
		GLRLM_RE_AVE,
		GLRLM_LGLRE_AVE,
		GLRLM_HGLRE_AVE,
		GLRLM_SRLGLE_AVE,
		GLRLM_SRHGLE_AVE,
		GLRLM_LRLGLE_AVE,
		GLRLM_LRHGLE_AVE,

		// GLDZM:
		GLDZM_SDE,		// Small Distance Emphasis
		GLDZM_LDE,		// Large Distance Emphasis
		GLDZM_LGLZE,	// Low Grey Level Zone Emphasis
		GLDZM_HGLZE,	// High Grey Level Zone Emphasis
		GLDZM_SDLGLE,	// Small Distance Low Grey Level Emphasis
		GLDZM_SDHGLE,	// Small Distance High Grey Level Emphasis
		GLDZM_LDLGLE,	// Large Distance Low Grey Level Emphasis
		GLDZM_LDHGLE,	// Large Distance High Grey Level Emphasis
		GLDZM_GLNU,		// Grey Level Non Uniformity
		GLDZM_GLNUN,	// Grey Level Non Uniformity Normalized
		GLDZM_ZDNU,		// Zone Distance Non Uniformity
		GLDZM_ZDNUN,	// Zone Distance Non Uniformity Normalized
		GLDZM_ZP,		// Zone Percentage
		GLDZM_GLM,		// Grey Level Mean
		GLDZM_GLV,		// Grey Level Variance
		GLDZM_ZDM,		// Zone Distance Mean
		GLDZM_ZDV,		// Zone Distance Variance
		GLDZM_ZDE,		// Zone Distance Entropy

		// GLSZM:
		GLSZM_SAE,	// Small Area Emphasis
		GLSZM_LAE,	// Large Area Emphasis
		GLSZM_GLN,	// Gray Level Non - Uniformity
		GLSZM_GLNN,	// Gray Level Non - Uniformity Normalized
		GLSZM_SZN,	// Size - Zone Non - Uniformity
		GLSZM_SZNN,	// Size - Zone Non - Uniformity Normalized
		GLSZM_ZP,	// Zone Percentage
		GLSZM_GLV,	// Gray Level Variance
		GLSZM_ZV,	// Zone Variance
		GLSZM_ZE,	// Zone Entropy
		GLSZM_LGLZE,	// Low Gray Level Zone Emphasis
		GLSZM_HGLZE,	// High Gray Level Zone Emphasis
		GLSZM_SALGLE,	// Small Area Low Gray Level Emphasis
		GLSZM_SAHGLE,	// Small Area High Gray Level Emphasis
		GLSZM_LALGLE,	// Large Area Low Gray Level Emphasis
		GLSZM_LAHGLE,	// Large Area High Gray Level Emphasis

		// GLDM:
		GLDM_SDE,		// Small Dependence Emphasis
		GLDM_LDE,		// Large Dependence Emphasis
		GLDM_GLN,		// Gray Level Non-Uniformity
		GLDM_DN,		// Dependence Non-Uniformity
		GLDM_DNN,		// Dependence Non-Uniformity Normalized
		GLDM_GLV,		// Gray Level Variance
		GLDM_DV,		// Dependence Variance
		GLDM_DE,		// Dependence Entropy
		GLDM_LGLE,		// Low Gray Level Emphasis
		GLDM_HGLE,		// High Gray Level Emphasis
		GLDM_SDLGLE,	// Small Dependence Low Gray Level Emphasis
		GLDM_SDHGLE,	// Small Dependence High Gray Level Emphasis
		GLDM_LDLGLE,	// Large Dependence Low Gray Level Emphasis
		GLDM_LDHGLE,	// Large Dependence High Gray Level Emphasis

		// NGLDM:
		NGLDM_LDE,		// Low Dependence Emphasis (IBSI # SODN)
		NGLDM_HDE,		// High Dependence Emphasis (IBSI # IMOQ)
		NGLDM_LGLCE,	// Low Grey Level Count Emphasis (IBSI # TL9H)
		NGLDM_HGLCE,	// High Grey Level Count Emphasis (IBSI # OAE7)
		NGLDM_LDLGLE,	// Low Dependence Low Grey Level Emphasis (IBSI # EQ3F)
		NGLDM_LDHGLE,	// Low Dependence High Grey Level Emphasis (IBSI # JA6D)
		NGLDM_HDLGLE,	// High Dependence Low Grey Level Emphasis (IBSI # NBZI)
		NGLDM_HDHGLE,	// High Dependence High Grey Level Emphasis (IBSI # 9QMG)
		NGLDM_GLNU,		// Grey Level Non-Uniformity (IBSI # FP8K)
		NGLDM_GLNUN,	// Grey Level Non-Uniformity Normalised (IBSI # 5SPA)
		NGLDM_DCNU,		// Dependence Count Non-Uniformity (IBSI # Z87G)
		NGLDM_DCNUN,	// Dependence Count Non-Uniformity Normalised (IBSI # OKJI)
		NGLDM_DCP,		// Dependence Count Percentage (IBSI # 6XV8)
		NGLDM_GLM,		// Grey Level Mean
		NGLDM_GLV,		// Grey Level Variance (IBSI # 1PFV)
		NGLDM_DCM,		// Dependence Count Mean
		NGLDM_DCV,		// Dependence Count Variance (IBSI # DNX2)
		NGLDM_DCENT,	// Dependence Count Entropy (IBSI # FCBV)
		NGLDM_DCENE,	// Dependence Count Energy (IBSI # CAS9)

		// NGTDM:
		NGTDM_COARSENESS,
		NGTDM_CONTRAST,
		NGTDM_BUSYNESS,
		NGTDM_COMPLEXITY,
		NGTDM_STRENGTH,

		// Low-frequency intensity distribution:
		FRAC_AT_D,
		GABOR,
		MEAN_FRAC,
		RADIAL_CV,
		ZERNIKE2D,

		// Spatial (raw) moments
		SPAT_MOMENT_00,
		SPAT_MOMENT_01,
		SPAT_MOMENT_02,
		SPAT_MOMENT_03,
		SPAT_MOMENT_10,
		SPAT_MOMENT_11,
		SPAT_MOMENT_12,
		SPAT_MOMENT_13,
		SPAT_MOMENT_20,
		SPAT_MOMENT_21,
		SPAT_MOMENT_22,
		SPAT_MOMENT_23,
		SPAT_MOMENT_30,

		// Central moments
		CENTRAL_MOMENT_00,
		CENTRAL_MOMENT_01,
		CENTRAL_MOMENT_02,
		CENTRAL_MOMENT_03,
		CENTRAL_MOMENT_10,
		CENTRAL_MOMENT_11,
		CENTRAL_MOMENT_12,
		CENTRAL_MOMENT_13,
		CENTRAL_MOMENT_20,
		CENTRAL_MOMENT_21,
		CENTRAL_MOMENT_22,
		CENTRAL_MOMENT_23,
		CENTRAL_MOMENT_30,
		CENTRAL_MOMENT_31,
		CENTRAL_MOMENT_32,
		CENTRAL_MOMENT_33,

		// Normalized (standardized) spatial moments
		NORM_SPAT_MOMENT_00,
		NORM_SPAT_MOMENT_01,
		NORM_SPAT_MOMENT_02,
		NORM_SPAT_MOMENT_03,
		NORM_SPAT_MOMENT_10,
		NORM_SPAT_MOMENT_11,
		NORM_SPAT_MOMENT_12,
		NORM_SPAT_MOMENT_13,
		NORM_SPAT_MOMENT_20,
		NORM_SPAT_MOMENT_21,
		NORM_SPAT_MOMENT_22,
		NORM_SPAT_MOMENT_23,
		NORM_SPAT_MOMENT_30,
		NORM_SPAT_MOMENT_31,
		NORM_SPAT_MOMENT_32,
		NORM_SPAT_MOMENT_33,

		// Normalized central moments
		NORM_CENTRAL_MOMENT_02,
		NORM_CENTRAL_MOMENT_03,
		NORM_CENTRAL_MOMENT_11,
		NORM_CENTRAL_MOMENT_12,
		NORM_CENTRAL_MOMENT_20,
		NORM_CENTRAL_MOMENT_21,
		NORM_CENTRAL_MOMENT_30,

		// Hu's moments 1-7 
		HU_M1,
		HU_M2,
		HU_M3,
		HU_M4,
		HU_M5,
		HU_M6,
		HU_M7,

		// Weighted spatial moments
		WEIGHTED_SPAT_MOMENT_00,
		WEIGHTED_SPAT_MOMENT_01,
		WEIGHTED_SPAT_MOMENT_02,
		WEIGHTED_SPAT_MOMENT_03,
		WEIGHTED_SPAT_MOMENT_10,
		WEIGHTED_SPAT_MOMENT_11,
		WEIGHTED_SPAT_MOMENT_12,
		WEIGHTED_SPAT_MOMENT_20,
		WEIGHTED_SPAT_MOMENT_21,
		WEIGHTED_SPAT_MOMENT_30,

		// Weighted central moments
		WEIGHTED_CENTRAL_MOMENT_02,
		WEIGHTED_CENTRAL_MOMENT_03,
		WEIGHTED_CENTRAL_MOMENT_11,
		WEIGHTED_CENTRAL_MOMENT_12,
		WEIGHTED_CENTRAL_MOMENT_20,
		WEIGHTED_CENTRAL_MOMENT_21,
		WEIGHTED_CENTRAL_MOMENT_30,

		// Weighted Hu's moments 1-7 
		WEIGHTED_HU_M1,
		WEIGHTED_HU_M2,
		WEIGHTED_HU_M3,
		WEIGHTED_HU_M4,
		WEIGHTED_HU_M5,
		WEIGHTED_HU_M6,
		WEIGHTED_HU_M7,

		_COUNT_
	};

	enum class Feature3D
	{
		// Intensity
		COV = (int) Feature2D::_COUNT_,
		COVERED_IMAGE_INTENSITY_RANGE,
		ENERGY,
		ENTROPY,
		EXCESS_KURTOSIS,
		HYPERFLATNESS,
		HYPERSKEWNESS,
		INTEGRATED_INTENSITY,
		INTERQUARTILE_RANGE,
		KURTOSIS,
		MAX,
		MEAN,
		MEAN_ABSOLUTE_DEVIATION,
		MEDIAN,
		MEDIAN_ABSOLUTE_DEVIATION,
		MIN,
		MODE,
		P01, P10, P25, P75, P90, P99,
		QCOD,
		RANGE,
		ROBUST_MEAN,
		ROBUST_MEAN_ABSOLUTE_DEVIATION,
		ROOT_MEAN_SQUARED,
		SKEWNESS,
		STANDARD_DEVIATION,
		STANDARD_DEVIATION_BIASED,
		STANDARD_ERROR,
		VARIANCE,
		VARIANCE_BIASED,
		UNIFORMITY,
		UNIFORMITY_PIU,

		// Morphology:
		VOLUME_PIXELS,
		CENTROID_X,
		CENTROID_Y,
		CENTROID_Z,
		BBOX_XMIN,
		BBOX_YMIN,
		BBOX_ZMIN,
		BBOX_HEIGHT,
		BBOX_WIDTH,
		BBOX_DEPTH,

		// Neighbor features
		NUM_NEIGHBORS,
		PERCENT_TOUCHING,
		CLOSEST_NEIGHBOR1_DIST,
		CLOSEST_NEIGHBOR1_ANG,
		CLOSEST_NEIGHBOR2_DIST,
		CLOSEST_NEIGHBOR2_ANG,
		ANG_BW_NEIGHBORS_MEAN,
		ANG_BW_NEIGHBORS_STDDEV,
		ANG_BW_NEIGHBORS_MODE,

		// Spatial (raw) moments
		SPAT_MOMENT_00,
		SPAT_MOMENT_01,
		SPAT_MOMENT_02,
		SPAT_MOMENT_03,
		SPAT_MOMENT_10,
		SPAT_MOMENT_11,
		SPAT_MOMENT_12,
		SPAT_MOMENT_13,
		SPAT_MOMENT_20,
		SPAT_MOMENT_21,
		SPAT_MOMENT_22,
		SPAT_MOMENT_23,
		SPAT_MOMENT_30,

		_COUNT_
	};

	enum class FeatureIMQ {
		// Image Quality features
		FOCUS_SCORE = (int) Feature3D::_COUNT_,
		LOCAL_FOCUS_SCORE,
		POWER_SPECTRUM_SLOPE,
		MAX_SATURATION,
		MIN_SATURATION,
		SHARPNESS,

		_COUNT_
	};

	enum class Fgroup2D
	{
		FG2_ALL = 0,
		FG2_INTENSITY, 
		FG2_MORPHOLOGY, 
		FG2_BASIC_MORPHOLOGY, 
		FG2_GLCM, 
		FG2_GLRLM, 
		FG2_GLDZM, 
		FG2_GLSZM,
		FG2_GLDM,
		FG2_NGLDM,
		FG2_NGTDM,
		FG2_BUT_GABOR,
		FG2_ALL_BUT_GLCM, 
		FG2_EASY, 
		FG2_NEIG,
		FG2_MOMENTS,
		_COUNT_
	};	
	
	enum class Fgroup3D
	{
		FG3_ALL = (int) Fgroup2D::_COUNT_,
		FG3_INTENSITY,
		FG3_MORPHOLOGY,
		FG3_TEXTURE,
		FG3_NEIG,
		FG3_MOMENTS,
		_COUNT_
	};

	enum class FgroupIMQ
	{
		ALL_IMQ= (int) Fgroup3D::_COUNT_,

		__COUNT_
	};
}

/// @brief Helper class to set and access user feature selection made via the command line or Python interface.
class FeatureSet
{
public:
	FeatureSet();
	void enableAll(bool newStatus = true) 
	{ 
		for (int i = 0; i < int(Nyxus::Feature2D::_COUNT_); i++) m_enabledFeatures[i] = newStatus; 
	}
	void enableAllIMQ(bool newStatus = true)
	{
		for (int i = int(Nyxus::Feature3D::_COUNT_); i < int(Nyxus::FeatureIMQ::_COUNT_); i++) {
			m_enabledFeatures[i] = newStatus; 
		}
	}
	void disableFeatures(const std::initializer_list<Nyxus::Feature2D>& desiredFeatures)
	{
		for (auto f : desiredFeatures)
			m_enabledFeatures[(int)f] = false;
	}
	void enableFeatures(const std::initializer_list<Nyxus::Feature2D>& desiredFeatures) {
		for (auto f : desiredFeatures)
			m_enabledFeatures[(int)f] = true;
	}
	void enableFeatures(const std::initializer_list<Nyxus::Feature3D>& desiredFeatures) 
	{
		for (auto f : desiredFeatures)
			m_enabledFeatures[(int)f] = true;
	}
	void enableFeatures(const std::initializer_list<Nyxus::FeatureIMQ>& desiredFeatures) {
		for (auto f : desiredFeatures)
			m_enabledFeatures[(int)f] = true;
	}
	void enableFeature (int f)
	{
		m_enabledFeatures [f] = true;
	}
	void enablePixelIntenStats() {
		enableAll(false);
		m_enabledFeatures[(int)Nyxus::Feature2D::MEAN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MEDIAN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MIN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MAX] =
			m_enabledFeatures[(int)Nyxus::Feature2D::RANGE] =
			m_enabledFeatures[(int)Nyxus::Feature2D::COVERED_IMAGE_INTENSITY_RANGE] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STANDARD_DEVIATION] =
			m_enabledFeatures[(int)Nyxus::Feature2D::SKEWNESS] =
			m_enabledFeatures[(int)Nyxus::Feature2D::KURTOSIS] =
			m_enabledFeatures[(int)Nyxus::Feature2D::EXCESS_KURTOSIS] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MEDIAN_ABSOLUTE_DEVIATION] =
			m_enabledFeatures[(int)Nyxus::Feature2D::ENERGY] =
			m_enabledFeatures[(int)Nyxus::Feature2D::ROOT_MEAN_SQUARED] =
			m_enabledFeatures[(int)Nyxus::Feature2D::ENTROPY] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MODE] =
			m_enabledFeatures[(int)Nyxus::Feature2D::UNIFORMITY] =
			m_enabledFeatures[(int)Nyxus::Feature2D::P10] = 
			m_enabledFeatures[(int)Nyxus::Feature2D::P25] = 
			m_enabledFeatures[(int)Nyxus::Feature2D::P75] = 
			m_enabledFeatures[(int)Nyxus::Feature2D::P90] =
			m_enabledFeatures[(int)Nyxus::Feature2D::QCOD] =
			m_enabledFeatures[(int)Nyxus::Feature2D::INTERQUARTILE_RANGE] =
			m_enabledFeatures[(int)Nyxus::Feature2D::ROBUST_MEAN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION] =
			m_enabledFeatures[(int)Nyxus::Feature2D::COV] =
			m_enabledFeatures[(int)Nyxus::Feature2D::WEIGHTED_CENTROID_Y] =
			m_enabledFeatures[(int)Nyxus::Feature2D::WEIGHTED_CENTROID_X] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MASS_DISPLACEMENT] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STANDARD_DEVIATION_BIASED] =
			m_enabledFeatures[(int)Nyxus::Feature2D::VARIANCE] =
			m_enabledFeatures[(int)Nyxus::Feature2D::VARIANCE_BIASED] = true;
	}
	void enableBoundingBox() {
		enableAll(false);
		m_enabledFeatures[(int)Nyxus::Feature2D::BBOX_YMIN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::BBOX_XMIN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::BBOX_HEIGHT] =
			m_enabledFeatures[(int)Nyxus::Feature2D::BBOX_WIDTH] = true;
	}
	void enableFeret() {
		enableAll(false);
		m_enabledFeatures[(int)Nyxus::Feature2D::MIN_FERET_DIAMETER] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MAX_FERET_DIAMETER] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MIN_FERET_ANGLE] =
			m_enabledFeatures[(int)Nyxus::Feature2D::MAX_FERET_ANGLE] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STAT_FERET_DIAM_MIN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STAT_FERET_DIAM_MAX] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STAT_FERET_DIAM_MEAN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STAT_FERET_DIAM_MEDIAN] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STAT_FERET_DIAM_STDDEV] =
			m_enabledFeatures[(int)Nyxus::Feature2D::STAT_FERET_DIAM_MODE] = true;
	}
	bool isEnabled (Nyxus::Feature2D fc) const
	{ 
		return fc < Nyxus::Feature2D::_COUNT_ ? m_enabledFeatures[(int)fc] : false;
	}
	bool isEnabled (Nyxus::Feature3D fc) const
	{ 
		return fc < Nyxus::Feature3D::_COUNT_ ? m_enabledFeatures[(int)fc] : false;
	}
	bool isEnabled (Nyxus::FeatureIMQ fc) const
	{ 
		return fc < Nyxus::FeatureIMQ::_COUNT_ ? m_enabledFeatures[(int)fc] : false;
	}

	bool anyEnabled (const std::initializer_list<Nyxus::Feature2D>& F) const
	{
		for (auto f : F)
			if (m_enabledFeatures[(int)f])
				return true;
		return false;
	}
	bool anyEnabled (const std::initializer_list<Nyxus::Feature3D>& F) const
	{
		for (auto f : F)
			if (m_enabledFeatures[(int)f])
				return true;
		return false;
	}
	bool anyEnabled (const std::initializer_list<Nyxus::FeatureIMQ>& F) const
	{
		for (auto f : F)
			if (m_enabledFeatures[(int)f])
				return true;
		return false;
	}
	int numOfEnabled() {
		int cnt = 0;
		for (int i = 0; i < (int) Nyxus::Feature2D::_COUNT_; i++)
			if (m_enabledFeatures[i])
				cnt++;
		return cnt;
	}
	bool find_2D_FeatureByString (const std::string & feature_name, Nyxus::Feature2D & feature_code);
	bool find_2D_GroupByString (const std::string & group_name, Nyxus::Fgroup2D & group_code);
	bool find_3D_FeatureByString (const std::string & feature_name, Nyxus::Feature3D & feature_code);
	bool find_3D_GroupByString (const std::string & group_name, Nyxus::Fgroup3D & group_code);
	bool find_IMQ_FeatureByString (const std::string & feature_name, Nyxus::FeatureIMQ & feature_code);
	bool find_IMQ_GroupByString (const std::string & group_name, Nyxus::FgroupIMQ & group_code);

	std::string findFeatureNameByCode (Nyxus::Feature2D code);
	std::string findFeatureNameByCode (Nyxus::Feature3D fcode);
	std::string findGroupNameByCode (Nyxus::Fgroup2D code);
	std::string findGroupNameByCode (Nyxus::Fgroup3D code);
	std::string findGroupNameByCode (Nyxus::FgroupIMQ code);

	void show_help();

	// Relying on RVO rather than std::move
	std::vector<std::tuple<std::string, int>> getEnabledFeatures();

private:
	bool m_enabledFeatures [(int) Nyxus::FeatureIMQ::_COUNT_];
};

namespace Nyxus
{
	extern FeatureSet theFeatureSet;
	extern std::map <std::string, Nyxus::Feature2D> UserFacingFeatureNames;
	extern std::map <std::string, Nyxus::FeatureIMQ> UserFacingIMQFeatureGroupNames;
	extern std::map <std::string, Nyxus::Fgroup2D> UserFacing2dFeaturegroupNames;
	extern std::map <std::string, Nyxus::Feature3D> UserFacing_3D_featureNames;
	extern std::map <std::string, Nyxus::Fgroup3D> UserFacing3dFeaturegroupNames;
}
