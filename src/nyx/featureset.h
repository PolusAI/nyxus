#pragma once

#include <map>
#include <vector>

namespace Nyxus
{
	/// @brief Feature codes
	enum AvailableFeatures
	{
		//==== 2D features
		
		// Pixel intensity stats
		INTEGRATED_INTENSITY = 0,
		MEAN,
		MEDIAN,
		MIN,
		MAX,
		RANGE,
		STANDARD_DEVIATION,
		STANDARD_ERROR,
		SKEWNESS,
		KURTOSIS,
		HYPERSKEWNESS,
		HYPERFLATNESS,
		MEAN_ABSOLUTE_DEVIATION,
		ENERGY,
		ROOT_MEAN_SQUARED,
		ENTROPY,
		MODE,
		UNIFORMITY,
		UNIFORMITY_PIU,
		P01, P10, P25, P75, P90, P99,
		INTERQUARTILE_RANGE,
		ROBUST_MEAN_ABSOLUTE_DEVIATION,

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
		GLCM_ANGULAR2NDMOMENT,
		GLCM_CONTRAST,
		GLCM_CORRELATION,
		GLCM_DIFFERENCEAVERAGE,
		GLCM_DIFFERENCEENTROPY,
		GLCM_DIFFERENCEVARIANCE,
		GLCM_ENERGY,
		GLCM_ENTROPY,
		GLCM_HOMOGENEITY,
		GLCM_INFOMEAS1,
		GLCM_INFOMEAS2,
		GLCM_INVERSEDIFFERENCEMOMENT,
		GLCM_SUMAVERAGE,
		GLCM_SUMENTROPY,
		GLCM_SUMVARIANCE,
		GLCM_VARIANCE,

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
		GLDM_SDE,	// Small Dependence Emphasis(SDE)
		GLDM_LDE,	// Large Dependence Emphasis (LDE)
		GLDM_GLN,	// Gray Level Non-Uniformity (GLN)
		GLDM_DN,	// Dependence Non-Uniformity (DN)
		GLDM_DNN,	// Dependence Non-Uniformity Normalized (DNN)
		GLDM_GLV,	// Gray Level Variance (GLV)
		GLDM_DV,	// Dependence Variance (DV)
		GLDM_DE,	// Dependence Entropy (DE)
		GLDM_LGLE,	// Low Gray Level Emphasis (LGLE)
		GLDM_HGLE,	// High Gray Level Emphasis (HGLE)
		GLDM_SDLGLE,	// Small Dependence Low Gray Level Emphasis (SDLGLE)
		GLDM_SDHGLE,	// Small Dependence High Gray Level Emphasis (SDHGLE)
		GLDM_LDLGLE,	// Large Dependence Low Gray Level Emphasis (LDLGLE)
		GLDM_LDHGLE,	// Large Dependence High Gray Level Emphasis (LDHGLE)

		// NGTDM:
		NGTDM_COARSENESS,
		NGTDM_CONTRAST,
		NGTDM_BUSYNESS,
		NGTDM_COMPLEXITY,
		NGTDM_STRENGTH,

		// Radial intensity distribution:
		ZERNIKE2D,
		FRAC_AT_D,
		MEAN_FRAC,
		RADIAL_CV,
			
		// Spatial (raw) moments
		SPAT_MOMENT_00,
		SPAT_MOMENT_01,
		SPAT_MOMENT_02,
		SPAT_MOMENT_03,
		SPAT_MOMENT_10,
		SPAT_MOMENT_11,
		SPAT_MOMENT_12,
		SPAT_MOMENT_20,
		SPAT_MOMENT_21,
		SPAT_MOMENT_30,

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

		// Central moments
		CENTRAL_MOMENT_02,
		CENTRAL_MOMENT_03,
		CENTRAL_MOMENT_11,
		CENTRAL_MOMENT_12,
		CENTRAL_MOMENT_20,
		CENTRAL_MOMENT_21,
		CENTRAL_MOMENT_30,

		// Weighted central moments
		WEIGHTED_CENTRAL_MOMENT_02,
		WEIGHTED_CENTRAL_MOMENT_03,
		WEIGHTED_CENTRAL_MOMENT_11,
		WEIGHTED_CENTRAL_MOMENT_12,
		WEIGHTED_CENTRAL_MOMENT_20,
		WEIGHTED_CENTRAL_MOMENT_21,
		WEIGHTED_CENTRAL_MOMENT_30,

		// Normalized central moments
		NORM_CENTRAL_MOMENT_02,
		NORM_CENTRAL_MOMENT_03,
		NORM_CENTRAL_MOMENT_11,
		NORM_CENTRAL_MOMENT_12,
		NORM_CENTRAL_MOMENT_20,
		NORM_CENTRAL_MOMENT_21,
		NORM_CENTRAL_MOMENT_30,

		// Normalized (standardized) spatial moments
		NORM_SPAT_MOMENT_00,
		NORM_SPAT_MOMENT_01,
		NORM_SPAT_MOMENT_02,
		NORM_SPAT_MOMENT_03,
		NORM_SPAT_MOMENT_10,
		NORM_SPAT_MOMENT_20,
		NORM_SPAT_MOMENT_30,

		// Hu's moments 1-7 
		HU_M1,
		HU_M2,
		HU_M3,
		HU_M4,
		HU_M5,
		HU_M6,
		HU_M7,

		// Weighted Hu's moments 1-7 
		WEIGHTED_HU_M1,
		WEIGHTED_HU_M2,
		WEIGHTED_HU_M3,
		WEIGHTED_HU_M4,
		WEIGHTED_HU_M5,
		WEIGHTED_HU_M6,
		WEIGHTED_HU_M7,

		GABOR,

		//==== 3D features
		
		// Pixel intensity stats
		D3_INTEGRATED_INTENSITY,
		D3_MEAN,
		D3_MEDIAN,
		D3_MIN,
		D3_MAX,
		D3_RANGE,
		D3_STANDARD_DEVIATION,
		D3_STANDARD_ERROR,
		D3_SKEWNESS,
		D3_KURTOSIS,
		D3_HYPERSKEWNESS,
		D3_HYPERFLATNESS,
		D3_MEAN_ABSOLUTE_DEVIATION,
		D3_ENERGY,
		D3_ROOT_MEAN_SQUARED,
		D3_ENTROPY,
		D3_MODE,
		D3_UNIFORMITY,
		D3_UNIFORMITY_PIU,
		D3_P01, D3_P10, D3_P25, D3_P75, D3_P90, D3_P99,
		D3_INTERQUARTILE_RANGE,
		D3_ROBUST_MEAN_ABSOLUTE_DEVIATION,

		/*
		// Morphology:
		D3_VOLUME_PIXELS_COUNT,
		D3_CENTROID_X,
		D3_CENTROID_Y,
		D3_CENTROID_Z, 
		D3_WEIGHTED_CENTROID_Y,
		D3_WEIGHTED_CENTROID_X,
		D3_WEIGHTED_CENTROID_Z,
		D3_MASS_DISPLACEMENT,
		D3_COMPACTNESS,
		D3_BBOX_YMIN,
		D3_BBOX_XMIN,
		D3_BBOX_ZMIN,
		D3_BBOX_HEIGHT,
		D3_BBOX_WIDTH,
		D3_BBOX_DEPTH,
		D3_EXTENT,
		D3_ASPECT_RATIO,
		*/

		// raw moments
		D3_RAW_MOMENT_000,

		D3_RAW_MOMENT_010,
		D3_RAW_MOMENT_011,
		D3_RAW_MOMENT_012,
		D3_RAW_MOMENT_013,

		D3_RAW_MOMENT_020,
		D3_RAW_MOMENT_021,
		D3_RAW_MOMENT_022,
		D3_RAW_MOMENT_023,

		D3_RAW_MOMENT_030,
		D3_RAW_MOMENT_031,
		D3_RAW_MOMENT_032,
		D3_RAW_MOMENT_033,

		D3_RAW_MOMENT_100,
		D3_RAW_MOMENT_101,
		D3_RAW_MOMENT_102,
		D3_RAW_MOMENT_103,

		D3_RAW_MOMENT_110,
		D3_RAW_MOMENT_111,
		D3_RAW_MOMENT_112,
		D3_RAW_MOMENT_113,

		D3_RAW_MOMENT_120,
		D3_RAW_MOMENT_121,
		D3_RAW_MOMENT_122,
		D3_RAW_MOMENT_123,

		D3_RAW_MOMENT_200,
		D3_RAW_MOMENT_201,
		D3_RAW_MOMENT_202,
		D3_RAW_MOMENT_203,

		D3_RAW_MOMENT_210,
		D3_RAW_MOMENT_211,
		D3_RAW_MOMENT_212,
		D3_RAW_MOMENT_213,

		D3_RAW_MOMENT_300,
		D3_RAW_MOMENT_301,
		D3_RAW_MOMENT_302,
		D3_RAW_MOMENT_303,

		// normalized raw moments
		D3_NORM_RAW_MOMENT_000, 
		D3_NORM_RAW_MOMENT_010, 
		D3_NORM_RAW_MOMENT_011, 
		D3_NORM_RAW_MOMENT_012, 
		D3_NORM_RAW_MOMENT_013, 
		D3_NORM_RAW_MOMENT_020, 
		D3_NORM_RAW_MOMENT_021, 
		D3_NORM_RAW_MOMENT_022, 
		D3_NORM_RAW_MOMENT_023, 
		D3_NORM_RAW_MOMENT_030, 
		D3_NORM_RAW_MOMENT_031, 
		D3_NORM_RAW_MOMENT_032, 
		D3_NORM_RAW_MOMENT_033, 
		D3_NORM_RAW_MOMENT_100, 
		D3_NORM_RAW_MOMENT_101, 
		D3_NORM_RAW_MOMENT_102, 
		D3_NORM_RAW_MOMENT_103, 
		D3_NORM_RAW_MOMENT_200, 
		D3_NORM_RAW_MOMENT_201, 
		D3_NORM_RAW_MOMENT_202, 
		D3_NORM_RAW_MOMENT_203, 
		D3_NORM_RAW_MOMENT_300, 
		D3_NORM_RAW_MOMENT_301, 
		D3_NORM_RAW_MOMENT_302, 
		D3_NORM_RAW_MOMENT_303, 

		// central moments
		D3_CENTRAL_MOMENT_020,
		D3_CENTRAL_MOMENT_021,
		D3_CENTRAL_MOMENT_022,
		D3_CENTRAL_MOMENT_023,

		D3_CENTRAL_MOMENT_030,
		D3_CENTRAL_MOMENT_031,
		D3_CENTRAL_MOMENT_032,
		D3_CENTRAL_MOMENT_033,

		D3_CENTRAL_MOMENT_110,
		D3_CENTRAL_MOMENT_111,
		D3_CENTRAL_MOMENT_112,
		D3_CENTRAL_MOMENT_113,

		D3_CENTRAL_MOMENT_120,
		D3_CENTRAL_MOMENT_121,
		D3_CENTRAL_MOMENT_122,
		D3_CENTRAL_MOMENT_123,

		D3_CENTRAL_MOMENT_200,
		D3_CENTRAL_MOMENT_201,
		D3_CENTRAL_MOMENT_202,
		D3_CENTRAL_MOMENT_203,

		D3_CENTRAL_MOMENT_210,
		D3_CENTRAL_MOMENT_211,
		D3_CENTRAL_MOMENT_212,
		D3_CENTRAL_MOMENT_213,

		D3_CENTRAL_MOMENT_300,
		D3_CENTRAL_MOMENT_301,
		D3_CENTRAL_MOMENT_302,
		D3_CENTRAL_MOMENT_303,

		// normalized central moments
		D3_NORM_CENTRAL_MOMENT_020,
		D3_NORM_CENTRAL_MOMENT_021,
		D3_NORM_CENTRAL_MOMENT_022,
		D3_NORM_CENTRAL_MOMENT_023,

		D3_NORM_CENTRAL_MOMENT_030,
		D3_NORM_CENTRAL_MOMENT_031,
		D3_NORM_CENTRAL_MOMENT_032,
		D3_NORM_CENTRAL_MOMENT_033,

		D3_NORM_CENTRAL_MOMENT_110,
		D3_NORM_CENTRAL_MOMENT_111,
		D3_NORM_CENTRAL_MOMENT_112,
		D3_NORM_CENTRAL_MOMENT_113,

		D3_NORM_CENTRAL_MOMENT_120,
		D3_NORM_CENTRAL_MOMENT_121,
		D3_NORM_CENTRAL_MOMENT_122,
		D3_NORM_CENTRAL_MOMENT_123,

		D3_NORM_CENTRAL_MOMENT_200,
		D3_NORM_CENTRAL_MOMENT_201,
		D3_NORM_CENTRAL_MOMENT_202,
		D3_NORM_CENTRAL_MOMENT_203,

		D3_NORM_CENTRAL_MOMENT_210,
		D3_NORM_CENTRAL_MOMENT_211,
		D3_NORM_CENTRAL_MOMENT_212,
		D3_NORM_CENTRAL_MOMENT_213,

		D3_NORM_CENTRAL_MOMENT_300,
		D3_NORM_CENTRAL_MOMENT_301,
		D3_NORM_CENTRAL_MOMENT_302,
		D3_NORM_CENTRAL_MOMENT_303,

		// special constant, not a feature
		_COUNT_
	};
}

using namespace Nyxus;

/// @brief Helper class to set and access user feature selection made via the command line or Python interface.
class FeatureSet
{
public:
	FeatureSet();
	void enableAll (bool newStatus = true) { for (int i = 0; i < AvailableFeatures::_COUNT_; i++) m_enabledFeatures[i] = newStatus; }
	void disableFeatures (std::initializer_list<AvailableFeatures>& desiredFeatures)
	{
		for (auto f : desiredFeatures)
			m_enabledFeatures[f] = false;
	}
	void enableFeatures(std::initializer_list<AvailableFeatures>& desiredFeatures) {
		for (auto f : desiredFeatures)
			m_enabledFeatures[f] = true;
	}
	void enableFeature(AvailableFeatures f) {
		m_enabledFeatures[f] = true;
	}
	void enablePixelIntenStats() {
		enableAll(false);
		m_enabledFeatures[MEAN] =
			m_enabledFeatures[MEDIAN] =
			m_enabledFeatures[MIN] =
			m_enabledFeatures[MAX] =
			m_enabledFeatures[RANGE] =
			m_enabledFeatures[STANDARD_DEVIATION] =
			m_enabledFeatures[SKEWNESS] =
			m_enabledFeatures[KURTOSIS] =
			m_enabledFeatures[MEAN_ABSOLUTE_DEVIATION] =
			m_enabledFeatures[ENERGY] =
			m_enabledFeatures[ROOT_MEAN_SQUARED] =
			m_enabledFeatures[ENTROPY] =
			m_enabledFeatures[MODE] =
			m_enabledFeatures[UNIFORMITY] =
			m_enabledFeatures[P10] = m_enabledFeatures[P25] = m_enabledFeatures[P75] = m_enabledFeatures[P90] =
			m_enabledFeatures[INTERQUARTILE_RANGE] =
			m_enabledFeatures[ROBUST_MEAN_ABSOLUTE_DEVIATION] =
			m_enabledFeatures[WEIGHTED_CENTROID_Y] =
			m_enabledFeatures[WEIGHTED_CENTROID_X] =
			m_enabledFeatures[MASS_DISPLACEMENT] = true;
	}
	void enableBoundingBox() {
		enableAll(false);
		m_enabledFeatures[BBOX_YMIN] =
			m_enabledFeatures[BBOX_XMIN] =
			m_enabledFeatures[BBOX_HEIGHT] =
			m_enabledFeatures[BBOX_WIDTH] = true;
	}
	void enableFeret() {
		enableAll(false);
		m_enabledFeatures[MIN_FERET_DIAMETER] =
			m_enabledFeatures[MAX_FERET_DIAMETER] =
			m_enabledFeatures[MIN_FERET_ANGLE] =
			m_enabledFeatures[MAX_FERET_ANGLE] =
			m_enabledFeatures[STAT_FERET_DIAM_MIN] =
			m_enabledFeatures[STAT_FERET_DIAM_MAX] =
			m_enabledFeatures[STAT_FERET_DIAM_MEAN] =
			m_enabledFeatures[STAT_FERET_DIAM_MEDIAN] =
			m_enabledFeatures[STAT_FERET_DIAM_STDDEV] =
			m_enabledFeatures[STAT_FERET_DIAM_MODE] = true;
	}
	bool isEnabled(int fc) const { return fc < AvailableFeatures::_COUNT_ ? m_enabledFeatures[fc] : false; }
	bool anyEnabled(std::initializer_list<int> F) const
	{
		for (auto f : F)
			if (m_enabledFeatures[f])
				return true;
		return false;
	}
	int numOfEnabled() {
		int cnt = 0;
		for (int i = 0; i < AvailableFeatures::_COUNT_; i++)
			if (m_enabledFeatures[i])
				cnt++;
		return cnt;
	}
	bool findFeatureByString (const std::string& featureName, AvailableFeatures& fcode);
	std::string findFeatureNameByCode (AvailableFeatures fcode);
	void show_help();

	// Relying on RVO rather than std::move
	std::vector<std::tuple<std::string, AvailableFeatures>> getEnabledFeatures();

private:
	bool m_enabledFeatures[AvailableFeatures::_COUNT_];

	//==== 2D and 3D 
public:
	bool need_2d() const	{ return need_2d_; }
	void need_2d(bool need)	{ need_2d_ = need; }
	bool need_3d() const	{ return need_3d_; }
	void need_3d(bool need)	{ need_3d_ = need; }
	bool is_2d(int fcode)	{ return fcode >= FIRST_2D_FEATURE && fcode <= LAST_2D_FEATURE; }
	bool is_3d(int fcode)	{ return fcode >= FIRST_3D_FEATURE && fcode <= LAST_3D_FEATURE; }
private:
	bool need_2d_ = false;
	bool need_3d_ = false;
	int FIRST_2D_FEATURE = AvailableFeatures::INTEGRATED_INTENSITY;
	int LAST_2D_FEATURE = AvailableFeatures::GABOR;
	int FIRST_3D_FEATURE = AvailableFeatures::D3_INTEGRATED_INTENSITY;
	int LAST_3D_FEATURE = AvailableFeatures::D3_CENTRAL_MOMENT_303;
};

namespace Nyxus
{
	extern FeatureSet theFeatureSet;
	extern std::map <std::string, AvailableFeatures> UserFacingFeatureNames;
}
