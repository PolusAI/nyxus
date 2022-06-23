#pragma once

#include <map>
#include <vector>

namespace Nyxus
{
	/// @brief Feature codes
	enum AvailableFeatures
	{
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
		WEIGHTED_CENTROID_Y,
		WEIGHTED_CENTROID_X,
		MASS_DISPLACEMENT,

		// Morphology:
		AREA_PIXELS_COUNT,
		AREA_UM2,
		CENTROID_X,
		CENTROID_Y,
		COMPACTNESS,
		BBOX_YMIN,
		BBOX_XMIN,
		BBOX_HEIGHT,
		BBOX_WIDTH,
		EXTENT,
		ASPECT_RATIO,

		// -- ellipticity related
		MAJOR_AXIS_LENGTH,
		MINOR_AXIS_LENGTH,
		ECCENTRICITY,
		ELONGATION,
		ORIENTATION,
		ROUNDNESS,

		// -- contour related
		PERIMETER,
		EQUIVALENT_DIAMETER,
		EDGE_MEAN_INTENSITY,
		EDGE_STDDEV_INTENSITY,
		EDGE_MAX_INTENSITY,
		EDGE_MIN_INTENSITY,
		EDGE_INTEGRATEDINTENSITY,	
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

		// -- neighboring ROI features
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
		GLCM_VARIANCE,
		GLCM_INVERSEDIFFERENCEMOMENT,
		GLCM_SUMAVERAGE,
		GLCM_SUMVARIANCE,
		GLCM_SUMENTROPY,
		GLCM_ENTROPY,
		GLCM_DIFFERENCEVARIANCE,
		GLCM_DIFFERENCEENTROPY,
		GLCM_INFOMEAS1,
		GLCM_INFOMEAS2,

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
};

namespace Nyxus
{
	extern FeatureSet theFeatureSet;
	extern std::map <std::string, AvailableFeatures> UserFacingFeatureNames;
}
