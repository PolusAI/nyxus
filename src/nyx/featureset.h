#pragma once

enum AvailableFeatures
{
	// Pixel intensity stats
	MEAN = 0, 
	MEDIAN,  
	MIN,  
	MAX,  
	RANGE,  
	STANDARD_DEVIATION,  
	SKEWNESS,  
	KURTOSIS,  
	MEAN_ABSOLUTE_DEVIATION,  
	ENERGY,  
	ROOT_MEAN_SQUARED,  
	ENTROPY,  
	MODE,  
	UNIFORMITY,  
	P10, P25, P75, P90, 
	INTERQUARTILE_RANGE,   
	ROBUST_MEAN_ABSOLUTE_DEVIATION,   
	WEIGHTED_CENTROID_Y,   
	WEIGHTED_CENTROID_X,  

	// Morphology:
	AREA_PIXELS_COUNT,
	CENTROID_X,  
	CENTROID_Y,  
	BBOX_YMIN,  
	BBOX_XMIN,  
	BBOX_HEIGHT,  
	BBOX_WIDTH,  

	MAJOR_AXIS_LENGTH,	// ellipticity  
	MINOR_AXIS_LENGTH,  // ellipticity
	ECCENTRICITY,		// ellipticity
	ORIENTATION,		// ellipticity

	NUM_NEIGHBORS,  
	
	EXTENT,  
	ASPECT_RATIO,  

	EQUIVALENT_DIAMETER,  
	CONVEX_HULL_AREA,  
	SOLIDITY,  
	PERIMETER,  
	CIRCULARITY,  

	EXTREMA_P1_X, EXTREMA_P1_Y,  
	EXTREMA_P2_X, EXTREMA_P2_Y,  
	EXTREMA_P3_X, EXTREMA_P3_Y,  
	EXTREMA_P4_X, EXTREMA_P4_Y,  
	EXTREMA_P5_X, EXTREMA_P5_Y,  
	EXTREMA_P6_X, EXTREMA_P6_Y,  
	EXTREMA_P7_X, EXTREMA_P7_Y,  
	EXTREMA_P8_X, EXTREMA_P8_Y, 

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

	EULER_NUBER, 

	POLYGONALITY_AVE,  
	HEXAGONALITY_AVE,  
	HEXAGONALITY_STDDEV,  

	DIAMETER_MIN_ENCLOSING_CIRCLE,  
	DIAMETER_CIRCUMSCRIBING_CIRCLE,  
	DIAMETER_INSCRIBING_CIRCLE,  
	GEODETIC_LENGTH,  
	THICKNESS,

	// Texture:
	TEXTURE_HARALICK2D, 
	TEXTURE_ZERNIKE2D,

	_COUNT_
};


class FeatureSet
{
public:
	FeatureSet() { enableAll(true); }
	void enableAll(bool newStatus=true) { for (int i = 0; i < AvailableFeatures::_COUNT_; i++) m_enabledFeatures[i] = newStatus;	}
	void disableFeatures(std::initializer_list<AvailableFeatures>& desiredFeatures)
	{
		for (auto f : desiredFeatures)
			m_enabledFeatures[f] = false;
	}
	void enableFeatures(std::initializer_list<AvailableFeatures>& desiredFeatures) {
		for (auto f : desiredFeatures)
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
		m_enabledFeatures[WEIGHTED_CENTROID_X] = true;
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
	bool isEnabled (int fc) { return fc < AvailableFeatures::_COUNT_ ? m_enabledFeatures[fc] : false; }
	bool anyEnabled (std::initializer_list<int> F)
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
protected:
	bool m_enabledFeatures [AvailableFeatures::_COUNT_];
};