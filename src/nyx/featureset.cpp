#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "sensemaker.h"


#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif



double LR::getValue (AvailableFeatures f)
{
	double v = -111.111;	// Default uninitialized value

	switch (f)
	{
		// Pixel intensity stats
		case MEAN: v = this->mean; break;
		case MEDIAN: v=this->labelMedians; break;
		case MIN: v=this->labelMins; break;
		case MAX: v=this->labelMaxs; break;
		case RANGE: v=this->labelMaxs-this->labelMins; break;
		case STANDARD_DEVIATION: v=this->labelStddev; break;
		case SKEWNESS: v=this->labelSkewness; break;
		case KURTOSIS: v=this->labelKurtosis; break;
		case MEAN_ABSOLUTE_DEVIATION: v=this->labelMAD; break;
		case ENERGY: v=this->labelMassEnergy; break;
		case ROOT_MEAN_SQUARED: v=this->labelRMS; break;
		case ENTROPY: v=this->labelEntropy; break;
		case MODE: v=this->labelMode; break;
		case UNIFORMITY: v=this->labelUniformity; break;
		case P10: v=this->labelP10; break; 
		case P25: v=this->labelP25; break; 
		case P75: v=this->labelP75; break; 
		case P90: v=this->labelP90; break;
		case INTERQUARTILE_RANGE: v=this->labelIQR; break;
		case ROBUST_MEAN_ABSOLUTE_DEVIATION: v=this->labelMAD; break;
		case WEIGHTED_CENTROID_Y: v=this->centroid_y; break;
		case WEIGHTED_CENTROID_X: v=this->centroid_x; break;

		// Morphology: v=this->; break;
		case AREA_PIXELS_COUNT: v=this->pixelCount; break;
		case CENTROID_X: v=this->centroid_x; break;
		case CENTROID_Y: v=this->centroid_y; break;
		case BBOX_YMIN: v=this->aabb.get_ymin(); break;
		case BBOX_XMIN: v=this->aabb.get_xmin(); break;
		case BBOX_HEIGHT: v=this->aabb.get_height(); break;
		case BBOX_WIDTH: v=this->aabb.get_width(); break;

		case MAJOR_AXIS_LENGTH: v=this->major_axis_length; break;
		case MINOR_AXIS_LENGTH: v=this->minor_axis_length; break;
		case ECCENTRICITY: v=this->eccentricity; break;
		case ORIENTATION: v=this->orientation; break;
		case NUM_NEIGHBORS: v=this->num_neighbors; break;
		case EXTENT: v=this->extent; break;
		case ASPECT_RATIO: v=this->aspectRatio; break;

		case EQUIVALENT_DIAMETER: v=this->equivDiam; break;
		case CONVEX_HULL_AREA: v=this->convHullArea; break;
		case SOLIDITY: v=this->solidity; break;
		case PERIMETER: v=this->roiPerimeter; break;
		case CIRCULARITY: v=this->circularity; break;

		case EXTREMA_P1_X: v=this->extremaP1x; break; 
		case EXTREMA_P1_Y: v=this->extremaP1y; break;
		case EXTREMA_P2_X: v=this->extremaP2x; break;
		case EXTREMA_P2_Y: v=this->extremaP2y; break;
		case EXTREMA_P3_X: v=this->extremaP3x; break;
		case EXTREMA_P3_Y: v=this->extremaP3y; break;
		case EXTREMA_P4_X: v=this->extremaP4x; break;
		case EXTREMA_P4_Y: v=this->extremaP4y; break;
		case EXTREMA_P5_X: v=this->extremaP5x; break;
		case EXTREMA_P5_Y: v=this->extremaP5y; break;
		case EXTREMA_P6_X: v=this->extremaP6x; break;
		case EXTREMA_P6_Y: v=this->extremaP6y; break;
		case EXTREMA_P7_X: v=this->extremaP7x; break;
		case EXTREMA_P7_Y: v=this->extremaP7y; break;
		case EXTREMA_P8_X: v=this->extremaP8x; break;
		case EXTREMA_P8_Y: v=this->extremaP8y; break;

		case MIN_FERET_DIAMETER: v=this->minFeretDiameter; break;
		case MAX_FERET_DIAMETER: v=this->maxFeretDiameter; break;
		case MIN_FERET_ANGLE: v=this->minFeretAngle; break;
		case MAX_FERET_ANGLE: v=this->maxFeretAngle; break;
		case STAT_FERET_DIAM_MIN: v=this->feretStats_minDiameter; break;
		case STAT_FERET_DIAM_MAX: v=this->feretStats_maxDiameter; break;
		case STAT_FERET_DIAM_MEAN: v=this->feretStats_meanDiameter; break;
		case STAT_FERET_DIAM_MEDIAN: v=this->feretStats_medianDiameter; break;
		case STAT_FERET_DIAM_STDDEV: v=this->feretStats_stddevDiameter; break;
		case STAT_FERET_DIAM_MODE: v=this->feretStats_modeDiameter; break;

		case STAT_MARTIN_DIAM_MIN: v=this->martinStats_minDiameter; break;
		case STAT_MARTIN_DIAM_MAX: v=this->martinStats_maxDiameter; break;
		case STAT_MARTIN_DIAM_MEAN: v=this->martinStats_meanDiameter; break;
		case STAT_MARTIN_DIAM_MEDIAN: v=this->martinStats_medianDiameter; break;
		case STAT_MARTIN_DIAM_STDDEV: v=this->martinStats_stddevDiameter; break;
		case STAT_MARTIN_DIAM_MODE: v=this->martinStats_modeDiameter; break;

		case STAT_NASSENSTEIN_DIAM_MIN: v=this->nassStats_minDiameter; break;
		case STAT_NASSENSTEIN_DIAM_MAX: v=this->nassStats_maxDiameter; break;
		case STAT_NASSENSTEIN_DIAM_MEAN: v=this->nassStats_meanDiameter; break;
		case STAT_NASSENSTEIN_DIAM_MEDIAN: v=this->nassStats_medianDiameter; break;
		case STAT_NASSENSTEIN_DIAM_STDDEV: v=this->nassStats_stddevDiameter; break;
		case STAT_NASSENSTEIN_DIAM_MODE: v=this->nassStats_modeDiameter; break;

		case EULER_NUBER: v=this->euler_number; break;

		case POLYGONALITY_AVE: v=this->polygonality_ave; break;
		case HEXAGONALITY_AVE: v=this->hexagonality_ave; break;
		case HEXAGONALITY_STDDEV: v=this->hexagonality_stddev; break;

		case DIAMETER_MIN_ENCLOSING_CIRCLE: v=this->diameter_min_enclosing_circle; break;
		case DIAMETER_CIRCUMSCRIBING_CIRCLE: v=this->diameter_circumscribing_circle; break;
		case DIAMETER_INSCRIBING_CIRCLE: v=this->diameter_inscribing_circle; break;
		case GEODETIC_LENGTH: v=this->geodeticLength; break;
		case THICKNESS: v=this->thickness; break;
		default: v = -999.999;  break;	// default unknown value
	}

	return v;
}
