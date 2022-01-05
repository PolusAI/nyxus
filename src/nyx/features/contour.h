#pragma once
#include "../featureset.h"
#include "image_matrix.h"

/// @brief A contour is a vector of X and Y coordinates of all the pixels on the border of a ROI. This class uses Moore's algorithm for cnotour detection.
class Contour
{
public:
	Contour();
	void calculate (const ImageMatrix& im);	// Leaves result in 'contour_pixels'
	StatsInt get_roi_perimeter();
	StatsReal get_diameter_equal_perimeter();
	std::tuple<StatsReal, StatsReal, StatsReal, StatsReal> get_min_max_mean_stddev_intensity();
	void clear()
	{
		contour_pixels.clear();
	}
	
	std::vector<Pixel2> contour_pixels;

	static bool required(const FeatureSet& fs) 
	{
		return theFeatureSet.anyEnabled({
			PERIMETER,
			EQUIVALENT_DIAMETER,
			EDGE_INTEGRATEDINTENSITY,
			EDGE_MAXINTENSITY,
			EDGE_MININTENSITY,
			EDGE_MEANINTENSITY,
			EDGE_STDDEVINTENSITY,
			// dependencies:
			CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY
			});
	}
};

