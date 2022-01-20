#pragma once
#include "../featureset.h"
#include "image_matrix.h"
#include "../feature_method.h"

/// @brief A contour is a vector of X and Y coordinates of all the pixels on the border of a ROI. This class uses Moore's algorithm for cnotour detection.
class ContourFeature: public FeatureMethod
{
public:
	ContourFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();

	#if 0
	void calculate (const ImageMatrix& im);	// Leaves result in 'contour_pixels'
	StatsInt get_roi_perimeter();
	StatsReal get_diameter_equal_perimeter();
	std::tuple<StatsReal, StatsReal, StatsReal, StatsReal> get_min_max_mean_stddev_intensity();
	std::vector<Pixel2> contour_pixels;
	#endif

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

private:
	std::tuple<double, double, double, double> calc_min_max_mean_stddev_intensity (const std::vector<Pixel2> & contour_pixels);
	double
		fval_PERIMETER = 0, 
		fval_EQUIVALENT_DIAMETER = 0, 
		fval_EDGE_MEAN_INTENSITY = 0, 
		fval_EDGE_STDDEV_INTENSITY = 0, 
		fval_EDGE_MAX_INTENSITY = 0, 
		fval_EDGE_MIN_INTENSITY = 0,
		fval_EDGE_INTEGRATEDINTENSITY = 0, 
		fval_EDGE_MAXINTENSITY = 0, 
		fval_EDGE_MININTENSITY = 0, 
		fval_EDGE_MEANINTENSITY = 0, 
		fval_EDGE_STDDEVINTENSITY = 0;
};

