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
			DIAMETER_EQUAL_PERIMETER, 
			EDGE_INTEGRATED_INTENSITY,
			EDGE_MAX_INTENSITY,
			EDGE_MIN_INTENSITY,
			EDGE_MEAN_INTENSITY,
			EDGE_STDDEV_INTENSITY,
			// dependencies:
			CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY, 
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
			ROI_RADIUS_MEAN, ROI_RADIUS_MAX, ROI_RADIUS_MEDIAN,
			ZERNIKE2D, FRAC_AT_D, MEAN_FRAC, RADIAL_CV
		});
	}

private:
	void buildRegularContour(LR& r);
	void buildRegularContour_nontriv(LR& r);
	void buildWholeSlideContour(LR& r);
	std::tuple<double, double, double, double> calc_min_max_mean_stddev_intensity (const std::vector<Pixel2> & contour_pixels);
	double
		fval_PERIMETER = 0, 
		fval_DIAMETER_EQUAL_PERIMETER = 0,
		fval_EDGE_MEAN_INTENSITY = 0, 
		fval_EDGE_STDDEV_INTENSITY = 0, 
		fval_EDGE_MAX_INTENSITY = 0, 
		fval_EDGE_MIN_INTENSITY = 0,
		fval_EDGE_INTEGRATEDINTENSITY = 0 
		//,fval_EDGE_MAXINTENSITY = 0, 
		//fval_EDGE_MININTENSITY = 0, 
		//fval_EDGE_MEANINTENSITY = 0 
		//,fval_EDGE_STDDEVINTENSITY = 0
		;
};

