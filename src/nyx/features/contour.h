#pragma once
#include "../featureset.h"
#include "image_matrix.h"
#include "../feature_method.h"

/// @brief A contour is a vector of X and Y coordinates of all the pixels on the border of a ROI. This class uses Moore's algorithm for cnotour detection.
class ContourFeature: public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::PERIMETER,
		Nyxus::Feature2D::DIAMETER_EQUAL_PERIMETER,
		Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY,
		Nyxus::Feature2D::EDGE_MAX_INTENSITY,
		Nyxus::Feature2D::EDGE_MIN_INTENSITY,
		Nyxus::Feature2D::EDGE_MEAN_INTENSITY,
		Nyxus::Feature2D::EDGE_STDDEV_INTENSITY
	};

	ContourFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	static bool required(const FeatureSet& fs);

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
		fval_EDGE_INTEGRATEDINTENSITY = 0;
};

