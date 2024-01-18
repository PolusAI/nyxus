#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include <tuple>
#include <vector>
#include "pixel.h"
#include "../feature_method.h"

/// @brief Class encapsulating circularity features of a ROI

class EnclosingInscribingCircumscribingCircleFeature: public FeatureMethod
{
public:
	EnclosingInscribingCircumscribingCircleFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Compatibility with manual reduce
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled ({ 
			Nyxus::Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE, 
			Nyxus::Feature2D::DIAMETER_INSCRIBING_CIRCLE, 
			Nyxus::Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE });
	}

private:
	double calculate_min_enclosing_circle_diam(std::vector<Pixel2>& Contour);
	std::tuple <double, double> calculate_inscribing_circumscribing_circle(std::vector<Pixel2>& contours, double xCentroid, double yCentroid);
	void minEnclosingCircle(
		// in:
		std::vector<Pixel2>& Contour,
		// out:
		Point2f& center,
		float& radius);
	void findMinEnclosingCircle(const std::vector<Pixel2>& contour, Point2f& center, float& radius);
	void findSecondPoint(const std::vector<Pixel2>& pts, int i, Point2f& center, float& radius);
	void findThirdPoint(const std::vector<Pixel2>& pts, int i, int j, Point2f& center, float& radius);
	void findCircle3pts(const std::vector<Pixel2>& pts, Point2f& center, float& radius);
	double d_minEnclo = 0, d_inscr = 0, d_circum = 0;
	const float EPS = 1.0e-4f;
};
