#pragma once

#include <unordered_map>
#include "../roi_data.h"
#include <tuple>
#include <vector>
#include "pixel.h"

/// @brief Class encapsulating circularity features of a ROI

class EnclosingInscribingCircumscribingCircle
{
public:
	static bool required(const FeatureSet& fs) {
		return fs.anyEnabled({ DIAMETER_MIN_ENCLOSING_CIRCLE, DIAMETER_INSCRIBING_CIRCLE, DIAMETER_CIRCUMSCRIBING_CIRCLE });
	}

	EnclosingInscribingCircumscribingCircle() {}
	double calculate_min_enclosing_circle_diam (std::vector<Pixel2>& Contour);
	std::tuple <double, double> calculate_inscribing_circumscribing_circle (std::vector<Pixel2>& contours, double xCentroid, double yCentroid);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	const float EPS = 1.0e-4f;

private:
	void minEnclosingCircle(
		// in:
		std::vector<Pixel2>& Contour,
		// out:
		Point2f& center,
		float& radius);
	void findMinEnclosingCircle(const std::vector<Pixel2>& pts, int count, Point2f& center, float& radius);
	void findSecondPoint(const std::vector<Pixel2>& pts, int i, Point2f& center, float& radius);
	void findThirdPoint(const std::vector<Pixel2>& pts, int i, int j, Point2f& center, float& radius);
	void findCircle3pts(const std::vector<Pixel2>& pts, Point2f& center, float& radius);
};
