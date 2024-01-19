#pragma once

#include <unordered_map>
#include "../feature_method.h"
#include "../featureset.h"
#include "../roi_cache.h"
#include "pixel.h"

/// @brief Class encapsulating Legendre's ellipse of inertia of ROI pixels (ellipse that has the same normalized second central moments as the particle shape).
class EllipseFittingFeature: public FeatureMethod
{
public:
	EllipseFittingFeature();

	// Trivial ROI
	void calculate (LR& r);	

	// Non-trivial ROI
	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {}
	void osized_calculate(LR& r, ImageLoader& imloader);	

	// Result saver
	void save_value(std::vector<std::vector<double>>& feature_vals);



	/// @brief Major axis length of Legendre's ellipse of inertia
	/// @return 
	double get_major_axis_length();

	/// @brief M axis length of Legendre's ellipse of inertia
	/// @return 
	double get_minor_axis_length();

	/// @brief The eccentricity of an ellipse is the ratio of the distance c between the center of the ellipse and each focus to the length of the semimajor axis a: 
	/// @return 
	double get_eccentricity();

	/// @brief Returns ellipse elongation=\frac{minor_axis_length}{major_axis_length} in the form of true inverse elongation 
	/// @return Value of the ellipse elongation feature
	double get_elongation();

	/// @brief Orientation describes whether the ellipse is horizontal or vertical
	/// @return 
	double get_orientation();

	/// @brief Roundness is calculated as (4 \times Area) / (\pi \times MajorAxis^2). Roundness should not be confused with circularity as the latter captures perimeter smoothness and not overall structural shape.
	/// @return 
	double get_roundness();

	static bool required (const FeatureSet& fs); 
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:

	double majorAxisLength = 0,
		minorAxisLength = 0,
		eccentricity = 0,
		elongation = 0, 
		orientation = 0, 
		roundness = 0;
};
