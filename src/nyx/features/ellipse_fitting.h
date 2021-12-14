#pragma once
#include <unordered_map>
#include "../roi_data.h"
#include "pixel.h"

class EllipseFittingFeatures
{
public:

	EllipseFittingFeatures();
	void initialize (const std::vector<Pixel2> & roi_pixels, double centroid_x, double centroid_y, double area);
	double get_major_axis_length();
	double get_minor_axis_length();
	double get_eccentricity();
	double get_orientation();
	double get_roundness();

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:

	double majorAxisLength = 0,
		minorAxisLength = 0,
		eccentricity = 0,
		orientation = 0, 
		roundness = 0;
};
