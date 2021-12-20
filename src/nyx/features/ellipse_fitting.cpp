#define _USE_MATH_DEFINES	// for M_PI, etc
#include <cmath>
#include "ellipse_fitting.h"

// Inspired by https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/19028/versions/1/previews/regiondata.m/index.html

EllipseFittingFeatures::EllipseFittingFeatures (const std::vector<Pixel2>& roi_pixels, double centroid_x, double centroid_y, double area)
{
	// Idea: calculate normalized second central moments for the region. 1/12 is the normalized second central moment of a pixel with unit length.

	double xSquaredTmp = 0, 
		ySquaredTmp = 0, 
		xySquaredTmp = 0;

	for (auto& pix : roi_pixels)
	{
		auto diffX = centroid_x - pix.x,
			diffY = centroid_y - pix.y;
		xSquaredTmp += diffX * diffX;
		ySquaredTmp += diffY * diffY;
		xySquaredTmp += diffX * diffY;
	}

	double n = (double) roi_pixels.size();
	double uxx = xSquaredTmp / n + 1. / 12.;
	double uyy = ySquaredTmp / n + 1. / 12.;
	double uxy = xySquaredTmp / n;

	// Calculate major axis length, minor axis length, and eccentricity.
	double common = sqrt((uxx - uyy) * (uxx - uyy) + 4. * uxy * uxy);
	majorAxisLength = 2. * sqrt(2.) * sqrt(uxx + uyy + common);
	minorAxisLength = 2. * sqrt(2.) * sqrt(uxx + uyy - common);
	eccentricity = 2. * sqrt((majorAxisLength / 2.) * (majorAxisLength / 2.) - (minorAxisLength / 2.) * (minorAxisLength / 2.)) / majorAxisLength;
	roundness = (4. * area) / (M_PI * majorAxisLength * majorAxisLength);

	// Calculate orientation [-90,90]
	double num, den;
	if (uyy > uxx) 
	{
		num = uyy - uxx + sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
		den = 2 * uxy;
	}
	else 
	{
		num = 2 * uxy;
		den = uxx - uyy + sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
	}

	if (num == 0 && den == 0)
		orientation = 0;
	else
		orientation = (180.0 / M_PI) * atan(num / den);
}

double EllipseFittingFeatures::get_major_axis_length()
{
	return majorAxisLength;
}

double EllipseFittingFeatures::get_minor_axis_length()
{
	return minorAxisLength;
}

double EllipseFittingFeatures::get_eccentricity()
{
	return eccentricity;
}

double EllipseFittingFeatures::get_orientation()
{
	return orientation;
}

double EllipseFittingFeatures::get_roundness()
{
	return roundness;
}

void EllipseFittingFeatures::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		EllipseFittingFeatures f (r.raw_pixels, r.fvals[CENTROID_X][0], r.fvals[CENTROID_Y][0], r.fvals[AREA_PIXELS_COUNT][0]);
		r.fvals[MAJOR_AXIS_LENGTH][0] = f.get_major_axis_length();
		r.fvals[MINOR_AXIS_LENGTH][0] = f.get_minor_axis_length();
		r.fvals[ECCENTRICITY][0] = f.get_eccentricity();
		r.fvals[ORIENTATION][0] = f.get_orientation();
		r.fvals[ROUNDNESS][0] = f.get_roundness();
	}
}
