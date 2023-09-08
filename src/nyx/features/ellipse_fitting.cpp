#define _USE_MATH_DEFINES
#include <cmath>
#include "ellipse_fitting.h"

// Inspired by https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/19028/versions/1/previews/regiondata.m/index.html

EllipseFittingFeature::EllipseFittingFeature() : FeatureMethod("EllipseFittingFeature")
{
	provide_features({
		MAJOR_AXIS_LENGTH,
		MINOR_AXIS_LENGTH,
		ECCENTRICITY,
		ELONGATION,
		ORIENTATION,
		ROUNDNESS });
}

//EllipseFittingFeature::EllipseFittingFeature(const std::vector<Pixel2>& roi_pixels, double centroid_x, double centroid_y, double area)
void EllipseFittingFeature::calculate (LR& r)
{
	// Idea: calculate normalized second central moments for the region. 1/12 is the normalized second central moment of a pixel with unit length.

	double centroid_x = r.fvals[CENTROID_X][0],
		centroid_y = r.fvals[CENTROID_Y][0],
		area = r.fvals[AREA_PIXELS_COUNT][0];

	double xSquaredTmp = 0,
		ySquaredTmp = 0,
		xySquaredTmp = 0;

	for (const auto& pix : r.raw_pixels)
	{
		auto diffX = centroid_x - pix.x,
			diffY = centroid_y - pix.y;
		xSquaredTmp += diffX * diffX;
		ySquaredTmp += diffY * diffY;
		xySquaredTmp += diffX * diffY;
	}

	double n = (double) r.raw_pixels.size();
	double uxx = xSquaredTmp / n + 1. / 12.;
	double uyy = ySquaredTmp / n + 1. / 12.;
	double uxy = xySquaredTmp / n;

	// Calculate major axis length, minor axis length, and eccentricity.
	double common = sqrt((uxx - uyy) * (uxx - uyy) + 4. * uxy * uxy);
	majorAxisLength = 2. * sqrt(2.) * sqrt(uxx + uyy + common);
	minorAxisLength = 2. * sqrt(2.) * sqrt(uxx + uyy - common);
	eccentricity = sqrt(1.0 - majorAxisLength* majorAxisLength / (minorAxisLength* minorAxisLength));
	elongation = sqrt(minorAxisLength/majorAxisLength);
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

	if (uxy == 0.)
	{
		if (uxx >= uyy)
			orientation = 0.;
		else
			orientation = 90.;
	}
	else
		orientation = 180./M_PI * atan(num/den);
}

double EllipseFittingFeature::get_major_axis_length()
{
	return majorAxisLength;
}

double EllipseFittingFeature::get_minor_axis_length()
{
	return minorAxisLength;
}

double EllipseFittingFeature::get_eccentricity()
{
	return eccentricity;
}

double EllipseFittingFeature::get_elongation()
{
	return elongation;
}

double EllipseFittingFeature::get_orientation()
{
	return orientation;
}

double EllipseFittingFeature::get_roundness()
{
	return roundness;
}

void EllipseFittingFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		EllipseFittingFeature f;
		f.calculate (r);
		f.save_value (r.fvals);
	}
}

void EllipseFittingFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	// Idea: calculate normalized second central moments for the region. 1/12 is the normalized second central moment of a pixel with unit length.

	double centroid_x = r.fvals[CENTROID_X][0],
		centroid_y = r.fvals[CENTROID_Y][0],
		area = r.fvals[AREA_PIXELS_COUNT][0];

	double xSquaredTmp = 0,
		ySquaredTmp = 0,
		xySquaredTmp = 0;

	for (const auto& pix : r.raw_pixels_NT)
	{
		auto diffX = centroid_x - pix.x,
			diffY = centroid_y - pix.y;
		xSquaredTmp += diffX * diffX;
		ySquaredTmp += diffY * diffY;
		xySquaredTmp += diffX * diffY;
	}

	double n = (double) r.raw_pixels_NT.size();
	double uxx = xSquaredTmp / n + 1. / 12.;
	double uyy = ySquaredTmp / n + 1. / 12.;
	double uxy = xySquaredTmp / n;

	// Calculate major axis length, minor axis length, and eccentricity.
	double common = sqrt((uxx - uyy) * (uxx - uyy) + 4. * uxy * uxy);
	majorAxisLength = 2. * sqrt(2.) * sqrt(uxx + uyy + common);
	minorAxisLength = 2. * sqrt(2.) * sqrt(uxx + uyy - common);
	eccentricity = sqrt(1.0 - majorAxisLength* majorAxisLength / (minorAxisLength* minorAxisLength));
	elongation = sqrt(minorAxisLength/majorAxisLength);
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

	if (uxy == 0.)
	{
		if (uxx >= uyy)
			orientation = 0.;
		else
			orientation = 90.;
	}
	else
		orientation = 180./M_PI * atan(num/den);
}

void EllipseFittingFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[MAJOR_AXIS_LENGTH][0] = get_major_axis_length();
	fvals[MINOR_AXIS_LENGTH][0] = get_minor_axis_length();
	fvals[ECCENTRICITY][0] = get_eccentricity();
	fvals[ELONGATION][0] = get_elongation();
	fvals[ORIENTATION][0] = get_orientation();
	fvals[ROUNDNESS][0] = get_roundness();
}
