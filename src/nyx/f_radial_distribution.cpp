#include <sstream>

#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>

#include "f_radial_distribution.h"
#include "image_matrix.h"
#include "sensemaker.h"

void RadialDistribution::initialize(const std::vector<Pixel2>& raw_pixels, const std::vector<Pixel2>& contour_pixels)
{
	// Cache the pixels count
	this->cached_num_pixels = raw_pixels.size();

	// Find the center (most distant pixel from the edge)
	int idxO = Pixel2::find_center(raw_pixels, contour_pixels);

	// Cache it
	this->cached_center_x = raw_pixels[idxO].x;
	this->cached_center_y = raw_pixels[idxO].y;

	// Get ahold of the center pixel
	const Pixel2& pxO = raw_pixels[idxO];

	// Distribute pixels into radial bins
	double binWidth = 1.0 / double(num_bins-1);
	for (auto& pxA : raw_pixels)
	{
		// If 'px' is a contour point, skip it
		if (pxA.belongs_to (contour_pixels))
			continue;

		// Find the contour point
		int idxCont = -1; // Pixel2& pxContour = conv_hull.CH[0];
		double distToRadius;

		for (int i=0; i< contour_pixels.size(); i++)
		{
			const Pixel2& pxC = contour_pixels[i];
			double dAC = pxA.sqdist(pxC);
			double dOC = pxO.sqdist(pxC);
			double dOA = pxO.sqdist(pxA);
			if (dOC < dAC || dOC < dOA)
				continue;	// Perpendicular from A onto OC is situated beyond OC - skip this degenerate case

			double dA_OC = pxA.sqdist_to_segment(pxC, pxO);
			if (idxCont<0 || dA_OC < distToRadius)
			{
				idxCont = i;
				distToRadius = dA_OC;
			}
		}
		const Pixel2& pxContour = contour_pixels[idxCont];

		// Distance center to cloud pixel
		double dstOA = std::sqrt (pxA.sqdist(pxO));

		// Distance center to contour
		double dstOC = std::sqrt (pxContour.sqdist(pxO));

		// Distance contour to pixel
		double dstAC = std::sqrt (pxContour.sqdist(pxA));

		// Intercept an error or weird condition
		if (dstOC < dstAC || dstOC < dstOA)
		{
			// Show A
			std::stringstream ss;
			if (dstOC < dstAC)
				ss << theIntFname << " Weird: OC=" << dstOC << " < AC=" << dstAC << ". Points O(" << pxO.x << "," << pxO.y << "), A(" << pxA.x << "," << pxA.y << "), and C(" << pxContour.x << "," << pxContour.y << ")";
			if (dstOC < dstOA)
				ss << theIntFname << " Weird: OC=" << dstOC << " < OA=" << dstOA << ". Points O(" << pxO.x << "," << pxO.y << "), A(" << pxA.x << "," << pxA.y << "), and C(" << pxContour.x << "," << pxContour.y << ")";
			ImageMatrix imCont (contour_pixels);
			imCont.print(ss.str(), "", { {pxO.x, pxO.y, "(O)"},  {pxA.x, pxA.y, "(A)"}, {pxContour.x, pxContour.y, "(C)"}});
		}

		// Ratio and bin
		double rat = dstOA / dstOC;
		int bi = int( rat / binWidth );	// bin index
		radial_count_bins[bi] ++;
		radial_intensity_bins[bi] += pxA.inten;

		// Cache this pixel's intensity for calculating the CV
		band_pixels[bi].push_back(pxA);
	}
}

const std::vector<double>& RadialDistribution::get_FracAtD()
{
	for (int i = 0; i < num_bins; i++)
		values_FracAtD[i] = double(radial_count_bins[i]) / double(this->cached_num_pixels);
	return values_FracAtD;
}

const std::vector<double>& RadialDistribution::get_MeanFrac()
{
	for (int i = 0; i < num_bins; i++)
		values_MeanFrac[i] = radial_intensity_bins[i] / double(radial_count_bins[i]);
	return values_MeanFrac;
}

const std::vector<double>& RadialDistribution::get_RadialCV()
{
	for (int i=0; i<band_pixels.size(); i++)
	{
		auto& band = band_pixels[i];

		std::vector<double> wedges;
		wedges.resize(RadialDistribution::num_bins, 0.0);

		for (auto& px : band)
		{
			int dx = px.x - cached_center_x,
				dy = px.y - cached_center_y;
			double ang = std::atan2(dy, dx);
			if (ang < 0)
				ang = 2.0 * M_PI + ang;
			double angW = 2.0 * M_PI / double(num_bins);
			int bin = ang / angW;
			wedges[bin] += px.inten;
		}

		// Mu
		double sum = 0.0;
		for (auto& w : wedges)
			sum += w;
		double mean = sum / double(RadialDistribution::num_bins);

		// Sigma
		sum = 0;
		for (auto& w : wedges)
			sum += (w - mean)*(w - mean);
		double var = sum / double(RadialDistribution::num_bins);
		double stddev = std::sqrt(var);
		double cv = stddev / mean;

		// Coefficient of variation
		values_RadialCV[i] = cv;
	}

	return values_RadialCV;
}
