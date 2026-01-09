#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <sstream>
#include "radial_distribution.h"
#include "image_matrix.h"
#include "../globals.h"

using namespace Nyxus;

RadialDistributionFeature::RadialDistributionFeature() : FeatureMethod("RadialDistributionFeature")
{
	provide_features ({ Feature2D::FRAC_AT_D, Feature2D::MEAN_FRAC, Feature2D::RADIAL_CV });
	add_dependencies ({ Feature2D::PERIMETER });	// Actually we need the LR::contour object so we declare a dependency on feature 'PERIMETER' that in turn requires the LR::contour prerequisite
}

void RadialDistributionFeature::reset_buffers()
{
	// Clear
	values_FracAtD.clear();
	values_MeanFrac.clear();
	values_RadialCV.clear();
	radial_count_bins.clear();
	radial_intensity_bins.clear();

	for (auto& bw : banded_wedges)
		bw.clear();
	banded_wedges.clear();

	// Reallocate
	auto n = RadialDistributionFeature::num_bins;
	radial_count_bins.resize(n, 0);
	radial_intensity_bins.resize(n, 0.0);

	banded_wedges.resize(n);
	for (int i = 0; i < n; i++)
		banded_wedges[i].resize(n, 0);

	values_FracAtD.resize(n, 0);
	values_MeanFrac.resize(n, 0);
	values_RadialCV.resize(n, 0);
}

void RadialDistributionFeature::calculate (LR& r, const Fsettings& s)
{
	reset_buffers();

	auto n = RadialDistributionFeature::num_bins;

	auto& raw_pixels = r.raw_pixels;

	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	// Skip calculation if we have insofficient informative data 
	if (raw_pixels.size() == 0 || K.size() == 0)
		return;

	// Cache the pixels count
	this->cached_num_pixels = raw_pixels.size();

	// Find the center (most distant pixel from the edge)
	int idxO = Pixel2::find_center(raw_pixels, K);

	// Cache it
	this->cached_center_x = raw_pixels[idxO].x;
	this->cached_center_y = raw_pixels[idxO].y;

	// Get ahold of the center pixel
	const Pixel2& pxO = raw_pixels[idxO];

	// Max radius
	double dstOC = std::sqrt (pxO.max_sqdist (K)); //std::sqrt(pxContour.sqdist(pxO));

	for (auto& pxA : raw_pixels)
	{
		// Distance center to cloud pixel
		double dstOA = std::sqrt(pxA.sqdist(pxO));		
		
		// Find the radial bin index
		double rat = dstOA / dstOC;
		int bi = int(rat * (n-1));	// bin index
		if (bi >= n)
			bi = n - 1;

		// Update the bin counters
		radial_count_bins[bi] ++;
		radial_intensity_bins[bi] += pxA.inten;

		// Cache this pixel's intensity for calculating the CV
		int dx = pxA.x - cached_center_x,
			dy = pxA.y - cached_center_y;
		double ang = std::atan2(dy, dx);
		if (ang < 0)
			ang = 2.0 * M_PI + ang;
		double angW = 2.0 * M_PI / double(num_bins);
		int w_bin = int(ang / angW);	// wedge bin

		banded_wedges[bi][w_bin] += pxA.inten;
	}

	// Calculate the features (result - bin vectors values_FracAtD, values_MeanFrac, and values_RadialCV)
	get_FracAtD();
	get_MeanFrac();
	get_RadialCV();
}

void RadialDistributionFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

size_t RadialDistributionFeature::find_center_NT (const OutOfRamPixelCloud& cloud, const std::vector<Pixel2>& contour)
{
	int idxMinDif = 0;
	auto minmaxDist = cloud[idxMinDif].min_max_sqdist(contour);
	double minDif = minmaxDist.second - minmaxDist.first;
	for (size_t n = cloud.size(), i = 1; i < n; i++)
	{
		auto minmaxDist = cloud[i].min_max_sqdist(contour);
		double dif = minmaxDist.second - minmaxDist.first;
		if (dif < minDif)
		{
			minDif = dif;
			idxMinDif = i;
		}
	}
	return idxMinDif;
}

void RadialDistributionFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imlo)
{
	reset_buffers();

	auto n = RadialDistributionFeature::num_bins;

	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	// Skip calculation if we have insofficient informative data 
	if (r.raw_pixels_NT.size() == 0 || K.size() == 0)
		return;

	// Cache the pixels count
	this->cached_num_pixels = r.raw_pixels_NT.size();

	// Find the center (most distant pixel from the edge)
	int idxO = find_center_NT (r.raw_pixels_NT, K);

	// Cache it
	this->cached_center_x = r.raw_pixels_NT[idxO].x;
	this->cached_center_y = r.raw_pixels_NT[idxO].y;

	// Get ahold of the center pixel
	const Pixel2 pxO = r.raw_pixels_NT[idxO];

	// Max radius
	double dstOC = std::sqrt(pxO.max_sqdist(K)); 

	for (auto pxA : r.raw_pixels_NT)
	{
		// Distance center to cloud pixel
		double dstOA = std::sqrt(pxA.sqdist(pxO));

		// Find the radial bin index 
		double rat = dstOA / dstOC;
		int bi = int(rat * (n - 1));	// bin index
		if (bi >= n)
			bi = n - 1;

		// Update the bin counters
		radial_count_bins[bi] ++;
		radial_intensity_bins[bi] += pxA.inten;

		int dx = pxA.x - cached_center_x,
			dy = pxA.y - cached_center_y;
		double ang = std::atan2(dy, dx);
		if (ang < 0)
			ang = 2.0 * M_PI + ang;
		double angW = 2.0 * M_PI / double(num_bins);
		int w_bin = int(ang / angW);	// wedge bin

		banded_wedges[bi][w_bin] += pxA.inten;
	}

	// Calculate the features (result - bin vectors values_FracAtD, values_MeanFrac, and values_RadialCV)
	get_FracAtD();
	get_MeanFrac();
	get_RadialCV();
}

void RadialDistributionFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::FRAC_AT_D] = values_FracAtD;
	fvals[(int)Feature2D::MEAN_FRAC] = values_MeanFrac;
	fvals[(int)Feature2D::RADIAL_CV] = values_RadialCV;
}

void RadialDistributionFeature::extract (LR& r, const Fsettings& s)
{
	RadialDistributionFeature rdf;
	rdf.calculate (r, s);
	rdf.save_value (r.fvals);
}

void RadialDistributionFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		extract (r, s);
	}
}

void RadialDistributionFeature::get_FracAtD()
{
	for (int i = 0; i < num_bins; i++)
		values_FracAtD[i] = double(radial_count_bins[i]) / (double(cached_num_pixels) + epsilon);
}

void RadialDistributionFeature::get_MeanFrac()
{
	for (int i = 0; i < num_bins; i++)
		values_MeanFrac[i] = radial_intensity_bins[i] / (double(radial_count_bins[i]) + epsilon);
}

void RadialDistributionFeature:: get_RadialCV()
{
	for (int i = 0; i < banded_wedges.size(); i++)
	{
		auto& wedges = banded_wedges[i];

		// Mu
		double sum = 0.0;
		for (auto& w : wedges)
			sum += w;
		double mean = sum / double(RadialDistributionFeature::num_bins);

		// Sigma
		sum = 0;
		for (auto& w : wedges)
			sum += (w - mean) * (w - mean);
		double var = sum / double(RadialDistributionFeature::num_bins);
		double stddev = std::sqrt(var);
		double cv = stddev / (mean + epsilon);

		// Coefficient of variation
		values_RadialCV[i] = cv;
	}
}

