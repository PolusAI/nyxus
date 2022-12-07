#include <sstream>

#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>

#include "radial_distribution.h"
#include "image_matrix.h"
#include "../globals.h"

RadialDistributionFeature::RadialDistributionFeature() : FeatureMethod("RadialDistributionFeature")
{
	provide_features ({ FRAC_AT_D, MEAN_FRAC, RADIAL_CV });
	add_dependencies ({PERIMETER});
}

void RadialDistributionFeature::calculate(LR& r)
{
	radial_count_bins.resize (RadialDistributionFeature::num_bins, 0);
	radial_intensity_bins.resize (RadialDistributionFeature::num_bins, 0.0);
	angular_bins.resize (RadialDistributionFeature::num_bins, 0);
	band_pixels.resize (RadialDistributionFeature::num_bins);

	values_FracAtD.resize (RadialDistributionFeature::num_bins, 0);
	values_MeanFrac.resize (RadialDistributionFeature::num_bins, 0);
	values_RadialCV.resize (RadialDistributionFeature::num_bins, 0);

	auto& raw_pixels = r.raw_pixels;
	auto& contour_pixels = r.contour;

	// Skip calculation if we have insofficient informative data 
	if (raw_pixels.size() == 0 || contour_pixels.size() == 0)
		return;

	// Cache the pixels count
	this->cached_num_pixels = raw_pixels.size();

	// Find the center (most distant pixel from the edge)
	int idxO = Pixel2::find_center(raw_pixels, contour_pixels);

	// Cache it
	this->cached_center_x = raw_pixels[idxO].x;
	this->cached_center_y = raw_pixels[idxO].y;

	// Get ahold of the center pixel
	const Pixel2& pxO = raw_pixels[idxO];

	// Max radius
	double dstOC = std::sqrt (pxO.max_sqdist (contour_pixels)); //std::sqrt(pxContour.sqdist(pxO));

	for (auto& pxA : raw_pixels)
	{
		// Distance center to cloud pixel
		double dstOA = std::sqrt(pxA.sqdist(pxO));		
		
		// Find the radial bin index and update the bin counters
		double rat = dstOA / dstOC;
		int bi = int(rat * (num_bins-1));	// bin index
		if (bi >= num_bins)
			bi = num_bins - 1;
		radial_count_bins[bi] ++;
		radial_intensity_bins[bi] += pxA.inten;

		// Cache this pixel's intensity for calculating the CV
		band_pixels[bi].push_back(pxA);
	}

	// Calculate the features (result - corresponding bin vectors)
	get_FracAtD();
	get_MeanFrac();
	get_RadialCV();
}

void RadialDistributionFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

size_t RadialDistributionFeature::find_osized_cloud_center (OutOfRamPixelCloud& cloud, std::vector<Pixel2> & contour)
{
	int idxMindiff = 0;	// initial pixel index 0
	
	auto minmaxDist = cloud.get_at(idxMindiff).min_max_sqdist (contour);	//--triv--> auto minmaxDist = cloud[idxMindiff].min_max_sqdist(contour);
	double minDif = minmaxDist.second - minmaxDist.first;

	for (size_t i = 1; i < cloud.get_size(); i++)
	{
		// Caclculate the difference of distances
		minmaxDist = cloud.get_at(i).min_max_sqdist (contour);	//--triv--> auto minmaxDist = cloud[i].min_max_sqdist(contour);

		// Update the minimum difference
		double dif = minmaxDist.second - minmaxDist.first;
		if (dif < minDif)
		{
			minDif = dif;
			idxMindiff = i;
		}
	}

	return idxMindiff;
}

void RadialDistributionFeature::osized_calculate (LR& r, ImageLoader& imlo)
{
	radial_count_bins.resize(RadialDistributionFeature::num_bins, 0);
	radial_intensity_bins.resize(RadialDistributionFeature::num_bins, 0.0);
	angular_bins.resize(RadialDistributionFeature::num_bins, 0);
	band_pixels.resize(RadialDistributionFeature::num_bins);

	values_FracAtD.resize(RadialDistributionFeature::num_bins, 0);
	values_MeanFrac.resize(RadialDistributionFeature::num_bins, 0);
	values_RadialCV.resize(RadialDistributionFeature::num_bins, 0);

	// Skip calculation if we have insofficient informative data 
	if (r.aux_area == 0 || r.contour.size() == 0)
		return;

	auto& contour = r.contour;
	OutOfRamPixelCloud& cloud = r.raw_pixels_NT;

	// Cache the pixels count
	this->cached_num_pixels = r.aux_area; 

	// Find the center (most distant pixel from the edge)
	size_t idxO = find_osized_cloud_center (cloud, contour);	//--triv--> int idxO = Pixel2::find_center(raw_pixels, contour);

	// Cache the center
	Pixel2 pxO = cloud.get_at(idxO);
	this->cached_center_x = pxO.x;
	this->cached_center_y = pxO.y;

	// Distribute pixels into radial bins
	double binWidth = 1.0 / double(num_bins - 1);
	for (size_t i = 0; i < cloud.get_size(); i++)	//--triv--> for (auto& pxA : raw_pixels)
	{
		auto pxA = cloud.get_at(i);

		// If 'px' is a contour point, skip it
		if (pxA.belongs_to(contour))
			continue;

		// Find the contour point
		int idxCont = -1; // Pixel2& pxContour = conv_hull.CH[0];
		double distToRadius;

		for (int i = 0; i < contour.size(); i++)
		{
			const Pixel2& pxC = contour[i];
			double dAC = pxA.sqdist(pxC);
			double dOC = pxO.sqdist(pxC);
			double dOA = pxO.sqdist(pxA);
			if (dOC < dAC || dOC < dOA)
				continue;	// Perpendicular from A onto OC is situated beyond OC - skip this degenerate case

			double dA_OC = pxA.sqdist_to_segment(pxC, pxO);
			if (idxCont < 0 || dA_OC < distToRadius)
			{
				idxCont = i;
				distToRadius = dA_OC;
			}
		}

		// Was the contour point found? I may sometimes not be found due to some degeneracy of the contour itself, for instance, the ROI or its island is so small that it consists of the contour
		if (idxCont < 0)
			continue;

		const Pixel2& pxContour = contour[idxCont];

		// Distance center to cloud pixel
		double dstOA = std::sqrt(pxA.sqdist(pxO));

		// Distance center to contour
		double dstOC = std::sqrt(pxContour.sqdist(pxO));

		// Distance contour to pixel
		double dstAC = std::sqrt(pxContour.sqdist(pxA));

		// Intercept an error or weird condition
		if (dstOC < dstAC || dstOC < dstOA)
		{
			// Show A
			std::stringstream ss;
			if (dstOC < dstAC)
				ss << Nyxus::theIntFname << " Weird: OC=" << dstOC << " < AC=" << dstAC << ". Points O(" << pxO.x << "," << pxO.y << "), A(" << pxA.x << "," << pxA.y << "), and C(" << pxContour.x << "," << pxContour.y << ")";
			if (dstOC < dstOA)
				ss << Nyxus::theIntFname << " Weird: OC=" << dstOC << " < OA=" << dstOA << ". Points O(" << pxO.x << "," << pxO.y << "), A(" << pxA.x << "," << pxA.y << "), and C(" << pxContour.x << "," << pxContour.y << ")";
			ImageMatrix imCont(contour);
			imCont.print(ss.str(), "", { {pxO.x, pxO.y, "(O)"},  {pxA.x, pxA.y, "(A)"}, {pxContour.x, pxContour.y, "(C)"} });
		}

		// Ratio and bin
		double rat = dstOA / dstOC;
		int bi = int(rat / binWidth);	// bin index
		radial_count_bins[bi] ++;
		radial_intensity_bins[bi] += pxA.inten;

		// Cache this pixel's intensity for calculating the CV
		band_pixels[bi].push_back(pxA);
	}

	// Calculate the features (result - corresponding bin vectors)
	get_FracAtD();
	get_MeanFrac();
	get_RadialCV();
}

void RadialDistributionFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[FRAC_AT_D] = values_FracAtD; 
	fvals[MEAN_FRAC] = values_MeanFrac;  
	fvals[RADIAL_CV] = values_RadialCV;  
}

void RadialDistributionFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Calculate the radial distribution
		RadialDistributionFeature rdf;
		rdf.calculate(r);
		rdf.save_value(r.fvals);
	}
}

void RadialDistributionFeature::get_FracAtD()
{
	for (int i = 0; i < num_bins; i++)
		values_FracAtD[i] = double(radial_count_bins[i]) / double(this->cached_num_pixels);
}

void RadialDistributionFeature::get_MeanFrac()
{
	for (int i = 0; i < num_bins; i++)
		values_MeanFrac[i] = radial_intensity_bins[i] / double(radial_count_bins[i]);
}

void RadialDistributionFeature::get_RadialCV()
{
	for (int i=0; i<band_pixels.size(); i++)
	{
		auto& band = band_pixels[i];

		std::vector<double> wedges;
		wedges.resize(RadialDistributionFeature::num_bins, 0.0);

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
		double mean = sum / double(RadialDistributionFeature::num_bins);

		// Sigma
		sum = 0;
		for (auto& w : wedges)
			sum += (w - mean)*(w - mean);
		double var = sum / double(RadialDistributionFeature::num_bins);
		double stddev = std::sqrt(var);
		double cv = stddev / mean;

		// Coefficient of variation
		values_RadialCV[i] = cv;
	}
}

