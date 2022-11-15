#include "pixel.h"
#include <cstdint>
#include <vector>
#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "moments.h"
#include "contour.h"

#include "../roi_cache.h"	// Required by the reduction function
#include "../parallel.h"

#include "../environment.h"		// regular or whole slide mode

ContourFeature::ContourFeature() : FeatureMethod("ContourFeature")
{
	provide_features({ 
		PERIMETER,
		EQUIVALENT_DIAMETER,
		EDGE_INTEGRATEDINTENSITY,
		EDGE_MAX_INTENSITY,
		EDGE_MIN_INTENSITY,
		EDGE_MEAN_INTENSITY,
		EDGE_STDDEV_INTENSITY 
		});
}

void ContourFeature::buildRegularContourFast(LR& r)
{

	r.contour.clear();

	// Order the RLE
	std::vector<Pixel2> rle = r.rle_pixels;
	std::sort (rle.begin(), rle.end(), compare_locations);

	// Get starting points of each row
	// Also merge overlapping RLE
	std::vector<uint32_t> rows;
	rows.push_back(0);
	for (uint32_t i = 2; i < rle.size(); i+=2) {

		// Finds row boundary
		if (rle[i].y != rle[i-1].y) {
			rows.push_back(i);
			continue;
		}

		// Find overlapping RLE and merge them
		if (rle[i].x-1 == rle[i-1].x) {
			rle[i-1] = rle[i+1];
			rle[i].x = rle[i+1].x+1;
		}
	}
	rows.push_back(rle.size());

	// The first row must be entirely on the edge, so store it in the contour
	uint32_t pix_ind = 0;
	for (uint32_t i = 0; i < rows[1]; i+=2) {
		uint32_t n = rle[i+1].x - rle[i].x + 1;
		for (uint32_t ind = pix_ind; ind < pix_ind + n; ++ind) {
			Pixel2 p(r.raw_pixels[ind].x,r.raw_pixels[ind].y,r.raw_pixels[ind].inten);
			r.contour.push_back(p);
		}
		pix_ind += n;
	}

	// Loop over rows from the 2nd row until the 2nd to last row
	for (uint32_t p = 1; p < rows.size()-2; ++p) {
		uint32_t prev = rows[p-1];
		uint32_t prev_end = rows[p];
		uint32_t next = rows[p+1];
		uint32_t next_end = rows[p+2];

		// Find the intersection of previous and next rows
		std::vector<uint32_t> overlap;
		while (prev < prev_end && next < next_end) {
			if (rle[prev].x > rle[next+1].x) {
				next += 2;
				continue;
			}
			if (rle[next].x > rle[prev+1].x) {
				prev += 2;
				continue;
			}

			// If we get to this point, we have found overlap
			overlap.push_back(std::max(rle[prev].x,rle[next].x));
			overlap.push_back(std::min(rle[prev+1].x,rle[next+1].x));

			// Determine whether to advance next or previous
			if (rle[prev+1].x > rle[next+1].x) {
				next += 2;
			} else {
				prev += 2;
			}
		}

		// If there is no overlap, all pixels are edge pixels
		if (overlap.empty()) {
			for (uint32_t current = rows[p]; current < rows[p+1]; current+=2) {
				uint32_t n = rle[current+1].x - rle[current].x + 1;
				for (uint32_t ind = pix_ind; ind < pix_ind + n; ++ind) {
					Pixel2 p(r.raw_pixels[ind].x,r.raw_pixels[ind].y,r.raw_pixels[ind].inten);
					r.contour.push_back(p);
				}
				pix_ind += n;
			}
		} else {
			// Add pixels to non-overlapping regions
			uint32_t oind = 0;
			uint32_t offset = 0;
			std::vector<uint32_t> nonoverlap;
			for (uint32_t current = rows[p]; current < rows[p+1]; current+=2) {

				// skip over invalid rle
				if (rle[current+1].x < rle[current].x) {
					continue;
				}
				
				// The first pixel is always on the edge
				nonoverlap.push_back(offset);

				uint32_t x_start = rle[current].x;

				while (oind < overlap.size()) {

					if (overlap[oind+1] < rle[current].x) {
						oind += 2;
						continue;
					}
					
					if (overlap[oind] < rle[current+1].x) {
						long end = std::max(rle[current].x + 1, (long) overlap[oind]);
						long start = std::min(rle[current + 1].x, (long) overlap[oind + 1]+1);
						nonoverlap.push_back(end - x_start + offset);
						nonoverlap.push_back(start - x_start + offset);
					}

					if (overlap[oind+1] < rle[current+1].x) {
						oind += 2;
					} else {
						break;
					}
				}

				// The last pixel is always on the edge
				// In the case of a single pixel gap, it might already be there
				uint32_t last = rle[current+1].x - x_start + 1 + offset;
				if (nonoverlap[nonoverlap.size()-2] == last) {
					nonoverlap.pop_back();
				} else {
					nonoverlap.push_back(last);
				}

				offset += rle[current+1].x - rle[current].x + 1;
			}

			// Add pixels to the contour
			for (uint32_t ind = 0; ind < nonoverlap.size(); ind += 2) {
				for (uint32_t pind = nonoverlap[ind]; pind < nonoverlap[ind+1]; ++pind) {
					Pixel2 p(r.raw_pixels[pix_ind+pind].x, r.raw_pixels[pix_ind+pind].y, r.raw_pixels[pix_ind+pind].inten);
					r.contour.push_back(p);
				}
			}
			pix_ind += offset;
		}
	}

	// The last row must be entirely on the edge, so store it in the contour
	pix_ind = r.raw_pixels.size();
	for (uint32_t i = rows[rows.size()-2]; i < rows[rows.size()-1]; i+=2) {
		uint32_t n = rle[i+1].x - rle[i].x + 1;
		pix_ind -= n;
		for (uint32_t ind = pix_ind; ind < pix_ind + n; ++ind) {
			Pixel2 p(r.raw_pixels[ind].x,r.raw_pixels[ind].y,r.raw_pixels[ind].inten);
			r.contour.push_back(p);
		}
	}
}

void ContourFeature::buildRegularContour(LR& r)
{

	// If RLE is present, use it
	if (!r.rle_pixels.empty()) {
		buildRegularContourFast(r);
		return;
	}

	//==== Pad the image

	int width = r.aux_image_matrix.width,
		height = r.aux_image_matrix.height;

	readOnlyPixels image = r.aux_image_matrix.ReadablePixels();

	int paddingColor = 0;
	std::vector<PixIntens> paddedImage((height + 2) * (width + 2), paddingColor);
	for (int x = 0; x < width + 2; x++)
		for (int y = 0; y < height + 2; y++)
		{
			if (x == 0 || y == 0 || x == width + 1 || y == height + 1)
			{
				paddedImage[x + y * (width + 2)] = paddingColor;
			}
			else
			{
				paddedImage[x + y * (width + 2)] = image[x - 1 + (y - 1) * width];
			}
		}

	const int WHITE = 0;
	r.contour.clear();

	bool inside = false;
	int pos = 0;

	//==== Prepare the contour image
	std::vector<PixIntens> borderImage((height + 2) * (width + 2), 0);

	// Set entire image to WHITE
	for (int y = 0; y < (height + 2); y++)
		for (int x = 0; x < (width + 2); x++)
		{
			borderImage[x + y * (width + 2)] = WHITE;
		}

	//==== Scan 
	for (int y = 0; y < (height + 2); y++)
		for (int x = 0; x < (width + 2); x++)
		{
			pos = x + y * (width + 2);

			// Scan for BLACK pixel
			if (borderImage[pos] != 0 /*borderImage[pos] == BLACK*/ && !inside)		// Entering an already discovered border
			{
				inside = true;
			}
			else if (paddedImage[pos] != 0 /*paddedImage[pos] == BLACK*/ && inside)	// Already discovered border point
			{
				continue;
			}
			else if (paddedImage[pos] == WHITE && inside)	// Leaving a border
			{
				inside = false;
			}
			else if (paddedImage[pos] != 0 /*paddedImage[pos] == BLACK*/ && !inside)	// Undiscovered border point
			{
				borderImage[pos] = paddedImage[pos]; /*BLACK*/

				int checkLocationNr = 1;	// The neighbor number of the location we want to check for a new border point
				int checkPosition;			// The corresponding absolute array address of checkLocationNr
				int newCheckLocationNr; 	// Variable that holds the neighborhood position we want to check if we find a new border at checkLocationNr
				int startPos = pos;			// Set start position
				int counter = 0; 			// Counter is used for the jacobi stop criterion
				int counter2 = 0; 			// Counter2 is used to determine if the point we have discovered is one single point

				// Defines the neighborhood offset position from current position and the neighborhood
				// position we want to check next if we find a new border at checkLocationNr
				int neighborhood[8][2] = {
						{-1,7},
						{-3 - width,7},
						{-width - 2,1},
						{-1 - width,1},
						{1,3},
						{3 + width,3},
						{width + 2,5},
						{1 + width,5}
				};
				// Trace around the neighborhood
				while (true)
				{
					checkPosition = pos + neighborhood[checkLocationNr - 1][0];
					newCheckLocationNr = neighborhood[checkLocationNr - 1][1];

					if (paddedImage[checkPosition] != 0 /*paddedImage[checkPosition] == BLACK*/) // Next border point found
					{
						if (checkPosition == startPos)
						{
							counter++;

							// Stopping criterion (jacob)
							if (newCheckLocationNr == 1 || counter >= 3)
							{
								// Close loop
								inside = true; // Since we are starting the search at were we first started we must set inside to true
								break;
							}
						}

						checkLocationNr = newCheckLocationNr; // Update which neighborhood position we should check next
						pos = checkPosition;
						counter2 = 0; 						// Reset the counter that keeps track of how many neighbors we have visited
						borderImage[checkPosition] = paddedImage[checkPosition]; /*BLACK*/
					}
					else
					{
						// Rotate clockwise in the neighborhood
						checkLocationNr = 1 + (checkLocationNr % 8);
						if (counter2 > 8)
						{
							// If counter2 is above 8 we have traced around the neighborhood and
							// therefor the border is a single black pixel and we can exit
							counter2 = 0;
							break;
						}
						else
						{
							counter2++;
						}
					}
				}
			}
		}

	//==== Remove padding and save the countour image as a vector of contour-onlu pixels
	AABB bb = r.aux_image_matrix.original_aabb;
	int base_x = bb.get_xmin(),
		base_y = bb.get_ymin();
	r.contour.clear();
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			size_t idx = x + 1 + (y + 1) * (width + 2);
			// clippedBorderImage[x + y * width] = borderImage [idx];
			auto inte = borderImage[idx];
			//?if (inte)
			{
				Pixel2 p(x + base_x, y + base_y, inte);
				r.contour.push_back(p);
			}
		}
	}
}

bool ContourFeature::compare_locations (const Pixel2& lhs, const Pixel2& rhs)
{
	return (lhs.y < rhs.y) || (lhs.y == rhs.y && lhs.x < rhs.x);
}

void ContourFeature::buildWholeSlideContour(LR& r)
{
	// Push the 4 slide vertices of dummy intensity 999
	Pixel2 tl (r.aabb.get_xmin(), r.aabb.get_ymin(), 999),
		tr (r.aabb.get_xmax(), r.aabb.get_ymin(), 999), 
		bl (r.aabb.get_xmin(), r.aabb.get_ymax(), 999), 
		br (r.aabb.get_xmax(), r.aabb.get_ymax(), 999);
	r.contour.push_back(tl);
	r.contour.push_back(tr);
	r.contour.push_back(br);
	r.contour.push_back(bl);
}

void ContourFeature::calculate(LR& r)
{
	if (Nyxus::theEnvironment.singleROI)
		buildWholeSlideContour(r);
	else
		buildRegularContour(r);

	//=== Calculate the features
	fval_PERIMETER = (StatsInt)r.contour.size();
	fval_EQUIVALENT_DIAMETER = fval_PERIMETER / M_PI;
	auto [cmin, cmax, cmean, cstddev] = calc_min_max_mean_stddev_intensity (r.contour);
	fval_EDGE_MEAN_INTENSITY = cmean;
	fval_EDGE_STDDEV_INTENSITY = cstddev;
	fval_EDGE_MAX_INTENSITY  = cmax;
	fval_EDGE_MIN_INTENSITY = cmin;

	fval_EDGE_INTEGRATEDINTENSITY = 0;
	for (auto& px : r.contour)
		fval_EDGE_INTEGRATEDINTENSITY += px.inten;
}

void ContourFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void ContourFeature::osized_calculate(LR& r, ImageLoader& imloader)
{}

void ContourFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[PERIMETER][0] = fval_PERIMETER;
	fvals[EQUIVALENT_DIAMETER][0] = fval_EQUIVALENT_DIAMETER;
	fvals[EDGE_MEAN_INTENSITY][0] = fval_EDGE_MEAN_INTENSITY;
	fvals[EDGE_STDDEV_INTENSITY][0] = fval_EDGE_STDDEV_INTENSITY;
	fvals[EDGE_MAX_INTENSITY][0] = fval_EDGE_MAX_INTENSITY;
	fvals[EDGE_MIN_INTENSITY][0] = fval_EDGE_MIN_INTENSITY;
	fvals[EDGE_INTEGRATEDINTENSITY][0] = fval_EDGE_INTEGRATEDINTENSITY;
}

void ContourFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{	
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel (ContourFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void ContourFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{}

void ContourFeature::cleanup_instance()
{}

std::tuple<double, double, double, double> ContourFeature::calc_min_max_mean_stddev_intensity (const std::vector<Pixel2>& contour_pixels)
{
	Moments4 m;
	for (auto px : contour_pixels)
		m.add(px.inten);

	StatsReal min_ = m.min__(),
		max_ = m.max__(),
		mean_ = m.mean(),
		stddev_ = m.std();

	return { min_, max_, mean_, stddev_ };
}

namespace Nyxus
{
	void calcRoiContour(LR& r)
	{
		if (r.roi_disabled)
			return;

		//==== Calculate ROI's image matrix if no RLE is present
		if (r.rle_pixels.empty()) {
			r.aux_image_matrix.calculate_from_pixelcloud (r.raw_pixels, r.aabb);
		}

		//==== Contour, ROI perimeter, equivalent circle diameter
		ContourFeature f;
		f.calculate(r);	// Consumes LR::aux_image_matrix, leaves contour pixels in LR::contour
		f.save_value(r.fvals);
	}

	void parallelReduceContour (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
	{
		for (auto i = start; i < end; i++)
		{
			int lab = (*ptrLabels)[i];
			LR& r = (*ptrLabelData)[lab];
			if (r.has_bad_data())
				continue;
			calcRoiContour(r);
		}
	}
}


