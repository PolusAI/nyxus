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
#include <list>
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

bool operator == (const Pixel2& p1, const Pixel2& p2)
{
	if (p1.x != p2.x || p1.y != p2.y || p1.inten != p1.inten)
		return false;
	return true;
}

void ContourFeature::buildRegularContour(LR& r)
{
	//==== Pad the mask image with 2 pixels
	int width = r.aabb.get_width(),
		height = r.aabb.get_height(), 
		minx = r.aabb.get_xmin(), 
		miny = r.aabb.get_ymin();
	int paddingColor = 0;
	std::vector<PixIntens> paddedImage((height + 2) * (width + 2), paddingColor);
	for (auto px : r.raw_pixels)
	{
		auto x = px.x - minx + 1, 
			y = px.y - miny + 1;
		paddedImage [x + y * (width + 2)] = 1;	// Building the contour around the whole ROI mask image
	}


	//
	//
	//debug
	//
	VERBOSLVL4(
		std::cout << "\n\n\n" << "-- ContourFeature / buildRegularContour / Padded image --\n";
		for (int y = 0; y < height+2; y++)
		{
			for (int x = 0; x < width+2; x++)
			{
				size_t idx = x + y * (width+2);
				auto inte = paddedImage[idx];
				if (inte)
					std::cout << '*'; 
				else
					std::cout << '.'; 
			}
			std::cout << "\n";
		}
		std::cout << "\n\n\n";
	);	
	//
	//
	//


	const int BLANK = 0;
	bool inside = false;
	int pos = 0;

	//==== Prepare the contour ("border") image
	std::vector<PixIntens> borderImage((height + 2) * (width + 2), 0);

	// Initialize the entire image to blank
	for (int y = 0; y < (height + 2); y++)
		for (int x = 0; x < (width + 2); x++)
			borderImage[x + y * (width + 2)] = BLANK;

	//==== Scan the padded image and fill the border one
	for (int y = 0; y < (height + 2); y++)
		for (int x = 0; x < (width + 2); x++)
		{
			pos = x + y * (width + 2);

			// Scan for a non-blank pixel
			if (borderImage[pos] != 0 && !inside)		// Entering an already discovered border
			{
				inside = true;
			}
			else if (paddedImage[pos] != 0 && inside)	// Already discovered border point
			{
				continue;
			}
			else if (paddedImage[pos] == BLANK && inside)	// Leaving a border
			{
				inside = false;
			}
			else if (paddedImage[pos] != 0 && !inside)	// Undiscovered border point
			{
				borderImage[pos] = paddedImage[pos];	// Non-blank

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

					if (paddedImage[checkPosition] != 0) // Next border point found?
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
						borderImage[checkPosition] = paddedImage[checkPosition]; // Non-blank
					}
					else
					{
						// Rotate clockwise in the neighborhood
						checkLocationNr = 1 + (checkLocationNr % 8);
						if (counter2 > 8)
						{
							// If counter2 is above 8, we have sought around the neighborhood and
							// therefor the border is a single non-blank pixel, and we can exit
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

	//
	//
	//debug
	//
	VERBOSLVL4(
		std::cout << "\n\n\n" << "-- ContourFeature / buildRegularContour / Contour image --\n";
		// header
		std::cout << "\t";	// indent
		for (int i = 0; i < width; i++)
			if (i % 10 == 0)
				std::cout << '|';
			else
				std::cout << '_';
		std::cout << "\n";
		//---
		for (int y = 0; y < height + 2; y++)
		{
			std::cout << "y=" << y << "\t";
			for (int x = 0; x < width + 2; x++)
			{
				size_t idx = x + y * (width + 2);
				auto inte = borderImage[idx];
				if (inte)
					std::cout << ' '; 
				else
					std::cout << '+'; 
			}
			std::cout << "\n";
		}
		std::cout << "\n\n\n";
	);
	//
	//
	//

	//==== Remove padding and save the countour image as a vector of non-blank pixels
	AABB & bb = r.aabb; // r.aux_image_matrix.original_aabb;
	int base_x = bb.get_xmin(),
		base_y = bb.get_ymin();
	r.contour.clear();

	for (int y = 0; y < height+2; y++)
		for (int x = 0; x < width+2; x++)
		{
			size_t idx = x + y * (width + 2);
			auto inte = borderImage[idx];
			if (inte)
			{
				Pixel2 p(x-1, y-1, inte);		
				r.contour.push_back(p);
			}
		}


	//==== Reorder
	 
	//	--containers for unordered (temp) and ordered (result) pixels
	std::list<Pixel2> unordered(r.contour.begin(), r.contour.end());
	std::vector<Pixel2> ordered;
	ordered.reserve(unordered.size());

	//	--initialize vector 'ordered' with 1st pixel of 'unordered'
	auto itBeg = unordered.begin();
	Pixel2& pxTip = *itBeg;
	ordered.push_back(pxTip);
	unordered.remove(pxTip);

	//	--tip of the ordered contour
	pxTip = ordered[0];

	//	--harvest items of 'unordered' 
	while (unordered.size())
	{
		//	--find the neighbor of the current tip pixel 
		std::vector<Pixel2> cands;	// candidates
		for (Pixel2& px : unordered)
		{
			//	--test for proximity and skip non-neighbors
			auto dx = std::fabs((int)px.x - (int)pxTip.x),
				dy = std::fabs((int)px.y - (int)pxTip.y);
			if (dx > 1 || dy > 1)
				continue;	// not a neighbor of pxTip

			//	--we found the neighbor; grab it; make it the new tip pixel; quit this search loop 
			cands.push_back(px);
		}

		//	--are there any tip's neighbr candidate?
		if (!cands.empty())
		{
			int distMin = pxTip.sqdist(cands[0]);
			int idxMin = 0;
			for (int i = 1; i < cands.size(); i++)
			{
				Pixel2& px = cands[i];
				int dist = pxTip.sqdist(cands[i]);
				if (dist < distMin)
				{
					idxMin = i;
					distMin = dist;
				}
			}
			Pixel2& px = cands[idxMin];
			// find the closest candidate to pxTip
			ordered.push_back(px);
			unordered.remove(px);
			pxTip = ordered[ordered.size() - 1];
		}
		else //	--any gaps left by the contour algorithm?
		{
			// Most likely unavailability of an immediate neighboring pixel is due to 
			// its sitting in the 'ordered' set already meaning that the contour is closed. 
			// Sometimes a contour is closed despite 'unordered' set is nonempty - such a 
			// redundancy is due to the Moore based algorithm above.
			VERBOSLVL4(
				std::cerr << "gap in contour!\n" << "tip pixel: " << pxTip.x << "," << pxTip.y << "\n";
				std::cerr << "ordered:\n";
				int i = 1;
				for (auto& pxo : ordered)
				{
					std::cerr << "\t" << pxo.x << "," << pxo.y;
					if (i++ % 10 == 0)
						std::cerr << "\n";
				}
				std::cerr << "\n";

				int neigR2 = 400;	// squared
				std::cerr << "unordered around the tip (R^2=" << neigR2 << "):\n";
				i = 1;
				for (auto& pxu : unordered)
				{
					// filter out the far neighborhood
					if (pxTip.sqdist(pxu) > neigR2)
						continue;

					std::cerr << "\t" << pxu.x << "," << pxu.y;
					if (i++ % 10 == 0)
						std::cerr << "\n";
				}
				std::cerr << "\n";
			);
			break;
		}
	}

	// replace the unordered contour with ordered one
	r.contour = ordered;
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

		//==== Calculate ROI's image matrix
		r.aux_image_matrix.calculate_from_pixelcloud (r.raw_pixels, r.aabb);

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


