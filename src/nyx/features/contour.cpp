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
#include <iomanip>
#include "moments.h"
#include "contour.h"
#include "../globals.h"

#include "../roi_cache.h"	// Required by the reduction function
#include "../parallel.h"

#include "../environment.h"		// regular or whole slide mode
#include "image_matrix_nontriv.h"

using namespace Nyxus;

bool ContourFeature::required(const FeatureSet& fs)
{

	return theFeatureSet.anyEnabled({
		// own features
		Feature2D::PERIMETER,
		Feature2D::DIAMETER_EQUAL_PERIMETER,
		Feature2D::EDGE_INTEGRATED_INTENSITY,
		Feature2D::EDGE_MAX_INTENSITY,
		Feature2D::EDGE_MIN_INTENSITY,
		Feature2D::EDGE_MEAN_INTENSITY,
		Feature2D::EDGE_STDDEV_INTENSITY,
		// dependencies:
		Feature2D::CONVEX_HULL_AREA, 
		Feature2D::SOLIDITY,
		Feature2D::CIRCULARITY,
		// weighted spatial moments
		Feature2D::WEIGHTED_SPAT_MOMENT_00,
		Feature2D::WEIGHTED_SPAT_MOMENT_01,
		Feature2D::WEIGHTED_SPAT_MOMENT_02,
		Feature2D::WEIGHTED_SPAT_MOMENT_03,
		Feature2D::WEIGHTED_SPAT_MOMENT_10,
		Feature2D::WEIGHTED_SPAT_MOMENT_11,
		Feature2D::WEIGHTED_SPAT_MOMENT_12,
		Feature2D::WEIGHTED_SPAT_MOMENT_20,
		Feature2D::WEIGHTED_SPAT_MOMENT_21,
		Feature2D::WEIGHTED_SPAT_MOMENT_30,
		// weighted central moments
		Feature2D::WEIGHTED_CENTRAL_MOMENT_02,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_03,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_11,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_12,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_20,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_21,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_30,
		// weighted normalized central moments
		Feature2D::WT_NORM_CTR_MOM_02,
		Feature2D::WT_NORM_CTR_MOM_03,
		Feature2D::WT_NORM_CTR_MOM_11,
		Feature2D::WT_NORM_CTR_MOM_12,
		Feature2D::WT_NORM_CTR_MOM_20,
		Feature2D::WT_NORM_CTR_MOM_21,
		Feature2D::WT_NORM_CTR_MOM_30,
		// weighted Hu's moments 1-7 
		Feature2D::WEIGHTED_HU_M1,
		Feature2D::WEIGHTED_HU_M2,
		Feature2D::WEIGHTED_HU_M3,
		Feature2D::WEIGHTED_HU_M4,
		Feature2D::WEIGHTED_HU_M5,
		Feature2D::WEIGHTED_HU_M6,
		Feature2D::WEIGHTED_HU_M7,

		// -- intensity raw moments
		Nyxus::Feature2D::IMOM_RM_00,
		Nyxus::Feature2D::IMOM_RM_01,
		Nyxus::Feature2D::IMOM_RM_02,
		Nyxus::Feature2D::IMOM_RM_03,
		Nyxus::Feature2D::IMOM_RM_10,
		Nyxus::Feature2D::IMOM_RM_11,
		Nyxus::Feature2D::IMOM_RM_12,
		Nyxus::Feature2D::IMOM_RM_13,
		Nyxus::Feature2D::IMOM_RM_20,
		Nyxus::Feature2D::IMOM_RM_21,
		Nyxus::Feature2D::IMOM_RM_22,
		Nyxus::Feature2D::IMOM_RM_23,
		Nyxus::Feature2D::IMOM_RM_30,
		// -- intensity central moments
		Nyxus::Feature2D::IMOM_CM_00,
		Nyxus::Feature2D::IMOM_CM_01,
		Nyxus::Feature2D::IMOM_CM_02,
		Nyxus::Feature2D::IMOM_CM_03,
		Nyxus::Feature2D::IMOM_CM_10,
		Nyxus::Feature2D::IMOM_CM_11,
		Nyxus::Feature2D::IMOM_CM_12,
		Nyxus::Feature2D::IMOM_CM_13,
		Nyxus::Feature2D::IMOM_CM_20,
		Nyxus::Feature2D::IMOM_CM_21,
		Nyxus::Feature2D::IMOM_CM_22,
		Nyxus::Feature2D::IMOM_CM_23,
		Nyxus::Feature2D::IMOM_CM_30,
		Nyxus::Feature2D::IMOM_CM_31,
		Nyxus::Feature2D::IMOM_CM_32,
		Nyxus::Feature2D::IMOM_CM_33,
		// -- intensity normalized raw moments
		Nyxus::Feature2D::IMOM_NRM_00,
		Nyxus::Feature2D::IMOM_NRM_01,
		Nyxus::Feature2D::IMOM_NRM_02,
		Nyxus::Feature2D::IMOM_NRM_03,
		Nyxus::Feature2D::IMOM_NRM_10,
		Nyxus::Feature2D::IMOM_NRM_11,
		Nyxus::Feature2D::IMOM_NRM_12,
		Nyxus::Feature2D::IMOM_NRM_13,
		Nyxus::Feature2D::IMOM_NRM_20,
		Nyxus::Feature2D::IMOM_NRM_21,
		Nyxus::Feature2D::IMOM_NRM_22,
		Nyxus::Feature2D::IMOM_NRM_23,
		Nyxus::Feature2D::IMOM_NRM_30,
		Nyxus::Feature2D::IMOM_NRM_31,
		Nyxus::Feature2D::IMOM_NRM_32,
		Nyxus::Feature2D::IMOM_NRM_33,
		// -- intensity normalized central moments
		Nyxus::Feature2D::IMOM_NCM_02,
		Nyxus::Feature2D::IMOM_NCM_03,
		Nyxus::Feature2D::IMOM_NCM_11,
		Nyxus::Feature2D::IMOM_NCM_12,
		Nyxus::Feature2D::IMOM_NCM_20,
		Nyxus::Feature2D::IMOM_NCM_21,
		Nyxus::Feature2D::IMOM_NCM_30,
		// -- intensity Hu's moments 1-7 
		Nyxus::Feature2D::IMOM_HU1,
		Nyxus::Feature2D::IMOM_HU2,
		Nyxus::Feature2D::IMOM_HU3,
		Nyxus::Feature2D::IMOM_HU4,
		Nyxus::Feature2D::IMOM_HU5,
		Nyxus::Feature2D::IMOM_HU6,
		Nyxus::Feature2D::IMOM_HU7,
		// -- intensity weighted raw moments
		Nyxus::Feature2D::IMOM_WRM_00,
		Nyxus::Feature2D::IMOM_WRM_01,
		Nyxus::Feature2D::IMOM_WRM_02,
		Nyxus::Feature2D::IMOM_WRM_03,
		Nyxus::Feature2D::IMOM_WRM_10,
		Nyxus::Feature2D::IMOM_WRM_11,
		Nyxus::Feature2D::IMOM_WRM_12,
		Nyxus::Feature2D::IMOM_WRM_20,
		Nyxus::Feature2D::IMOM_WRM_21,
		Nyxus::Feature2D::IMOM_WRM_30,
		// -- intensity weighted central moments
		Nyxus::Feature2D::IMOM_WCM_02,
		Nyxus::Feature2D::IMOM_WCM_03,
		Nyxus::Feature2D::IMOM_WCM_11,
		Nyxus::Feature2D::IMOM_WCM_12,
		Nyxus::Feature2D::IMOM_WCM_20,
		Nyxus::Feature2D::IMOM_WCM_21,
		Nyxus::Feature2D::IMOM_WCM_30,
		// -- intensity weighted normalized central moments
		Nyxus::Feature2D::IMOM_WNCM_02,
		Nyxus::Feature2D::IMOM_WNCM_03,
		Nyxus::Feature2D::IMOM_WNCM_11,
		Nyxus::Feature2D::IMOM_WNCM_12,
		Nyxus::Feature2D::IMOM_WNCM_20,
		Nyxus::Feature2D::IMOM_WNCM_21,
		Nyxus::Feature2D::IMOM_WNCM_30,
		// -- intensity weighted Hu's moments 1-7 
		Nyxus::Feature2D::IMOM_WHU1,
		Nyxus::Feature2D::IMOM_WHU2,
		Nyxus::Feature2D::IMOM_WHU3,
		Nyxus::Feature2D::IMOM_WHU4,
		Nyxus::Feature2D::IMOM_WHU5,
		Nyxus::Feature2D::IMOM_WHU6,
		Nyxus::Feature2D::IMOM_WHU7,

		// misc
		Feature2D::ROI_RADIUS_MEAN, 
		Feature2D::ROI_RADIUS_MAX, 
		Feature2D::ROI_RADIUS_MEDIAN,
		Feature2D::FRAC_AT_D, 
		Feature2D::MEAN_FRAC, 
		Feature2D::RADIAL_CV
		});
}

ContourFeature::ContourFeature() : FeatureMethod("ContourFeature")
{
	provide_features (ContourFeature::featureset);
}

std::vector<Pixel2> find_cands (const std::list<Pixel2> & unordered, const Pixel2 & tip)
{
	std::vector<Pixel2> cands;	// candidates
	for (const Pixel2& px : unordered)
	{
		//	--test for proximity and skip non-neighbors
		auto dx = std::fabs((int)px.x - (int)tip.x),
			dy = std::fabs((int)px.y - (int)tip.y);
		if (dx > 1 || dy > 1)
			continue;	// not a neighbor of pxTip

		//	--we found the neighbor; grab it; make it the new tip pixel; quit this search loop 
		cands.push_back (px);
	}
	return cands;
}

Pixel2 find_closest (const std::list<Pixel2>& unordered, const Pixel2& tip)
{
	Pixel2 retval = tip; // default, in case 'unordered' is empty and we don't find the closest element

	double mindist = std::numeric_limits<double>::max();
	for (const Pixel2& p : unordered)
	{
		double d = tip.sqdist(p);
		if (d < mindist)
		{
			mindist = d;
			retval = p;
		}
	}

	return retval;
}

std::vector<int> score_cands (const std::vector<Pixel2>& cands, const Pixel2& pxTip)
{
	std::vector<int> S;

	for (const Pixel2& c : cands)
	{
		double dist = pxTip.sqdist (c);	// either 1 or 2
		int score = (int)dist; // favor diagonal moves
		S.push_back (score);
	}
	return S;
}

void ContourFeature::buildRegularContour(LR& r)
{
	//==== Pad the image

	int width = r.aabb.get_width(),
		height = r.aabb.get_height(),
		base_x = r.aabb.get_xmin(),
		base_y = r.aabb.get_ymin();
	int paddingColor = 0;
	std::vector<PixIntens> paddedImage((height + 2) * (width + 2), paddingColor);
	for (auto px : r.raw_pixels)
	{
		auto x = px.x - base_x + 1,
			y = px.y - base_y + 1;
		paddedImage.at(x + y * (width + 2)) = px.inten + 1;	// we build a contour keeping corresponding intensities
	}

	VERBOSLVL4 (dump_2d_image_1d_layout(paddedImage, width + 2, height + 2, "\n\n\n ContourFeature / buildRegularContour / Padded image ROI " + std::to_string(r.aabb.get_width()) + " W " + std::to_string(r.aabb.get_height()) + " H \n",  "\n\n\n"));

	const int BLANK = 0;
	bool inside = false;
	int pos = 0;

	//==== Prepare the contour ("border") image
	std::vector<PixIntens> borderImage((height + 2) * (width + 2), 0);

	// Initialize the entire image to blank
	for (int y = 0; y < (height + 2); y++)
		for (int x = 0; x < (width + 2); x++)
			borderImage.at(x + y * (width + 2)) = BLANK;

	//==== Scan the padded image and fill the border one

	for (int y = 0; y < (height + 2); y++)
	{
		for (int x = 0; x < (width + 2); x++)
		{
			pos = x + y * (width + 2);

			// Scan for a non-blank pixel
			PixIntens bi, pi;
			bi = borderImage.at(pos);
			pi = paddedImage.at(pos);
			if (bi != 0 && !inside)		// Entering an already discovered border
			{
				inside = true;
			}
			else if (pi != 0 && inside)	// Already discovered border point
			{
				continue;
			}
			else if (pi == BLANK && inside)	// Leaving a border
			{
				inside = false;
			}
			else if (pi != 0 && !inside)	// Undiscovered border point
			{
				borderImage.at(pos) = paddedImage.at(pos); // Non-blank

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

					PixIntens pi2 = 0;
					if (checkPosition >= paddedImage.size())		// we're done if we start checking outside the image
						break;
					pi2 = paddedImage.at(checkPosition);
					if (pi2 != 0) // Next border point found?
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
						borderImage.at(checkPosition) = paddedImage.at(checkPosition); // Non-blank
					}
					else
					{
						// Rotate clockwise in the neighborhood
						checkLocationNr = 1 + (checkLocationNr % 8);
						if (counter2 > 8)
						{
							// If counter2 is above 8, we have sought around the neighborhood and
							// therefore the border is a single non-blank pixel, and we can exit
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
	}

	VERBOSLVL4(dump_2d_image_1d_layout (borderImage, width + 2, height + 2, "\n\n-- ContourFeature / buildRegularContour / Padded contour image --\n", "\n\n"));

	//==== remove padding 
	r.contour.clear();

	// gather contour pixels undecorating their intensities back to original values
	for (int y = 0; y < height + 2; y++)
		for (int x = 0; x < width + 2; x++)
		{
			size_t idx = x + y * (width + 2);
			auto inte = borderImage.at(idx);
			if (inte)
			{
				// register a pixel only if it has any immediate neighbor
				bool hasNeig = false;
				if (x > 0)	// left neighbor
				{
					size_t idxNeig = (x-1) + y * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x < width-1)	// right neighbor
				{
					size_t idxNeig = (x+1) + y * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (y > 0)	// upper neighbor
				{
					size_t idxNeig = x + (y-1) * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (y < height-1)	// lower neighbor
				{
					size_t idxNeig = x + (y+1) * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x>0 && y > 0)	// upper left neighbor
				{
					size_t idxNeig = (x-1) + (y-1) * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x < width-1 && y > 0)	// upper right neighbor
				{
					size_t idxNeig = (x+1) + (y-1) * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x>0 && y < height-1)	// lower left neighbor
				{
					size_t idxNeig = (x-1) + (y+1) * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x < width-1 && y < height-1)	// lower right neighbor
				{
					size_t idxNeig = (x+1) + (y+1) * (width+2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (!hasNeig)
					continue;
				// pixel is good, save it
				Pixel2 p(x, y, inte - 1);
				r.contour.push_back(p);
			}
		}

	VERBOSLVL4(dump_2d_image_with_vertex_chain(borderImage, r.contour, width + 2, height + 2, "\n\n-- ContourFeature / buildRegularContour / Padded contour image + UNsorted contour--\n", "\n\n"));

	//==== Reorder the contour cloud

	//	--containers for unordered (temp) and ordered (result) pixels
	std::list<Pixel2> unordered(r.contour.begin(), r.contour.end());
	std::vector<Pixel2> ordered;
	ordered.reserve(unordered.size());
	std::vector<Pixel2> pants;

	//	--initialize vector 'ordered' with 1st pixel of 'unordered'
	auto itBeg = unordered.begin();
	Pixel2 pxTip = *itBeg;
	ordered.push_back(pxTip);
	unordered.remove(pxTip);

	//	-- tip of the ordered contour
	pxTip = ordered.at(0);

	//	-- harvest items of 'unordered' 
	while (unordered.size())
	{
		//	--find tip's neighbors 
		std::vector<Pixel2> cands = find_cands (unordered, pxTip);
		if (cands.empty())
		{
			// -- we have a gap and need to fix it
			VERBOSLVL4(dump_2d_image_with_halfcontour(borderImage, unordered, ordered, pxTip, width + 2, height + 2, "\nhalfcontour:\n", ""));
				
			// -- no 'break;' ,instead, jump the tip to the closest U-pixel
			Pixel2 pxPants;
			pxPants = pants.back();
			pxTip = pxPants;
			Pixel2 closest = find_closest (unordered, pxTip);

			// -- discharge
			ordered.push_back (closest);
			unordered.remove (closest);
			pxTip = ordered.at(ordered.size() - 1);
		}
		else
		{
			// -- register pants
			if (cands.size() >= 2)
				pants.push_back(pxTip);

			// -- score thems
			std::vector<int> candScores = score_cands(cands, pxTip);

			// -- choose the best
			auto itBest = std::min_element (candScores.begin(), candScores.end());
			int idxBest = (int)std::distance (candScores.begin(), itBest);

			// -- discharge the found pixel from set 'unordered' and update the tip
			Pixel2& px = cands.at(idxBest);
			ordered.push_back(px);
			unordered.remove(px);
			pxTip = ordered.at(ordered.size() - 1);
		}
	}

	// done sorting. Now set the ordered contour in the ROI
	r.contour = ordered;

	VERBOSLVL4(dump_2d_image_with_vertex_chain(borderImage, r.contour, width + 2, height + 2, "\n\n-- ContourFeature / buildRegularContour / Padded contour image + sorted contour--\n", "\n\n"));

	// finally, fix pixel positions back to absolute
	for (Pixel2& p : r.contour)
	{
		p.x += base_x;
		p.y += base_y;
	}
}

void ContourFeature::buildRegularContour_nontriv(LR& r)
{
	//==== Pad the image

	int width = r.aabb.get_width(),
		height = r.aabb.get_height(),
		minx = r.aabb.get_xmin(),
		miny = r.aabb.get_ymin();
	int paddingColor = 0;

	WriteImageMatrix_nontriv paddedImage("paddedImage", r.label);
	paddedImage.allocate(width + 2, height + 2, paddingColor);

	for (auto px : r.raw_pixels_NT)
	{
		auto x = px.x - minx + 1,
			y = px.y - miny + 1;
		paddedImage.set_at(x + y * (width + 2), px.inten + 1);	// Decorate the intensity
	}

	const int WHITE = 0;
	r.contour.clear();

	bool inside = false;
	int pos = 0;

	//==== Prepare the contour image
	WriteImageMatrix_nontriv borderImage("borderImage", r.label);
	paddedImage.allocate(width + 2, height + 2, 0);

	// Set entire image to WHITE
	for (int y = 0; y < (height + 2); y++)
		for (int x = 0; x < (width + 2); x++)
		{
			borderImage.set_at(x + y * (width + 2), WHITE);
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
				borderImage.set_at(pos, paddedImage[pos]); /*BLACK*/

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
						borderImage.set_at(checkPosition, paddedImage[checkPosition]); /*BLACK*/
					}
					else
					{
						// Rotate clockwise in the neighborhood
						checkLocationNr = 1 + (checkLocationNr % 8);
						if (counter2 > 8)
						{
							// If counter2 is above 8 we have traced around the neighborhood and
							// therefore the border is a single black pixel and we can exit
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

	//==== Remove padding and save the contour image as a vector of contour-onlu pixels
	AABB& bb = r.aabb;
	int base_x = bb.get_xmin(),
		base_y = bb.get_ymin();
	r.contour.clear();

	for (int y = 0; y < height + 2; y++)
		for (int x = 0; x < width + 2; x++)
		{
			size_t idx = x + y * (width + 2);
			auto inte = borderImage[idx];
			if (inte)
			{
				Pixel2 p(x, y, inte - 1);	// Undecorate the intensity
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
	Pixel2 pxTip = *itBeg;
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
			//	--test for proximity and skio non-neighbors
			auto dx = std::fabs((int)px.x - (int)pxTip.x),
				dy = std::fabs((int)px.y - (int)pxTip.y);
			if (dx > 1 || dy > 1)
				continue;	// not a neighbor of pxTip

			//	--we found the neighbor; grab it; make it the new tip pixel; quit this search loop 
			cands.push_back(px);
		}

		if (!cands.empty())
		{
			int distMin = (int)pxTip.sqdist(cands[0]);
			int idxMin = 0;
			for (int i = 1; i < cands.size(); i++)
			{
				Pixel2& px = cands[i];
				int dist = (int)pxTip.sqdist(cands[i]);
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
			// Most likely cause is the closing pixel is in 'ordered' already and the pixels in 'unordered' are redundant due to the Moore based algorithm above
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
	// Create 4 slide corner vertices of slide's max intensity
	PixIntens maxI = r.aux_max;
	Pixel2 tl (r.aabb.get_xmin(), r.aabb.get_ymin(), maxI),
		tr (r.aabb.get_xmax(), r.aabb.get_ymin(), maxI),
		bl (r.aabb.get_xmin(), r.aabb.get_ymax(), maxI),
		br (r.aabb.get_xmax(), r.aabb.get_ymax(), maxI);
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
	fval_PERIMETER = 0;
	size_t clen = r.contour.size();
	for (size_t i = 0; i < clen; i++)
		if (i == 0)
		{
			Pixel2& p1 = r.contour[clen-1],
				& p2 = r.contour[i];
			fval_PERIMETER += std::sqrt(p1.sqdist(p2));
		}
		else
		{
			Pixel2& p1 = r.contour[i - 1],
				& p2 = r.contour[i];
			fval_PERIMETER += std::sqrt(p1.sqdist(p2));
		}

	fval_DIAMETER_EQUAL_PERIMETER = fval_PERIMETER / M_PI;
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
{
	buildRegularContour_nontriv(r);	// leaves the contour pixels in field 'contour'

	//=== Calculate the features
	fval_PERIMETER = (StatsInt)r.contour.size();
	fval_DIAMETER_EQUAL_PERIMETER = fval_PERIMETER / M_PI;
	auto [cmin, cmax, cmean, cstddev] = calc_min_max_mean_stddev_intensity(r.contour);
	fval_EDGE_MEAN_INTENSITY = cmean;
	fval_EDGE_STDDEV_INTENSITY = cstddev;
	fval_EDGE_MAX_INTENSITY = cmax;
	fval_EDGE_MIN_INTENSITY = cmin;

	fval_EDGE_INTEGRATEDINTENSITY = 0;
	for (auto& px : r.contour)
		fval_EDGE_INTEGRATEDINTENSITY += px.inten;
}

void ContourFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::PERIMETER][0] = fval_PERIMETER;
	fvals[(int)Feature2D::DIAMETER_EQUAL_PERIMETER][0] = fval_DIAMETER_EQUAL_PERIMETER;
	fvals[(int)Feature2D::EDGE_MEAN_INTENSITY][0] = fval_EDGE_MEAN_INTENSITY;
	fvals[(int)Feature2D::EDGE_STDDEV_INTENSITY][0] = fval_EDGE_STDDEV_INTENSITY;
	fvals[(int)Feature2D::EDGE_MAX_INTENSITY][0] = fval_EDGE_MAX_INTENSITY;
	fvals[(int)Feature2D::EDGE_MIN_INTENSITY][0] = fval_EDGE_MIN_INTENSITY;
	fvals[(int)Feature2D::EDGE_INTEGRATED_INTENSITY][0] = fval_EDGE_INTEGRATEDINTENSITY;
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

void ContourFeature::extract (LR& r)
{
	ContourFeature f;
	f.calculate (r);
	f.save_value (r.fvals);
}

void ContourFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		ContourFeature f;
		f.calculate (r);
		f.save_value (r.fvals);
	}
}
