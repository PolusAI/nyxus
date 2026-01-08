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
#include "../environment.h"		// regular or whole slide mode
#include "image_matrix_nontriv.h"

using namespace Nyxus;

bool ContourFeature::required (const FeatureSet & fset)
{
	return fset.anyEnabled(
		{
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

//
// returns neighboring pixels of 'tip'
//
/*static*/ std::vector<Pixel2> ContourFeature::find_cands (const std::list<Pixel2> &unordered, const Pixel2 &tip)
{
	// straight neighbors first
	std::vector<Pixel2> cands10;
	for (const Pixel2& px : unordered)
	{
		auto dx = std::fabs((int)px.x - (int)tip.x),
			dy = std::fabs((int)px.y - (int)tip.y);
		if (dx+dy>1)
			continue;
		cands10.push_back (px);
	}

	// straight neighbors, if any
	if (cands10.size())
		return cands10;

	// then diagonal neighbors
	std::vector<Pixel2> cands11;

	for (const Pixel2& px : unordered)
	{
		auto dx = std::fabs((int)px.x - (int)tip.x),
			dy = std::fabs((int)px.y - (int)tip.y);
		if (dx==1 && dy==1)
			cands11.push_back(px);
	}

	// diagonal neighbors
	return cands11;
}

/*static*/ int ContourFeature::get_dial_pos (const Pixel2 &d)
{
	/*

	  (-,-)  (0,-)  (+,-)
		  4    3    2
			 \ | /
	(-,0) 5----o----1 (+,0)
			 / | \
          6    7    8
	     -3   -2   -1
	  (-,+)  (0,+)  (+,+)

	*/

	int p = -1; // initially, no position
	if (d.x > 0)
	{
		if (d.y < 0)
			p = 2;
		else
		if (d.y > 0)
			p = -1; // eq pos 8;
		else
			p = 1;
	}
	else
	if (d.x < 0)
	{
		if (d.y < 0)
			p = 4;
		else
		if (d.y > 0)
			p = -3; // eq pos 6;
		else
			p = 5;
	}
	else // (d.x == 0)
	{
		if (d.y < 0)
			p = 3;
		else
		if (d.y > 0)
			p = -2; // eq pos 7;
		else
			p = 0;
	}
	return p;
}

// returns T if c1 is better than c2 as a contour walking pixel candidate
/*static*/ bool ContourFeature::better_step (const Pixel2 &o, const Pixel2 &c1, const Pixel2 &c2)
{
	Pixel2 dif1 = c1 - o,
		dif2 = c2 - o;
	int pos1 = get_dial_pos (dif1),
		pos2 = get_dial_pos (dif2);
	return pos1 > pos2;
}

/*static*/ bool ContourFeature::prune_cands (std::vector<Pixel2> &cands, const Pixel2 &o)
{
	if (cands.empty())
		return false;

	if (cands.size() == 1)
		return true;

	Pixel2 best = cands[0];
	for (int i=1; i<cands.size(); i++)
		if (better_step (o, cands[i], best))
			best = cands[i];

	cands.clear();
	cands.push_back (best);
	return true;
}

//
// given a starting element "t" ("tip") of set "R" ("raw"), returns loop's length or 0 if "t" leads to no closing
//
/*static*/ int ContourFeature::check_loop(
	// out
	std::vector<Pixel2>& S,
	// in
	const std::list<Pixel2>& R,
	const Pixel2& origin)
{
	// void loop?
	if (R.size() == 0)
		return 0;

	// worker vars
	std::list<Pixel2> U (R.begin(), R.end());
	std::vector<Pixel2> P;
	int looplen = 0;

	//	--initialize vector 'ordered' (S) with 1st pixel of 'unordered' (U) to prevent finding it as a loop element
	S.push_back (origin);
	U.remove (origin);
	Pixel2 pxTip = origin;

	//	-- harvest items of 'unordered' 
	while (U.size())
	{
		//	--find tip's neighbors 
		std::vector<Pixel2> cands = find_cands (U, pxTip);

		//	--do we have pants?
		size_t n_cands = cands.size();
		if (n_cands > 1) 
		{
			P.push_back (pxTip);
		}

		bool okCand = prune_cands (cands, pxTip);

		switch (cands.size())
		{
		case 0:
			{
				// Reached the end of the chain. 
				// Can we close it and observe a loop ("origin" is "pxTip's" neighbor), or we have a loose chain (pxTip has no neighbors)?
				Pixel2 dist2org = pxTip - origin;
				if (std::fabs(dist2org.x) == 1 || std::fabs(dist2org.y) == 1)
				{
					looplen++;
					return looplen;
				}
				else
				{
					if (P.empty())
						return 0;	// not a loop
					else
					{
						// restore the pre-bifurcation tip and proceed with the other (initially rejected) candidate
						pxTip = P.back();
						P.pop_back();
					}
				}
			}
			break;
		case 1:
			// trivial loop element
			looplen++;
			pxTip = cands[0];
			S.push_back(pxTip);
			U.remove(pxTip);
			break;
		default:
			// 2-branch or 3-branch: normally we should not fall through here
			return 0;	// not a loop
			break;
		}
	}

	return looplen;
}

/*static*/ void ContourFeature::gather_multicontour (
	// out
	std::vector<std::vector<Pixel2>>& multicountour,
	// in
	std::vector<PixIntens> P,	// padded image
	int w,
	int h,
	int verbose)
{
	const int BLANK = 0;
	bool inside = false;
	int pos = 0;

	//==== Prepare a contour ("border") image
	std::vector<PixIntens> borderImage((h + 2) * (w + 2), 0);

	// Initialize the entire image to blank
	for (int y = 0; y < (h + 2); y++)
		for (int x = 0; x < (w + 2); x++)
			borderImage.at(x + y * (w + 2)) = BLANK;

	//==== Scan the padded image and fill the border one

	for (int y = 0; y < (h + 2); y++)
	{
		for (int x = 0; x < (w + 2); x++)
		{
			pos = x + y * (w + 2);

			// Scan for a non-blank pixel
			PixIntens bi, pi;
			bi = borderImage.at(pos);
			pi = P.at(pos);
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
				borderImage.at(pos) = P.at(pos); // Non-blank

				int checkLocationNr = 1;	// The neighbor number of the location we want to check for a new border point
				int checkPosition;			// The corresponding absolute array address of checkLocationNr
				int newCheckLocationNr; 	// Variable that holds the neighborhood position we want to check if we find a new border at checkLocationNr
				int startPos = pos;			// Set start position
				int counter = 0; 			// Counter is used for the jacobi stop criterion
				int counter2 = 0; 			// Counter2 is used to determine if the point we have discovered is one single point

				// Defines the neighborhood offset position from current position and the neighborhood
				// position we want to check next if we find a new border at checkLocationNr
				int neighborhood[8][2] =
				{
						{-1,7},
						{-3 - w,7},
						{-w - 2,1},
						{-1 - w,1},
						{1,3},
						{3 + w,3},
						{w + 2,5},
						{1 + w,5}
				};

				// Trace around the neighborhood
				while (true)
				{
					checkPosition = pos + neighborhood[checkLocationNr - 1][0];
					newCheckLocationNr = neighborhood[checkLocationNr - 1][1];

					PixIntens pi2 = 0;
					if (checkPosition >= P.size())		// we're done if we start checking outside the image
						break;
					pi2 = P.at(checkPosition);
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
						borderImage.at(checkPosition) = P.at(checkPosition); // Non-blank
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

	if (verbose >= 4)
		dump_2d_image_1d_layout(borderImage, w + 2, h + 2, "\n\nPadded contour image \n", "\n\n");

	//==== remove padding 
	std::vector<Pixel2> C;	// current temp contour

	// gather contour pixels undecorating their intensities back to original values
	Pixel2 lastNonzeroPx(0, 0, 0);
	for (int y = 0; y < h + 2; y++)
		for (int x = 0; x < w + 2; x++)
		{
			size_t idx = x + y * (w + 2);
			auto inte = borderImage.at(idx);
			if (inte)
			{
				// this pixel may happen to be isolated (a speckle), nonetheless, remember it 
				// as we'll need to report it as a degenerate contour if no properly neighbored 
				// pixel group is found
				lastNonzeroPx = { x, y, inte - 1 };

				// register a pixel only if it has any immediate neighbor
				bool hasNeig = false;
				if (x > 0)	// left neighbor
				{
					size_t idxNeig = (x - 1) + y * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x < w - 1)	// right neighbor
				{
					size_t idxNeig = (x + 1) + y * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (y > 0)	// upper neighbor
				{
					size_t idxNeig = x + (y - 1) * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (y < h - 1)	// lower neighbor
				{
					size_t idxNeig = x + (y + 1) * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x > 0 && y > 0)	// upper left neighbor
				{
					size_t idxNeig = (x - 1) + (y - 1) * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x < w - 1 && y > 0)	// upper right neighbor
				{
					size_t idxNeig = (x + 1) + (y - 1) * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x > 0 && y < h - 1)	// lower left neighbor
				{
					size_t idxNeig = (x - 1) + (y + 1) * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (x < w - 1 && y < h - 1)	// lower right neighbor
				{
					size_t idxNeig = (x + 1) + (y + 1) * (w + 2);
					hasNeig = hasNeig || borderImage.at(idxNeig) != 0;
				}
				if (!hasNeig)
					continue;
				// pixel is good, save it
				Pixel2 p(x, y, inte - 1);
				C.push_back(p);
			}
		}

	if (verbose >= 4)
		dump_2d_image_with_vertex_chain(borderImage, C, w + 2, h + 2, "\n\nUnsorted contour cloud (on padded image)\n", "\n\n");

	// no candidates for a contour?
	if (C.size() == 0)
		return;

	// C - raw unordered contour
	// U - unordered subset of C subject to order
	// S - ordered subset of C

	std::list<Pixel2> U (C.begin(), C.end());

	// fix X-crossing points
	int n_xxings = 0;
	for (Pixel2 p : U)
	{
		Pixel2 pN (p.x, p.y-1, p.inten),
			pS (p.x, p.y+1, p.inten),
			pW (p.x-1, p.y, p.inten),
			pE (p.x+1, p.y, p.inten);
		auto itN = std::find(U.begin(), U.end(), pN);
		auto itS = std::find(U.begin(), U.end(), pS);
		auto itE = std::find(U.begin(), U.end(), pE);
		auto itW = std::find(U.begin(), U.end(), pW);
		if (itN != U.end() && itS != U.end() && itW != U.end() && itE != U.end())
		{
			U.remove(p);
			n_xxings++;
		}
	}

	// find contour by contour
	while (!U.empty())
	{
		std::vector<Pixel2> S;
		Pixel2 pxTip = U.front();

		int looplen = check_loop (S, U, pxTip);
		if (looplen)
		{
			if (verbose >= 4)
				dump_2d_image_with_vertex_chain (borderImage, S, w+2, h+2, "\n\nLoop\n", "||S||=" + std::to_string(S.size()) + "\n");
			
			multicountour.push_back(S);

			// subtract the newly found contour from U
			for (Pixel2 & p : S)
				U.remove (p);
		}
		else
		{
			// failed loop, look into why it failed
			dump_2d_image_with_vertex_chain (borderImage, S, w+2, h+2, "\n\nfailed loop\n", "||S||=" + std::to_string(S.size()) + "\n");
		}
	}
}

/*static*/ void ContourFeature::buildRegularContour(LR& r, const Fsettings& s)
{
	// cast ROI's pixel cloud to a padded image 

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

	if (STNGS_VERBOSLVL(s) >= 4)
		dump_2d_image_1d_layout(paddedImage, width + 2, height + 2, "\n\n\n ContourFeature / buildRegularContour / Padded image ROI " + std::to_string(r.aabb.get_width()) + " W " + std::to_string(r.aabb.get_height()) + " H \n", "\n\n\n");

	// gather all the contours
	gather_multicontour (r.multicontour_, paddedImage, width+2, height+2, STNGS_VERBOSLVL(s));

	// fix pixel positions back to absolute
	for (std::vector<Pixel2> &K : r.multicontour_)
		for (Pixel2 &p : K)
		{
			p.x += base_x;
			p.y += base_y;
		}
}

void ContourFeature::buildRegularContour_nontriv (LR& r, const Fsettings& s)
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
	
	r.multicontour_.clear();
	r.multicontour_.push_back (std::vector<Pixel2>());

	for (int y = 0; y < height + 2; y++)
		for (int x = 0; x < width + 2; x++)
		{
			size_t idx = x + y * (width + 2);
			auto inte = borderImage[idx];
			if (inte)
			{
				Pixel2 p(x, y, inte - 1);	// Undecorate the intensity
				r.multicontour_[0].push_back(p);
			}
		}

	//==== Reorder
	//	--containers for unordered (temp) and ordered (result) pixels
	std::list<Pixel2> unordered (r.multicontour_[0].begin(), r.multicontour_[0].end());
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
			if (STNGS_VERBOSLVL(s) >= 4)
			{
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
			}
			break;
		}
	}

	// replace the unordered contour with ordered one
	r.multicontour_[0] = ordered;
}


void ContourFeature::buildWholeSlideContour(LR& r)
{
	// Create 4 slide corner vertices of slide's max intensity
	PixIntens maxI = r.aux_max;
	Pixel2 tl (r.aabb.get_xmin(), r.aabb.get_ymin(), maxI),
		tr (r.aabb.get_xmax(), r.aabb.get_ymin(), maxI),
		bl (r.aabb.get_xmin(), r.aabb.get_ymax(), maxI),
		br (r.aabb.get_xmax(), r.aabb.get_ymax(), maxI);

	std::vector<Pixel2> K;
	K.push_back(tl);
	K.push_back(tr);
	K.push_back(br);
	K.push_back(bl);
	K.clear();
	r.multicontour_.push_back (K);
}

void ContourFeature::calculate (LR& r, const Fsettings& s)
{
	if (s[(int)NyxSetting::SINGLEROI].bval)	// former Nyxus::theEnvironment.singleROI
		buildWholeSlideContour(r);
	else
		buildRegularContour (r, s);

	// flat representation
	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	//=== Calculate the features
	fval_PERIMETER = 0;
	size_t clen = K.size();
	for (size_t i = 0; i < clen; i++)
		if (i == 0)
		{
			Pixel2& p1 = K[clen-1],
				& p2 = K[i];
			fval_PERIMETER += std::sqrt(p1.sqdist(p2));
		}
		else
		{
			Pixel2& p1 = K[i-1],
				& p2 = K[i];
			fval_PERIMETER += std::sqrt(p1.sqdist(p2));
		}

	fval_DIAMETER_EQUAL_PERIMETER = fval_PERIMETER / M_PI;
	auto [cmin, cmax, cmean, cstddev] = calc_min_max_mean_stddev_intensity (K);
	fval_EDGE_MEAN_INTENSITY = cmean;
	fval_EDGE_STDDEV_INTENSITY = cstddev;
	fval_EDGE_MAX_INTENSITY  = cmax;
	fval_EDGE_MIN_INTENSITY = cmin;

	fval_EDGE_INTEGRATEDINTENSITY = 0;
	for (const Pixel2 &px : K)
		fval_EDGE_INTEGRATEDINTENSITY += px.inten;
}

void ContourFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void ContourFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	buildRegularContour_nontriv(r, s);	// leaves the contour pixels in field 'contour'

	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	//=== Calculate the features
	fval_PERIMETER = (StatsInt) K.size();
	fval_DIAMETER_EQUAL_PERIMETER = fval_PERIMETER / M_PI;
	auto [cmin, cmax, cmean, cstddev] = calc_min_max_mean_stddev_intensity (K);
	fval_EDGE_MEAN_INTENSITY = cmean;
	fval_EDGE_STDDEV_INTENSITY = cstddev;
	fval_EDGE_MAX_INTENSITY = cmax;
	fval_EDGE_MIN_INTENSITY = cmin;

	fval_EDGE_INTEGRATEDINTENSITY = 0;
	for (const Pixel2 &px : K)
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

void ContourFeature::extract (LR& r, const Fsettings& s)
{
	ContourFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void ContourFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & fst, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		ContourFeature f;
		f.calculate (r, fst);
		f.save_value (r.fvals);
	}
}


