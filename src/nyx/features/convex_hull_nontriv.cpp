#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "../feature_method.h"
#include "image_matrix_nontriv.h"
#include "convex_hull.h"

ConvexHullFeature::ConvexHullFeature() : FeatureMethod("ConvexHullFeature")
{
	provide_features ({CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY});
}

void ConvexHullFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals [CONVEX_HULL_AREA][0] = area;
	fvals[SOLIDITY][0] = solidity; 
	fvals [CIRCULARITY][0] = circularity;
}

void ConvexHullFeature::cleanup_instance()
{
	area = solidity = circularity = 0.0;
}

void ConvexHullFeature::calculate (LR& r)
{
	build_convex_hull(r.raw_pixels, r.convHull_CH);
	double s_hull = polygon_area(r.convHull_CH),
		s_roi = r.raw_pixels.size(), 
		p = r.fvals[PERIMETER][0];
	area = s_hull;
	solidity = s_roi / s_hull;
	circularity = 4.0 * M_PI * s_roi / (p*p);
}

void ConvexHullFeature::build_convex_hull (const std::vector<Pixel2>& roi_cloud, std::vector<Pixel2>& convhull)
{
	convhull.clear();

	// Skip calculation if the ROI is too small
	if (roi_cloud.size() < 2)
		return;

	std::vector<Pixel2>& upperCH = convhull;
	std::vector<Pixel2> lowerCH;

	size_t n = roi_cloud.size();

	// Sorting points
	std::vector<Pixel2> cloud = roi_cloud;	// Safely copy the ROI for fear of changing the original pixels order
	std::sort (cloud.begin(), cloud.end(), compare_locations);

	// Computing upper convex hull
	upperCH.push_back (cloud[0]);
	upperCH.push_back (cloud[1]);

	for (size_t i = 2; i < n; i++)
	{
		while (upperCH.size() > 1 && (!right_turn(upperCH[upperCH.size() - 2], upperCH[upperCH.size() - 1], cloud[i])))
			upperCH.pop_back();
		upperCH.push_back(cloud[i]);
	}

	// Computing lower convex hull
	lowerCH.push_back(cloud[n - 1]);
	lowerCH.push_back(cloud[n - 2]);

	for (size_t i = 2; i < n; i++)
	{
		while (lowerCH.size() > 1 && (!right_turn(lowerCH[lowerCH.size() - 2], lowerCH[lowerCH.size() - 1], cloud[n - i - 1])))
			lowerCH.pop_back();
		lowerCH.push_back(cloud[n - i - 1]);
	}

	// We could use 
	//		upperCH.insert (upperCH.end(), lowerCH.begin(), lowerCH.end());
	// but we don't need duplicate points in the result contour
	for (auto& p : lowerCH)
		if (std::find(upperCH.begin(), upperCH.end(), p) == upperCH.end())
			upperCH.push_back(p);
}

// Sort criterion: points are sorted with respect to their x-coordinate.
//                 If two points have the same x-coordinate then we compare
//                 their y-coordinates
bool ConvexHullFeature::compare_locations (const Pixel2& lhs, const Pixel2& rhs)
{
	return (lhs.x < rhs.x) || (lhs.x == rhs.x && lhs.y < rhs.y);
}

// Check if three points make a right turn using cross product
bool ConvexHullFeature::right_turn (const Pixel2& P1, const Pixel2& P2, const Pixel2& P3)
{
	return ((P3.x - P1.x) * (P2.y - P1.y) - (P3.y - P1.y) * (P2.x - P1.x)) > 0;
}

double ConvexHullFeature::polygon_area (const std::vector<Pixel2>& vertices)
{
	// Blank polygon?
	if (vertices.size() == 0)
		return 0.0;

	// Normal polygon
	double area = 0.0;
	int n = (int)vertices.size();
	for (int i = 0; i < n - 1; i++)
	{
		const Pixel2& p_i = vertices[i], & p_ii = vertices[i + 1];
		area += p_i.x * p_ii.y - p_i.y * p_ii.x;
	}
	area += vertices[n - 1].x * vertices[0].y - vertices[0].x * vertices[n - 1].y;
	area = std::abs(area) / 2.0;
	return area;
}

void ConvexHullFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	r.convHull_CH.clear();

	size_t n = r.osized_pixel_cloud.get_size();

	// Skip calculation if the ROI is too small
	if (n < 2)
		return;

	std::vector<Pixel2>& upperCH = r.convHull_CH;
	std::vector<Pixel2> lowerCH;

	// Computing upper convex hull
	upperCH.push_back(r.osized_pixel_cloud.get_at(0));
	upperCH.push_back(r.osized_pixel_cloud.get_at(1));

	for (size_t i = 2; i < n; i++)
	{
		while (upperCH.size() > 1 && (!right_turn(upperCH[upperCH.size() - 2], upperCH[upperCH.size() - 1], r.osized_pixel_cloud.get_at(i))))
			upperCH.pop_back();
		upperCH.push_back(r.osized_pixel_cloud.get_at(i));
	}

	// Computing lower convex hull
	lowerCH.push_back(r.osized_pixel_cloud.get_at(n - 1));
	lowerCH.push_back(r.osized_pixel_cloud.get_at(n - 2));

	for (size_t i = 2; i < n; i++)
	{
		while (lowerCH.size() > 1 && (!right_turn(lowerCH[lowerCH.size() - 2], lowerCH[lowerCH.size() - 1], r.osized_pixel_cloud.get_at(n - i - 1))))
			lowerCH.pop_back();
		lowerCH.push_back(r.osized_pixel_cloud.get_at(n - i - 1));
	}

	// We could use 
	//		upperCH.insert (upperCH.end(), lowerCH.begin(), lowerCH.end());
	// but we don't need duplicate points in the result contour
	for (auto& p : lowerCH)
		if (std::find(upperCH.begin(), upperCH.end(), p) == upperCH.end())
			upperCH.push_back(p);
}


namespace Nyxus
{
	void parallelReduceConvHull (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
	{
		for (auto i = start; i < end; i++)
		{
			int lab = (*ptrLabels)[i];
			LR& r = (*ptrLabelData)[lab];

			if (r.has_bad_data())
				continue;

			ConvexHullFeature fea;
			fea.calculate (r);
			fea.save_value (r.fvals);
		}
	}
}

