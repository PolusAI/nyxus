#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "../feature_method.h"
#include "image_matrix_nontriv.h"
#include "convex_hull.h"

ConvexHullFeature::ConvexHullFeature() : FeatureMethod("ConvexHullFeature")
{
	provide_features ({ CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY });
	add_dependencies ({ PERIMETER });
}

void ConvexHullFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals [CONVEX_HULL_AREA][0] = area;
	fvals [SOLIDITY][0] = solidity; 
	fvals [CIRCULARITY][0] = circularity;
}

void ConvexHullFeature::cleanup_instance()
{
	area = solidity = circularity = 0.0;
}

void ConvexHullFeature::calculate (LR& r)
{
	// Build the convex hull
	build_convex_hull(r.raw_pixels, r.convHull_CH);

	// Calculate related features
	double s_hull = polygon_area(r.convHull_CH),
		s_roi = r.raw_pixels.size(), 
		p = r.fvals[PERIMETER][0];
	area = s_hull;
	solidity = s_roi / s_hull;
	circularity = sqrt(4.0 * M_PI * s_roi / (p*p));
}

void ConvexHullFeature::build_convex_hull (const std::vector<Pixel2>& cloud, std::vector<Pixel2>& convhull)
{
	convhull.clear();

	// Skip calculation if the ROI is too small
	if (cloud.size() < 2)
		return;

	std::vector<Pixel2>& upperCH = convhull;
	std::vector<Pixel2> lowerCH;

	size_t n = cloud.size();
	
	//	No need to sort pixels because we accumulate them in a raster pattern without multithreading//
	//	std::vector<Pixel2> cloud = roi_cloud;	// Safely copy the ROI for fear of changing the original pixels order
	//	std::sort (cloud.begin(), cloud.end(), compare_locations);
	//

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

void ConvexHullFeature::build_convex_hull (const OutOfRamPixelCloud& cloud, std::vector<Pixel2>& convhull)
{
	convhull.clear();

	// Skip calculation if the ROI is too small
	if (cloud.size() < 2)
		return;

	std::vector<Pixel2>& upperCH = convhull;
	std::vector<Pixel2> lowerCH;

	size_t n = cloud.size();

//
//	No need to sort pixels because we accumulate them in a raster pattern without multithreading
//	std::vector<Pixel2> cloud = roi_cloud;	// Safely copy the ROI for fear of changing the original pixels order
//	std::sort(cloud.begin(), cloud.end(), compare_locations);
//
 
	// Computing upper convex hull
	upperCH.push_back(cloud[0]);
	upperCH.push_back(cloud[1]);

	for (size_t i = 2; i < n; i++)
	{
		while (upperCH.size() > 1 && (!right_turn(upperCH[upperCH.size() - 2], upperCH[upperCH.size() - 1], cloud[i])))
			upperCH.pop_back();
		upperCH.push_back(cloud[i]);
	}

	// Computing lower convex hull
	lowerCH.push_back(cloud[n-1]);
	lowerCH.push_back(cloud[n-2]);

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
// If two points have the same x-coordinate then we compare their y-coordinates
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
	size_t n = vertices.size();
	for (size_t i = 0; i < n-1; i++)
	{
		const Pixel2 &p1 = vertices[i], 
			&p2 = vertices[i+1];
		area += p1.x * p2.y - p1.y * p2.x;
	}
	const Pixel2& p1 = vertices[0],
		& p2 = vertices[n-1];
	area += p1.x * p2.y - p1.y * p2.x;
	area = std::abs(area) / 2.0;
	return area;
}

void ConvexHullFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	// Build the convex hull
	build_convex_hull (r.raw_pixels_NT, r.convHull_CH);

	// Calculate related features
	double s_hull = polygon_area(r.convHull_CH),
		s_roi = r.raw_pixels_NT.size(),
		p = r.fvals[PERIMETER][0];
	area = s_hull;
	solidity = s_roi / s_hull;
	circularity = sqrt(4.0 * M_PI * s_roi / (p * p));
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

