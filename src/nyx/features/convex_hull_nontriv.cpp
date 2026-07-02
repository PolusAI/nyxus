#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <algorithm>
#include <cmath>
#include <numeric>
#include "../dataset.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "image_matrix_nontriv.h"
#include "convex_hull.h"

using namespace Nyxus;

// Number of integer lattice points lying ON the hull boundary = sum over edges of gcd(|dx|,|dy|).
// By Pick's theorem the pixel-count-equivalent hull area (matching scikit-image convex_area, which
// counts the pixels of the rasterised hull) is shoelace_area + boundary_points/2 + 1. The bare
// shoelace area runs through pixel CENTRES and under-counts coverage, so for small/elongated ROIs
// it falls below the ROI pixel count -> SOLIDITY > 1 (impossible). This correction fixes that.
static long hull_boundary_points (const std::vector<Pixel2>& v)
{
	size_t n = v.size();
	if (n < 2)
		return 0;
	long B = 0;
	for (size_t i = 0; i < n; i++)
	{
		const Pixel2& p1 = v[i];
		const Pixel2& p2 = v[(i + 1) % n];
		long dx = (long)p2.x - (long)p1.x; if (dx < 0) dx = -dx;
		long dy = (long)p2.y - (long)p1.y; if (dy < 0) dy = -dy;
		B += std::gcd(dx, dy);
	}
	return B;
}

ConvexHullFeature::ConvexHullFeature() : FeatureMethod("ConvexHullFeature")
{
	provide_features ({ Feature2D::CONVEX_HULL_AREA, Feature2D::SOLIDITY, Feature2D::CIRCULARITY });
	add_dependencies ({ Feature2D::PERIMETER });
}

void ConvexHullFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals [(int)Feature2D::CONVEX_HULL_AREA][0] = area;
	fvals [(int)Feature2D::SOLIDITY][0] = solidity;
	fvals [(int)Feature2D::CIRCULARITY][0] = circularity;
}

void ConvexHullFeature::cleanup_instance()
{
	area = solidity = circularity = 0.0;
}

void ConvexHullFeature::calculate (LR& r, const Fsettings& s)
{
	// Build the convex hull
	build_convex_hull(r.raw_pixels, r.convHull_CH);

	// Calculate related features. Pixel-count-equivalent hull area (Pick's theorem) so the hull is
	// measured on the same basis as the ROI (pixel count) -> SOLIDITY <= 1.
	double s_hull = polygon_area(r.convHull_CH) + hull_boundary_points(r.convHull_CH) / 2.0 + 1.0,
		s_roi = r.raw_pixels.size(),
		p = r.fvals[(int)Feature2D::PERIMETER][0];
	area = s_hull;
	solidity = (s_hull > 0.0) ? s_roi / s_hull : 0.0;
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

	// Monotonic-chain hull construction requires x/y-sorted input.
	// Avoid copying/sorting when the ingestion path already provides it.
	const std::vector<Pixel2>* orderedCloud = &cloud;
	std::vector<Pixel2> sortedCloud;
	if (!std::is_sorted(cloud.begin(), cloud.end(), compare_locations))
	{
		sortedCloud = cloud;
		std::sort(sortedCloud.begin(), sortedCloud.end(), compare_locations);
		orderedCloud = &sortedCloud;
	}
	const std::vector<Pixel2>& ordered = *orderedCloud;
	size_t n = ordered.size();

	// Computing upper convex hull
	upperCH.push_back (ordered[0]);
	upperCH.push_back (ordered[1]);

	for (size_t i = 2; i < n; i++)
	{
		while (upperCH.size() > 1 && (!right_turn(upperCH[upperCH.size() - 2], upperCH[upperCH.size() - 1], ordered[i])))
			upperCH.pop_back();
		upperCH.push_back(ordered[i]);
	}

	// Computing lower convex hull
	lowerCH.push_back(ordered[n - 1]);
	lowerCH.push_back(ordered[n - 2]);

	for (size_t i = 2; i < n; i++)
	{
		while (lowerCH.size() > 1 && (!right_turn(lowerCH[lowerCH.size() - 2], lowerCH[lowerCH.size() - 1], ordered[n - i - 1])))
			lowerCH.pop_back();
		lowerCH.push_back(ordered[n - i - 1]);
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

	// Monotonic-chain hull construction requires x/y-sorted input.
	// We must materialize out-of-RAM pixels, but can still skip sorting
	// when their storage order is already monotonic.
	std::vector<Pixel2> orderedCloud;
	orderedCloud.reserve(cloud.size());
	bool alreadySorted = true;
	for (size_t i = 0; i < cloud.size(); ++i)
	{
		Pixel2 px = cloud.get_at(i);
		if (!orderedCloud.empty() && compare_locations(px, orderedCloud.back()))
			alreadySorted = false;
		orderedCloud.push_back(px);
	}
	if (!alreadySorted)
		std::sort(orderedCloud.begin(), orderedCloud.end(), compare_locations);
	size_t n = orderedCloud.size();
 
	// Computing upper convex hull
	upperCH.push_back(orderedCloud[0]);
	upperCH.push_back(orderedCloud[1]);

	for (size_t i = 2; i < n; i++)
	{
		while (upperCH.size() > 1 && (!right_turn(upperCH[upperCH.size() - 2], upperCH[upperCH.size() - 1], orderedCloud[i])))
			upperCH.pop_back();
		upperCH.push_back(orderedCloud[i]);
	}

	// Computing lower convex hull
	lowerCH.push_back(orderedCloud[n-1]);
	lowerCH.push_back(orderedCloud[n-2]);

	for (size_t i = 2; i < n; i++)
	{
		while (lowerCH.size() > 1 && (!right_turn(lowerCH[lowerCH.size() - 2], lowerCH[lowerCH.size() - 1], orderedCloud[n - i - 1])))
			lowerCH.pop_back();
		lowerCH.push_back(orderedCloud[n - i - 1]);
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
	const Pixel2& p1 = vertices[n-1],
		& p2 = vertices[0];
	area += p1.x * p2.y - p1.y * p2.x;
	area = std::abs(area) / 2.0;
	return area;
}

void ConvexHullFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	// Build the convex hull
	build_convex_hull (r.raw_pixels_NT, r.convHull_CH);

	// Calculate related features (Pick's-theorem pixel-count-equivalent hull area; see calculate())
	double s_hull = polygon_area(r.convHull_CH) + hull_boundary_points(r.convHull_CH) / 2.0 + 1.0,
		s_roi = r.raw_pixels_NT.size(),
		p = r.fvals[(int)Feature2D::PERIMETER][0];
	area = s_hull;
	solidity = (s_hull > 0.0) ? s_roi / s_hull : 0.0;
	circularity = sqrt(4.0 * M_PI * s_roi / (p * p));
}

void ConvexHullFeature::extract (LR& r, const Fsettings& s)
{
	ConvexHullFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

namespace Nyxus
{
	void parallelReduceConvHull (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
	{
		for (auto i = start; i < end; i++)
		{
			int lab = (*ptrLabels)[i];
			LR& r = (*ptrLabelData)[lab];

			if (r.has_bad_data())
				continue;

			ConvexHullFeature f;
			f.calculate (r, s);
			f.save_value (r.fvals);
		}
	}
}

