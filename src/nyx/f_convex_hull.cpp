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
#include "f_convex_hull.h"


// Sort criterion: points are sorted with respect to their x-coordinate.
//                 If two points have the same x-coordinate then we compare
//                 their y-coordinates
bool sortPoints(const Pixel2& lhs, const Pixel2& rhs)
{
	return (lhs.x < rhs.x) || (lhs.x == rhs.x && lhs.y < rhs.y);
}

// Check if three points make a right turn using cross product
bool right_turn(const Pixel2& P1, const Pixel2& P2, const Pixel2& P3)
{
	return ((P3.x - P1.x) * (P2.y - P1.y) - (P3.y - P1.y) * (P2.x - P1.x)) > 0;
}

double getPolygonArea(std::vector<Pixel2>& vertices)
{
	double area = 0.0;
	int n = (int)vertices.size();
	for (int i = 0; i < n - 1; i++)
	{
		Pixel2& p_i = vertices[i], & p_ii = vertices[i + 1];
		area += p_i.x * p_ii.y - p_i.y * p_ii.x;
	}
	area += vertices[n - 1].x * vertices[0].y - vertices[0].x * vertices[n - 1].y;
	area = std::abs(area) / 2.0;
	return area;
}

void ConvexHull::calculate(std::vector<Pixel2> & point_cloud) 
{
	CH.clear();

	std::vector<Pixel2>& upperCH = CH;
	std::vector<Pixel2> lowerCH;

	int n = (int)point_cloud.size();

	//Sorting points
	sort(point_cloud.begin(), point_cloud.end(), sortPoints);

	//Computing upper convex hull
	upperCH.push_back(point_cloud[0]);
	upperCH.push_back(point_cloud[1]);

	for (int i = 2; i < n; i++)
	{
		while (upperCH.size() > 1 && (!right_turn(upperCH[upperCH.size() - 2], upperCH[upperCH.size() - 1], point_cloud[i])))
			upperCH.pop_back();
		upperCH.push_back(point_cloud[i]);
	}

	//Computing lower convex hull
	lowerCH.push_back(point_cloud[n - 1]);
	lowerCH.push_back(point_cloud[n - 2]);

	for (int i = 2; i < n; i++)
	{
		while (lowerCH.size() > 1 && (!right_turn(lowerCH[lowerCH.size() - 2], lowerCH[lowerCH.size() - 1], point_cloud[n - i - 1])))
			lowerCH.pop_back();
		lowerCH.push_back(point_cloud[n - i - 1]);
	}

	// We could use 
	//  upperCH.insert (upperCH.end(), lowerCH.begin(), lowerCH.end());
	// but we don't need duplicate points in the result contour
	for (auto& p : lowerCH)
		if (std::find(upperCH.begin(), upperCH.end(), p) == upperCH.end())
			upperCH.push_back(p);
}

double ConvexHull::getArea ()
{
	return getPolygonArea(CH);
}

