#pragma once

#include <vector>
#include "../featureset.h"
#include "pixel.h"
#include "../feature_method.h"

class ConvexHullFeature : public FeatureMethod
{
public:
	ConvexHullFeature();

	// Trivial ROI
	void calculate(LR& r);	 // Consumes LR::raw_pixels, populates LR::conv_hull_pixels, calculates 3 features

	// Non-trivial ROI
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
	void osized_calculate (LR& r, ImageLoader& imloader);	// Consumes LR::raw_pixels, populates LR::conv_hull_pixels, calculates the 3 features
	void save_value (std::vector<std::vector<double>>& feature_vals);
	void cleanup_instance();

	// Part of the legacy reduce(), will be gone
	static bool required(const FeatureSet& fs) { return fs.anyEnabled({ CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY }); }

private:
	void build_convex_hull(const std::vector<Pixel2>& contour, std::vector<Pixel2>& convhull);
	static bool compare_locations(const Pixel2& lhs, const Pixel2& rhs);
	bool right_turn(const Pixel2& P1, const Pixel2& P2, const Pixel2& P3);
	double polygon_area(const std::vector<Pixel2>& vertices);

	double area = 0, solidity = 0, circularity = 0;	
};

class ConvexHull
{
public:
	static bool required(const FeatureSet& fs) { return fs.anyEnabled({CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY}); }
	ConvexHull() {}
	void calculate(std::vector<Pixel2>& rawPixels);
	double getSolidity();
	double getArea();
	void clear() { CH.clear(); }

	std::vector<Pixel2> CH;
};
