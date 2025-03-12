#pragma once

#include <vector>
#include "../featureset.h"
#include "pixel.h"
#include "../feature_method.h"

class ConvexHullFeature : public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
			Nyxus::Feature2D::CONVEX_HULL_AREA,
			Nyxus::Feature2D::SOLIDITY,
			Nyxus::Feature2D::CIRCULARITY,
			Nyxus::Feature2D::POLYGONALITY_AVE,
			Nyxus::Feature2D::HEXAGONALITY_AVE,
			Nyxus::Feature2D::HEXAGONALITY_STDDEV
	};

	ConvexHullFeature();

	// Trivial ROI
	void calculate(LR& r);

	// Non-trivial ROI
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
	void osized_calculate (LR& r, ImageLoader& imloader);
	void save_value (std::vector<std::vector<double>>& feature_vals);
	void cleanup_instance();

	// Support of manual reduce
	static bool required(const FeatureSet& fs) 
	{ 
		return fs.anyEnabled (ConvexHullFeature::featureset); 
	}

	static void extract(LR& roi); // extracts the feature of- and saves to ROI

private:
	void build_convex_hull(const std::vector<Pixel2>& contour, std::vector<Pixel2>& convhull);
	void build_convex_hull(const OutOfRamPixelCloud& roi_cloud, std::vector<Pixel2>& convhull);
	static bool compare_locations(const Pixel2& lhs, const Pixel2& rhs);
	bool right_turn(const Pixel2& P1, const Pixel2& P2, const Pixel2& P3);
	double polygon_area(const std::vector<Pixel2>& vertices);

	double area = 0, solidity = 0, circularity = 0;	
};
