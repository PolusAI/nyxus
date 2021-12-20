#pragma once

#include <vector>
#include "../featureset.h"
#include "pixel.h"

class ConvexHull
{
public:
	static bool required(const FeatureSet& fs) { return fs.anyEnabled({CONVEX_HULL_AREA, SOLIDITY, CIRCULARITY}); }
	ConvexHull() {}
	void calculate(std::vector<Pixel2>& rawPixels);
	double getSolidity();
	double getArea();

	std::vector<Pixel2> CH;
};
