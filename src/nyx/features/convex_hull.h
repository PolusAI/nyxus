#pragma once
#include <vector>
#include "pixel.h"

class ConvexHull
{
public:
	ConvexHull() {}
	void calculate(std::vector<Pixel2>& rawPixels);
	double getSolidity();
	double getArea();

	std::vector<Pixel2> CH;
};
