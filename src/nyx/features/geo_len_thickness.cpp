#include <iostream>
#include "geodetic_len_thickness.h"

std::tuple<double, double> GeodeticLength_and_Thickness::calculate(StatsInt roiArea, StatsInt roiPerimeter)
{
	//------------------------------Geodetic Length and Thickness------------
	//https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L244
	/*
	* Accroding to imea code: https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L244
	* the definitions of Geodetic Length and Thickness are as follows.
	The geodetic lengths and thickness are approximated by a rectangle
	with the same area and perimeter:
	(1) `area = geodeticlength * thickness`
	(2) `perimeter = 2 * (geodetic_length + thickness)`

	# White the help of Equation (1) we can rewrite Equation (2) to:
	# `geodetic_length**2 - 0.5 * perimeter * geodetic_length + area = 0`
	# Which we can solve with the pq-formula:
	# `geodetic_length = perimeter/4 +- sqrt((perimeter/4)**2 - area)`
	# since only the positive solution makes sense in our application
	*/

	if (roiPerimeter <= 0)
	{
		std::cout << " Perimeter should be a positive value greater than zero" << std::endl;
		return { 0,0 };
	}

	if (roiArea <= 0)
	{
		std::cout << " Area should be a positive value greater than zero" << std::endl;
		return { 0,0 };
	}

	double SqRootTmp = roiPerimeter * roiPerimeter / 16 - (double)roiArea;

	//Make sure value under SqRootTmp is always positive
	if (SqRootTmp < 0) SqRootTmp = 0;

	//Calcuate geodetic_length with pq-formula (see above):
	double geodetic_length = roiPerimeter / 4 + sqrt(SqRootTmp);

	//Calculate thickness by rewriting Equation (2):
	double thickness = roiPerimeter / 2 - geodetic_length;

	return { geodetic_length, thickness };
}

void GeodeticLength_and_Thickness::reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Skip if the contour, convex hull, and neighbors are unavailable, otherwise the related features will be == NAN. Those feature will be equal to the default unassigned value.
		if (r.contour.contour_pixels.size() == 0 || r.convHull.CH.size() == 0 || r.fvals[NUM_NEIGHBORS][0] == 0)
			continue;

		GeodeticLength_and_Thickness glt;
		auto [geoLen, thick] = glt.calculate(r.raw_pixels.size(), (StatsInt)r.fvals[PERIMETER][0]);
		r.fvals[GEODETIC_LENGTH][0] = geoLen;
		r.fvals[THICKNESS][0] = thick;
	}
}

