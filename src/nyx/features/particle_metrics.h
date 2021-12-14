#pragma once

#include <vector>
#include "pixel.h"

// Longest chord, Feret, Martin, Nassenstein diameters
class ParticleMetrics
{
public:
	ParticleMetrics(std::vector<Pixel2>& _convex_hull);

	void calc_ferret(
		// output:
		double& minFeretDiameter,
		double& minFeretAngle,
		double& maxFeretDiameter,
		double& maxFeretAngle,
		std::vector<double>& all_D);
	void calc_martin(std::vector<double>& D);
	void calc_nassenstein(std::vector<double>& D);
	const int NY = 10;
	const int rot_angle_increment = 10;	// degrees
protected:
	std::vector<Pixel2>& convex_hull;
};
