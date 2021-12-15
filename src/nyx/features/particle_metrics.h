#pragma once

#include <unordered_map>
#include <vector>
#include "../roi_data.h"
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

	static void reduce_feret (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce_martin (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce_nassenstein (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	const int NY = 10;
	const int rot_angle_increment = 10;	// degrees

private:
	std::vector<Pixel2>& convex_hull;
};
