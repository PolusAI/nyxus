#pragma once

#include <unordered_map>
#include "../roi_data.h"
#include <tuple>

class Hexagonality_and_Polygonality
{
public:
	static bool required (const FeatureSet& fs) { return fs.anyEnabled({ POLYGONALITY_AVE, HEXAGONALITY_AVE, HEXAGONALITY_STDDEV }); }
	Hexagonality_and_Polygonality() {}
	std::tuple<double, double, double> calculate(int num_neighbors, int roi_area, int roi_perimeter, double convhull_area, double min_feret_diam, double max_feret_diam);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
};

