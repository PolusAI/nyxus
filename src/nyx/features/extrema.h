#pragma once

#include <tuple>
#include <vector>
#include <unordered_map>
#include "../feature_method.h"
#include "../roi_cache.h"
#include "pixel.h"

class ExtremaFeature: public FeatureMethod
{
public:
	ExtremaFeature();

	// Trivial ROI
	void calculate(LR& r);

	// Non-trivial ROI
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
	void osized_calculate (LR& r, ImageLoader& imloader);

	// Result saver
	void save_value(std::vector<std::vector<double>>& feature_vals);

	// Compatibility with manual
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled({
			EXTREMA_P1_Y,
			EXTREMA_P1_X,
			EXTREMA_P2_Y,
			EXTREMA_P2_X,
			EXTREMA_P3_Y,
			EXTREMA_P3_X,
			EXTREMA_P4_Y,
			EXTREMA_P4_X,
			EXTREMA_P5_Y,
			EXTREMA_P5_X,
			EXTREMA_P6_Y,
			EXTREMA_P6_X,
			EXTREMA_P7_Y,
			EXTREMA_P7_X,
			EXTREMA_P8_Y,
			EXTREMA_P8_X });
	}
	std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> get_values();
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	int x1 =0, y1 =0, x2 =0, y2 =0, x3 =0, y3 =0, x4 =0, y4 =0, x5 =0, y5 =0, x6 =0, y6 =0, x7 =0, y7 =0, x8 =0, y8 =0;
};
