#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include <tuple>
# include "../feature_method.h"

/// @brief Estimates ROI's proximity to a hexagon and polygon.

class HexagonalityPolygonalityFeature: public FeatureMethod
{
public:
	HexagonalityPolygonalityFeature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Compatibility with manual reduce
	static bool required (const FeatureSet& fs) { return fs.anyEnabled({ Nyxus::Feature2D::POLYGONALITY_AVE, Nyxus::Feature2D::HEXAGONALITY_AVE, Nyxus::Feature2D::HEXAGONALITY_STDDEV }); }

private:
	double polyAve = 0, hexAve = 0, hexSd = 0;
	static const int novalue = -1;	// feature falue made intentionally blank and unsuitable for model training
};

