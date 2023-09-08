#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"

/// @brief Determine number of erosions that are necessary to fully erase all the pixels in a binary image.
class ErosionPixelsFeature: public FeatureMethod
{
public:
	ErosionPixelsFeature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	static bool required(FeatureSet& fs) { return fs.anyEnabled({ EROSIONS_2_VANISH, EROSIONS_2_VANISH_COMPLEMENT }); }

private:

	const int SANITY_MAX_NUM_EROSIONS = 1000;	// Prevent infinite erosions

	// Structuring element definition:
	static const int SE_R = 3, SE_C = 3; 	// rows, columns
	int strucElem [SE_R][SE_C] = { {0,1,0}, {1,1,1}, {0,1,0} };

#if 0
	// 5x5
	static const int SE_R = 5, SE_C = 5;	 // rows, columns
	int strucElem[SE_R][SE_C] = {
		{0,     0,     1,     0,     0},
		{0,     1,     1,     1,     0},
		{1,     1,     1,     1,     1},
		{0,     1,     1,     1,     0},
		{0,     0,     1,     0,     0}
	};
#endif

#if 0
	// 7x7
	static const int SE_R = 7, SE_C = 7;		// rows, columns

	int strucElem[SE_R][SE_C] = {
		{0,     0,     0,     1,     0,     0,     0},
		{0,     0,     1,     1,     1,     0,     0},
		{0,     1,      1,      1,      1,      1,      0},
		{1,     1,      1,      1,      1,      1,      1},
		{0,     1,      1,      1,      1,      1,      0},
		{0,     0,      1,      1,      1,      0,      0},
		{0,     0,      0,      1,      0,      0,      0}
	};
#endif

#if 0
	// 11x11
	static const int SE_R = 11, SE_C = 11;		// rows, columns
	int strucElem[SE_R][SE_C] = {
		{0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0},
		{0,     0,     0,     0,     1,     1,     1,     0,     0,     0,     0},
		{0,     0,     0,     1,     1,     1,     1,     1,     0,     0,     0},
		{0,     0,     1,     1,     1,     1,     1,     1,     1,     0,     0},
		{0,     1,     1,     1,     1,     1,     1,     1,     1,     1,     0},
		{1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1},
		{0,     1,     1,     1,     1,     1,     1,     1,     1,     1,     0},
		{0,     0,     1,     1,     1,     1,     1,     1,     1,     0,     0},
		{0,     0,     0,     1,     1,     1,     1,     1,     0,     0,     0},
		{0,     0,     0,     0,     1,     1,     1,     0,     0,     0,     0},
		{0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0}
	};
#endif

	int numErosions = 0;	// Feature value
};
