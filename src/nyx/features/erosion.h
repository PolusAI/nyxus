#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"

/// @brief Determine number of erosions that are necessary to fully erase all the pixels in a binary image.
class ErosionPixels_feature
{
public:
	static bool required(FeatureSet& fs) { return fs.anyEnabled({ EROSIONS_2_VANISH, EROSIONS_2_VANISH_COMPLEMENT }); }
	ErosionPixels_feature (const ImageMatrix & im);
	int get_feature_value();
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:

	const int SANITY_MAX_NUM_EROSIONS = 1000;	// Prevent infinite erosions

	static const int SE_R = 3,	// rows
		SE_C = 3;	// columns
	
	int strucElem [SE_R][SE_C] = { {0,1,0}, {1,1,1}, {0,1,0} };

	int numErosions = 0;	// Feature value
};