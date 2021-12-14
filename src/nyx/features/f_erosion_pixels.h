#pragma once
#include <unordered_map>
#include "../roi_data.h"
#include "image_matrix.h"

// Determine number of erosions that are necessary to fully erase all the pixels in a binary image[1].
// [1] Deutsches Institut für Normung e.V. (2012).DIN ISO 9276 - 6 -
// Darstellung der Ergebnisse von Par - tikelgrößenanalysen:
// Teil 6 : Deskriptive und quantitative Darstellung der Form
// und Morphologie von Partikeln.

class ErosionPixels
{
public:

	ErosionPixels() {}
	int calc_feature (ImageMatrix & );
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

protected:

	const int SANITY_MAX_NUM_EROSIONS = 1000;	// Prevent infinite erosions

	static const int SE_R = 3,	// rows
		SE_C = 3;	// columns
	int strucElem [SE_R][SE_C] = { {0,1,0}, {1,1,1}, {0,1,0} };
};