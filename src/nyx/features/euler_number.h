#pragma once

#include <unordered_map>
#include "../roi_cache.h"

#include <vector>
#include "pixel.h"
#include "aabb.h"
#include "../feature_method.h"

/// @brief The Euler characteristic of a ROI. Equal to the number of 'objects' in the image minus the number of holes in those objects. For modules built to date, the number of 'objects' in the image is always 1.
class EulerNumberFeature: public FeatureMethod
{
public:
	EulerNumberFeature();

	// Trivial ROI
	void calculate(LR& r);

	// Non-trivial ROI
	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {}
	void osized_calculate (LR& r, ImageLoader& imloader);

	// Result saver
	void save_value (std::vector<std::vector<double>>& feature_vals);

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static bool required (const FeatureSet& fs) { return fs.isEnabled(EULER_NUMBER); }

private:
	const int mode = 8;		// Using mode=8 following WNDCHRM example
	long calculate_euler (std::vector<unsigned char>& I, int height, int width, int mode);

	long euler_number = 0;

	static constexpr unsigned char Px[10] = { //MM: 0 or 1 in the left side of << represent binary pixel values
	// P1 - single pixel  8/4/2/1
	(1 << 3) | (0 << 2) |
	(0 << 1) | (0 << 0),
	(0 << 3) | (1 << 2) |
	(0 << 1) | (0 << 0),
	(0 << 3) | (0 << 2) |
	(1 << 1) | (0 << 0),
	(0 << 3) | (0 << 2) |
	(0 << 1) | (1 << 0),
		// P3 - 3-pixel   7/11/13/14
		(0 << 3) | (1 << 2) |
		(1 << 1) | (1 << 0),
		(1 << 3) | (0 << 2) |
		(1 << 1) | (1 << 0),
		(1 << 3) | (1 << 2) |
		(0 << 1) | (1 << 0),
		(1 << 3) | (1 << 2) |
		(1 << 1) | (0 << 0),
		// Pd - diagonals  9/6
		(1 << 3) | (0 << 2) |
		(0 << 1) | (1 << 0),
		(0 << 3) | (1 << 2) |
		(1 << 1) | (0 << 0)
	};
};
