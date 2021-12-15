#pragma once

#include <unordered_map>
#include "../roi_data.h"

#include <vector>
#include "pixel.h"

class EulerNumber
{
public:
	EulerNumber(std::vector<Pixel2>& P, StatsInt min_x, StatsInt  min_y, StatsInt max_x, StatsInt max_y, int mode);
	long euler_number;	// Calculated via calculate()
	
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	long calculate(std::vector<unsigned char>& I, int height, int width, int mode);
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

