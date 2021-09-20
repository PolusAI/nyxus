#pragma once

#include <vector>
#include "pixel.h"

class Rotation
{
public:
	static void rotate_around_center(
		// in 
		const std::vector<Pixel2>& P,
		float angle_deg,
		// out
		std::vector<Pixel2>& P_rot);

};