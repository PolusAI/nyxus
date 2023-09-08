#pragma once

#include <vector>
#include "aabb.h"
#include "pixel.h"
#include "image_matrix_nontriv.h"

/// @brief Helper class for rotation calculations
class Rotation
{
public:
	static void rotate_around_center(
		// [in]
		const std::vector<Pixel2>& P,
		float angle_deg,
		// [out]
		std::vector<Pixel2>& P_rot);

	static void rotate_cloud(
		// [in]
		const std::vector<Pixel2>& P,
		const double cx,
		const double cy,
		float theta,
		// [out]
		std::vector<Pixel2>& P_rot);

	static void rotate_cloud_NT(
		// [in]
		const OutOfRamPixelCloud & cloud,
		const double cx,
		const double cy,
		float theta_radians,
		// [out]
		OutOfRamPixelCloud & rotated_cloud);
};
