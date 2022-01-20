#define _USE_MATH_DEFINES
#include <cmath>
#include "rotation.h"

void Rotation::rotate_around_center(
	// in 
	const std::vector<Pixel2>& P,
	float angle_deg,
	// out
	std::vector<Pixel2>& P_rot)
{
	P_rot.clear();

	// Find the center
	double cx = 0, cy = 0;
	for (auto& p : P)
	{
		cx += p.x;
		cy += p.y;
	}
	cx /= double(P.size());
	cy /= double(P.size());

	// Rotate
	float theta = angle_deg * float(M_PI) / 180.f;	// Angle in radians
	for (auto& p : P)
	{
		// Physics coordinate system
		//	x_rot = ((x - cx) * cos(theta)) - ((y - cy) * sin(theta)) + cx;
		//	y_rot = ((x - cx) * sin(theta)) + ((y - cy) * cos(theta)) + cy;
		
		// Screen coordinate system
		double x_rot = ((p.x - cx) * cos(theta)) - ((cy - p.y) * sin(theta)) + cx;
		double y_rot = cy - ((cy - p.y) * cos(theta)) + ((p.x - cx) * sin(theta));

		Pixel2 p_rot ((float)x_rot, (float)y_rot, p.inten);

		P_rot.push_back (p_rot);
	}
}

void Rotation::rotate_cloud (
	// in 
	const std::vector<Pixel2>& P,
	const double cx, 
	const double cy,
	float theta,
	// out
	std::vector<Pixel2>& P_rot)
{
	P_rot.clear();

	for (auto& p : P)
	{
		// Physics coordinate system:
		//		x_rot = ((x - cx) * cos(theta)) - ((y - cy) * sin(theta)) + cx;
		//		y_rot = ((x - cx) * sin(theta)) + ((y - cy) * cos(theta)) + cy;

		// Screen coordinate system:
		double x_rot = ((p.x - cx) * cos(theta)) - ((cy - p.y) * sin(theta)) + cx;
		double y_rot = cy - ((cy - p.y) * cos(theta)) + ((p.x - cx) * sin(theta));
		Pixel2 p_rot ((int)x_rot, (int)y_rot, p.inten);
		P_rot.push_back (p_rot);
	}
}

void Rotation::rotate_cloud(
	// [in]
	const OutOfRamPixelCloud& cloud,
	const double cx,
	const double cy,
	float theta_radians,
	// [out]
	OutOfRamPixelCloud& rotated_cloud,
	AABB& rotated_aabb)
{
	for (size_t i=0; i<cloud.get_size(); i++)
	{
		const Pixel2 p = cloud.get_at(i);

		// Background:
		//		x_rot = ((x - cx) * cos(theta)) - ((y - cy) * sin(theta)) + cx;
		//		y_rot = ((x - cx) * sin(theta)) + ((y - cy) * cos(theta)) + cy;

		// Screen coordinate system:
		double x_rot = ((p.x - cx) * cos(theta_radians)) - ((cy - p.y) * sin(theta_radians)) + cx;
		double y_rot = cy - ((cy - p.y) * cos(theta_radians)) + ((p.x - cx) * sin(theta_radians));

		Pixel2 p_rot((int)x_rot, (int)y_rot, p.inten);

		rotated_cloud.add_pixel (p_rot);
	}
}

