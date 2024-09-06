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
	float theta = angle_deg * float(M_PI) / 180.f;
	double s = sin(theta), 
		c = cos(theta);
	for (auto& p : P)
	{
		double x_rot = (p.x - cx) * c - (p.y - cy) * s + cx;
		double y_rot = (p.y - cy) * c + (p.x - cx) * s + cy;
		Pixel2 p_rot ((float)x_rot, (float)y_rot, p.inten);
		P_rot.push_back (p_rot);
	}
}

void Rotation::rotate_cloud (
	// in 
	const std::vector<Pixel2>& P,
	const double cx, 
	const double cy,
	float theta_rad,
	// out
	std::vector<Pixel2>& P_rot)
{
	P_rot.clear();

	double s = sin(theta_rad),
		c = cos(theta_rad);

	for (auto& p : P)
	{
		double x_rot = (p.x - cx) * c - (p.y - cy) * s + cx;
		double y_rot = (p.y - cy) * c + (p.x - cx) * s + cy;
		Pixel2 p_rot((float)x_rot, (float)y_rot, p.inten);
		P_rot.push_back(p_rot);
	}
}

void Rotation::rotate_cloud_NT (
	// [in]
	const OutOfRamPixelCloud& cloud,
	const double cx,
	const double cy,
	float theta_radians,
	// [out]
	OutOfRamPixelCloud& rotated_cloud)
{
	double s = sin(theta_radians),
		c = cos(theta_radians);

	for (auto p: cloud)
	{
		double x_rot = (p.x - cx) * c - (p.y - cy) * s + cx;
		double y_rot = (p.y - cy) * c + (p.x - cx) * s + cy;
		Pixel2 p_rot((float)x_rot, (float)y_rot, p.inten);
		rotated_cloud.add_pixel (p_rot);
	}
}

