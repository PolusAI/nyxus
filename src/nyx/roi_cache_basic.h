#pragma once
#include <string>
#include "features/aabb.h"
#include "features/pixel.h"

class BasicLR
{
public:
	BasicLR (int lab) : label(lab)
	{
		// use default label '-1' and slide index '-1' (no slide available)
		slide_idx = -1;
	}
	void init_aabb (StatsInt x, StatsInt y);
	void update_aabb (StatsInt x, StatsInt y);
	void init_aabb_3D (StatsInt x, StatsInt y, StatsInt z);
	void update_aabb_3D (StatsInt x, StatsInt y, StatsInt z);
	void make_nonanisotropic_aabb() { aabb = ph_aabb; }
	void make_anisotropic_aabb(double ax, double ay, double az = 1.0) 
	{ 
		aabb = ph_aabb;
		aabb.apply_anisotropy (ax, ay, az);
	}

	AABB ph_aabb;
	AABB aabb;
	int label;

	// index in the dataset properties container, that is a vector of slide properties, linking a ROI to its slide
	int slide_idx;

	// The size of a voxel along each axis, as Nyxus::resolve_anisotropy resolves it for the ROI's
	// slide; (1,1,1) on an isotropic grid. A volumetric ROI's voxels are cached on the grid they were
	// acquired on, so the intensity and texture families see every voxel exactly once, and the shape
	// family scales its coordinates by this spacing to report physical geometry.
	double spacing_x = 1.0,
		spacing_y = 1.0,
		spacing_z = 1.0;
	void set_spacing (double sx, double sy, double sz) { spacing_x = sx; spacing_y = sy; spacing_z = sz; }

};

