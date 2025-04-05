#pragma once
#include <string>
#include "features/aabb.h"
#include "features/pixel.h"

class BasicLR
{
public:
	BasicLR (int lab) : label(lab) {}
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
	std::string segFname, intFname;	// full paths

};

