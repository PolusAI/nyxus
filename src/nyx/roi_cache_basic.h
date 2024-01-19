#pragma once
#include <string>
#include "features/aabb.h"
#include "features/pixel.h"

class BasicLR
{
public:
	void init_aabb(StatsInt x, StatsInt y);
	void update_aabb(StatsInt x, StatsInt y);
	void init_aabb_3D (StatsInt x, StatsInt y, StatsInt z);
	void update_aabb_3D (StatsInt x, StatsInt y, StatsInt z);

	AABB aabb;
	int label;
	std::string segFname, intFname;	// Full paths

private:
};

