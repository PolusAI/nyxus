#pragma once
#include "features/aabb.h"
#include "features/pixel.h"

class BasicLR
{
public:
	void init_aabb(StatsInt x, StatsInt y);
	void update_aabb(StatsInt x, StatsInt y);

	AABB aabb;
	int label;

private:
};

