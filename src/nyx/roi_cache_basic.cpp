#include "roi_cache_basic.h"

void BasicLR::init_aabb(StatsInt x, StatsInt y, StatsInt z)
{
	aabb.init_x(x);
	aabb.init_y(y);
	aabb.init_z(z);
}

void BasicLR::update_aabb(StatsInt x, StatsInt y, StatsInt z)
{
	aabb.update_x(x);
	aabb.update_y(y);
	aabb.update_z(z);
}

