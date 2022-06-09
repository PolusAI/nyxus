#include "roi_cache_basic.h"

void BasicLR::init_aabb(StatsInt x, StatsInt y)
{
	aabb.init_x(x);
	aabb.init_y(y);
}

void BasicLR::update_aabb(StatsInt x, StatsInt y)
{
	aabb.update_x(x);
	aabb.update_y(y);
}

