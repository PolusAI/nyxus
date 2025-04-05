#include "roi_cache_basic.h"

void BasicLR::init_aabb(StatsInt x, StatsInt y)
{
	ph_aabb.init_x(x);
	ph_aabb.init_y(y);
}

void BasicLR::update_aabb(StatsInt x, StatsInt y)
{
	ph_aabb.update_x(x);
	ph_aabb.update_y(y);
}

void BasicLR::init_aabb_3D (StatsInt x, StatsInt y, StatsInt z)
{
	ph_aabb.init_x(x);
	ph_aabb.init_y(y);
	ph_aabb.init_z(z);
}

void BasicLR::update_aabb_3D (StatsInt x, StatsInt y, StatsInt z)
{
	ph_aabb.update_x(x);
	ph_aabb.update_y(y);
	ph_aabb.update_z(z);
}