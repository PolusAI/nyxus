#pragma once

#include <utility>
#include "pixel.h"

class AABB
{
public:
	AABB() {}
	void init_x(StatsInt x) { xmin = xmax = x; }
	void init_y(StatsInt y) { ymin = ymax = y; }
	void update_x(StatsInt x)
	{
		xmin = std::min(xmin, x);
		xmax = std::max(xmax, x);
	}
	void update_y(StatsInt y)
	{
		ymin = std::min(ymin, y);
		ymax = std::max(ymax, y);
	}
	StatsInt get_height() { return ymax - ymin + 1; }
	StatsInt get_width() { return xmax - xmin + 1; }
	StatsInt get_area() { return get_width() * get_height(); }
	inline StatsInt get_xmin() { return xmin; }
	inline StatsInt get_xmax() { return xmax; }
	inline StatsInt get_ymin() { return ymin; }
	inline StatsInt get_ymax() { return ymax; }

protected:
	StatsInt xmin = INT32_MAX, xmax = INT32_MIN, ymin = INT32_MAX, ymax = INT32_MIN;
};