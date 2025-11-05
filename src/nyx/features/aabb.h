#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>
#include "pixel.h"

/// @brief Class encapsulating ROI axis aligned bounding box
class AABB
{
public:
	AABB() {}
	AABB(const std::vector<Pixel2> & cloud) 
	{
		for (auto& px : cloud)
		{
			update_x(px.x);
			update_y(px.y);
		}
	}
	void init_x(StatsInt x) { xmin = xmax = x; }
	void init_y(StatsInt y) { ymin = ymax = y; }
	void init_z(StatsInt z) { zmin = zmax = z; }
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
	void update_z(StatsInt z)
	{
		zmin = std::min(zmin, z);
		zmax = std::max(zmax, z);
	}
	inline StatsInt get_height() const { return ymax - ymin + 1; }
	inline StatsInt get_width() const { return xmax - xmin + 1; }
	inline StatsInt get_z_depth() const { return zmax - zmin + 1; }

	inline StatsInt get_area() const { return get_width() * get_height(); }
	inline StatsInt get_xmin() const { return xmin; }
	inline StatsInt get_xmax() const { return xmax; }
	inline StatsInt get_ymin() const { return ymin; }
	inline StatsInt get_ymax() const { return ymax; }
	inline StatsInt get_zmin() const { return zmin; }
	inline StatsInt get_zmax() const { return zmax; }

	void init_from_wh (StatsInt w, StatsInt h)
	{
		xmin = 0;
		xmax = w;
		ymin = 0;
		ymax = h;
	}

	void init_from_whd (StatsInt w, StatsInt h, StatsInt d)
	{
		xmin = 0;
		xmax = w;
		ymin = 0;
		ymax = h;
		zmin = 0;
		zmax = d;
	}

	static std::tuple<StatsInt, StatsInt, StatsInt, StatsInt> from_pixelcloud (const std::vector<Pixel2>& P)
	{
		AABB bb;
		for (auto& p : P)
		{
			bb.update_x(p.x);
			bb.update_y(p.y);
		}
		return {bb.get_xmin(), bb.get_ymin(), bb.get_xmax(), bb.get_ymax()};
	}

	void update_from_voxelcloud (const std::vector<Pixel3> & V)
	{
		auto cmpX = [](const Pixel3& p1, const Pixel3& p2) {return p1.x < p2.x; };
		StatsInt minx = (*std::min_element(V.begin(), V.end(), cmpX)).x;
		StatsInt maxx = (*std::max_element(V.begin(), V.end(), cmpX)).x;

		auto cmpY = [](const Pixel3& p1, const Pixel3& p2) {return p1.y < p2.y; };
		StatsInt miny = (*std::min_element(V.begin(), V.end(), cmpY)).y;
		StatsInt maxy = (*std::max_element(V.begin(), V.end(), cmpY)).y;

		auto cmpZ = [](const Pixel3& p1, const Pixel3& p2) {return p1.z < p2.z; };
		StatsInt minz = (*std::min_element(V.begin(), V.end(), cmpZ)).z;
		StatsInt maxz = (*std::max_element(V.begin(), V.end(), cmpZ)).z;

		this->xmin = minx;
		this->xmax = maxx;

		this->ymin = miny;
		this->ymax = maxy;

		this->zmin = minz;
		this->zmax = maxz;
	}

	inline bool contains(const AABB& other)
	{
		bool retval = get_xmin() <= other.get_xmin() &&
			get_xmax() >= other.get_xmax() &&
			get_ymin() <= other.get_ymin() &&
			get_ymax() >= other.get_ymax();
		return retval;
	}

	inline void apply_anisotropy (double ax, double ay, double az = 1.0)
	{
		xmin = StatsInt(xmin * ax);
		ymin = StatsInt(ymin * ay);
		zmin = StatsInt(zmin * az);
		
		auto orgMax = xmax;
		xmax = StatsInt(xmax * ax);
		if (StatsInt(double(xmax + 1) / ax) == orgMax)
			xmax = xmax+1;

		orgMax = ymax;
		ymax = StatsInt(ymax * ay);
		if (StatsInt(double(ymax + 1) / ay) == orgMax)
			ymax = ymax + 1;

		orgMax = zmax;
		zmax = StatsInt(zmax * az);
		if (StatsInt(double(zmax + 1) / az) == orgMax)
			zmax =zmax + 1;
	}

private:
	StatsInt xmin = INT32_MAX, 
		xmax = INT32_MIN, 
		ymin = INT32_MAX, 
		ymax = INT32_MIN, 
		zmin = INT32_MAX, 
		zmax = INT32_MIN;
};