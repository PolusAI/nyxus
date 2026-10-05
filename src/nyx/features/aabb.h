#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include "pixel.h"

// The out-of-line half of a check whose comparisons are meant to inline into a per-pixel loop
#if defined(_MSC_VER)
#define NYX_AABB_NOINLINE __declspec(noinline)
#else
#define NYX_AABB_NOINLINE __attribute__((noinline))
#endif

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

	// An empty cloud has no box, so it is refused rather than given one
	void update_from_voxelcloud (const std::vector<Pixel3> & V)
	{
		if (V.empty())
			throw std::invalid_argument ("AABB::update_from_voxelcloud: the voxel cloud is empty, so it has no bounding box");

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

	// The buffers sized from a box index each pixel by its offset from the box's low corner, so a
	// pixel outside the box would land in another row, or past the end of the buffer. These throw
	// std::out_of_range naming the pixel and the box instead. The comparisons stay inline in the
	// writers' loops; the message is built out of line, only when a pixel is refused.
	void require_contains (StatsInt x, StatsInt y) const
	{
		if (x < xmin || x > xmax || y < ymin || y > ymax)
			throw_outside (x, y);
	}

	void require_contains (StatsInt x, StatsInt y, StatsInt z) const
	{
		if (x < xmin || x > xmax || y < ymin || y > ymax || z < zmin || z > zmax)
			throw_outside (x, y, z);
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
	[[noreturn]] NYX_AABB_NOINLINE void throw_outside (StatsInt x, StatsInt y) const
	{
		throw std::out_of_range ("pixel (" + std::to_string(x) + "," + std::to_string(y)
			+ ") lies outside the bounding box x " + std::to_string(xmin) + ".." + std::to_string(xmax)
			+ ", y " + std::to_string(ymin) + ".." + std::to_string(ymax) + " that sizes its buffer");
	}

	[[noreturn]] NYX_AABB_NOINLINE void throw_outside (StatsInt x, StatsInt y, StatsInt z) const
	{
		throw std::out_of_range ("voxel (" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z)
			+ ") lies outside the bounding box x " + std::to_string(xmin) + ".." + std::to_string(xmax)
			+ ", y " + std::to_string(ymin) + ".." + std::to_string(ymax)
			+ ", z " + std::to_string(zmin) + ".." + std::to_string(zmax) + " that sizes its buffer");
	}

	StatsInt xmin = INT32_MAX,
		xmax = INT32_MIN, 
		ymin = INT32_MAX, 
		ymax = INT32_MIN, 
		zmin = INT32_MAX, 
		zmax = INT32_MIN;
};