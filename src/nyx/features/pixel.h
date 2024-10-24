#pragma once

#include <cmath>
#include <vector>

using PixIntens = unsigned int;
using RealPixIntens = float;
using StatsInt = long;
using StatsReal = double;
using gpureal = double;


/// @brief Geneal purpose class encapsulating a 2D Cartesian vector
/// @tparam T 
template <typename T>
struct Point2
{
	T x, y;
	Point2(T x_, T y_) : x(x_), y(y_) {}
	Point2() : x(0), y(0) {}

	float normL2() const { return sqrt(x * x + y * y); }

	Point2 operator - ()
	{
		Point2 p2(-(this->x), -(this->y));
		return p2;
	}
	Point2 operator - (const Point2& v)
	{
		Point2 p2(this->x - v.x, this->y - v.y);
		return p2;
	}
	Point2 operator + (const Point2& v)
	{
		Point2 p2(this->x + v.x, this->y + v.y);
		return p2;
	}
	Point2 operator / (float k)
	{
		Point2 p2(this->x / k, this->y / k);
		return p2;
	}
};

using Point2i = Point2<StatsInt>;
using Point2f = Point2<float>;
inline float normL2(const Point2f& p) { return p.normL2(); }

/// @brief Class encapsulating a 2D pixel
struct Pixel2 : public Point2i
{
	PixIntens inten;

	Pixel2() : Point2(0, 0), inten(0) {}
	Pixel2 (StatsInt x_, StatsInt y_, PixIntens i_) : Point2(x_, y_), inten(i_) {}
	Pixel2 (int x_, int y_, PixIntens i_) : Point2(x_, y_), inten(i_) {}
	Pixel2 (float x_, float y_, PixIntens i_) : Point2((StatsInt)x_, (StatsInt)y_), inten(i_) {}
	Pixel2 (size_t x_, size_t y_, double i_) : Point2((StatsInt)x_, (StatsInt)y_), inten((PixIntens)i_) {}
	Pixel2 (size_t x_, size_t y_, PixIntens i_) : Point2((StatsInt)x_, (StatsInt)y_), inten(i_) {}

	bool operator == (const Pixel2& p2)
	{
		return this->x == p2.x && this->y == p2.y;
	}
	Pixel2 operator - ()
	{
		Pixel2 p2(-(this->x), -(this->y), this->inten);
		return p2;
	}
	Pixel2 operator - (const Pixel2& v) const
	{
		Pixel2 p2(this->x - v.x, this->y - v.y, this->inten);
		return p2;
	}
	Pixel2 operator + (const Pixel2& v) const
	{
		Pixel2 p2(this->x + v.x, this->y + v.y, this->inten);
		return p2;
	}
	Pixel2 operator / (float k) const
	{
		Pixel2 p2(StatsInt(this->x / k), StatsInt(this->y / k), this->inten);
		return p2;
	}
	Pixel2 operator * (float k) const
	{
		Pixel2 p2(StatsInt(this->x * k), StatsInt(this->y * k), this->inten);
		return p2;
	}
	operator Point2f () const 
	{ 
		Point2f p((float)this->x, (float)this->y); 
		return p; 
	}

	// vertically or horizontally aligned
	bool aligned (const Pixel2 & p) const
	{
		return this->x == p.x || this->y == p.y;
	}

	/// @brief Returns squared distance between 'x' and 'y'
	double sqdist(int x, int y) const;

	/// @brief Returns the squared distance between this pixel and 'px'
	double sqdist(const Pixel2& px) const;

	/// @brief Returns the sum of squared distances from this pixel to each pixel of 'cloud'
	double sum_sqdist(const std::vector<Pixel2>& cloud) const;

	/// @brief Returns the squared distance from this pixel to segment 'p1,p2'
	double sqdist_to_segment(const Pixel2& p1, const Pixel2& p2) const;

	/// @brief Returns the distance from this pixel to segment 'p1,p2'
		double dist_to_segment (const Pixel2 & p1, const Pixel2 & p2) const;

	/// @brief Returns true if this and 'other' pixels share the same location
	bool colocating(const Pixel2& other) const
	{
		return this->x == other.x && this->y == other.y;
	}

	/// @brief Returns true if this pixel belongs to 'cloud'
	bool belongs_to(const std::vector<Pixel2>& cloud) const;

	/// @brief Returns an index in argument 'cloud'
	static int find_center(const std::vector<Pixel2>& cloud, const std::vector<Pixel2>& contour);

	std::pair<double, double> min_max_sqdist(const std::vector<Pixel2>& contour) const;

	/// @brief Returns the minimum squared distance squared distance from <this> pixel to the <cloud>
	double min_sqdist (const std::vector<Pixel2>& cloud) const;

	/// @brief Returns the maximum squared distance squared distance from <this> pixel to the <cloud>
	double max_sqdist (const std::vector<Pixel2>& cloud) const;

	/// @brief Returns the angle in radians between this pixel and 'other' relative to the origin 
	double angle(const Pixel2& other) const;

};

bool operator == (const Pixel2& p1, const Pixel2& p2);

template <typename T>
struct Point3
{
	T x, y, z;
	Point3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
	Point3() : x(0), y(0), z(0) {}

	float normL2() const { return sqrt(x*x + y*y + z*z); }

	Point3 operator - ()
	{
		Point3 p (-(this->x), -(this->y), -(this->z));
		return p;
	}
	Point3 operator - (const Point3 & v)
	{
		Point3 p (this->x - v.x, this->y - v.y, this->z - v.z);
		return p;
	}
	Point3 operator + (const Point3 & v)
	{
		Point3 p (this->x + v.x, this->y + v.y, this->z + v.z);
		return p;
	}
	Point3 operator / (float k)
	{
		Point3 p (this->x / k, this->y / k, this->z / k);
		return p;
	}
};

using Point3i = Point3<StatsInt>;

struct Pixel3 : public Point3i
{
	PixIntens inten;

	Pixel3() : Point3(0, 0, 0), inten(0) {}
	Pixel3(StatsInt x_, StatsInt y_, StatsInt z_, PixIntens i_) : Point3(x_, y_, z_), inten(i_) {}
	Pixel3(int x_, int y_, int z_, PixIntens i_) : Point3(x_, y_,z_), inten(i_) {}
	Pixel3(float x_, float y_, float z_, PixIntens i_) : Point3((StatsInt)x_, (StatsInt)y_, (StatsInt)z_), inten(i_) {}
	Pixel3(size_t x_, size_t y_, size_t z_, double i_) : Point3((StatsInt)x_, (StatsInt)y_, (StatsInt)z_), inten((PixIntens)i_) {}
	Pixel3(size_t x_, size_t y_, size_t z_, PixIntens i_) : Point3((StatsInt)x_, (StatsInt)y_, (StatsInt)z_), inten(i_) {}

	bool operator == (const Pixel3& p)
	{
		return this->x == p.x && this->y == p.y && this->z == p.z;
	}
	Pixel3 operator - ()
	{
		Pixel3 p(-(this->x), -(this->y), -(this->z), this->inten);
		return p;
	}
	Pixel3 operator - (const Pixel3& v) const
	{
		Pixel3 p(this->x - v.x, this->y - v.y, this->z - v.z, this->inten);
		return p;
	}
	Pixel3 operator + (const Pixel3& v) const
	{
		Pixel3 p(this->x + v.x, this->y + v.y, this->z + v.z, this->inten);
		return p;
	}
	Pixel3 operator / (float k) const
	{
		Pixel3 p(StatsInt(this->x / k), StatsInt(this->y / k), StatsInt(this->z / k), this->inten);
		return p;
	}
	Pixel3 operator * (float k) const
	{
		Pixel3 p(StatsInt(this->x * k), StatsInt(this->y * k), StatsInt(this->z * k), this->inten);
		return p;
	}

	/// @brief Returns squared distance between 'x' and 'y'
	double sqdist(int x, int y, int z) const;

	/// @brief Returns the squared distance between this pixel and 'px'
	double sqdist(const Pixel3& px) const;

	/// @brief Returns an index in argument 'cloud'
	static int find_center(const std::vector<Pixel3>& cloud, const std::vector<Pixel3>& contour);

	std::pair<double, double> min_max_sqdist(const std::vector<Pixel3>& contour) const;

	/// @brief Returns the minimum squared distance squared distance from <this> pixel to the <cloud>
	double min_sqdist(const std::vector<Pixel3>& cloud) const;

	/// @brief Returns the maximum squared distance squared distance from <this> pixel to the <cloud>
	double max_sqdist(const std::vector<Pixel3>& cloud) const;

	/// @brief Returns the angle in radians between this pixel and 'other' relative to the origin 
	double angle(const Pixel3& other) const;

};

