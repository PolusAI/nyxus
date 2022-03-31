#pragma once

#include <cmath>
#include <vector>

using PixIntens = unsigned int;
using StatsInt = long;
using StatsReal = double;

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
	Pixel2 (size_t x_, size_t y_, double i_) : Point2(x_, y_), inten(i_) {}

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
	operator Point2f () const { Point2f p((float)this->x, (float)this->y); return p; }

	double sqdist(int x, int y) const
	{
		double dx = (double)x - double(this->x),
			dy = (double)y - double(this->y);
		double retval = dx * dx + dy * dy;
		return retval;
	}

	double sqdist(const Pixel2 & px) const
	{
		double retval = sqdist(px.x, px.y);
		return retval;
	}

	double sum_sqdist(const std::vector<Pixel2>& cloud) const
	{
		double retval = 0.0;
		for (auto& px : cloud)
		{
			double sqd = this->sqdist(px);
			retval += sqd;
		}
		return retval;
	}

	double sqdist_to_segment (const Pixel2 & p1, const Pixel2 & p2) const
	{
		double x21 = p2.x - p1.x,
			y21 = p2.y - p1.y;
		double retval = (x21 * (p1.y-this->y) - (p1.x-this->x) * y21) / std::sqrt(x21*x21 + y21*y21);
		return std::abs(retval);
	}

	bool colocating(const Pixel2& other) const
	{
		return this->x == other.x && this->y == other.y;
	}

	bool belongs_to(const std::vector<Pixel2> & cloud) const
	{
		for (auto& px : cloud)
			if (this->colocating(px) && this->inten == px.inten)
				return true;
		return false;
	}

	// Returns an index in argument 'cloud'
	static int find_center (const std::vector<Pixel2> & cloud, const std::vector<Pixel2> & contour)
	{
		int idxMinDif = 0;
		auto minmaxDist = cloud[idxMinDif].min_max_sqdist(contour);
		double minDif = minmaxDist.second - minmaxDist.first;
		for (int i = 1; i < cloud.size(); i++)
		{
			//double dist = cloud[i].sum_sqdist(contour);
			auto minmaxDist = cloud[i].min_max_sqdist(contour);
			double dif = minmaxDist.second - minmaxDist.first;
			if (dif < minDif)
			{
				minDif = dif;
				idxMinDif = i;
			}
		}
		return idxMinDif;
	}

	std::pair<double, double> min_max_sqdist (const std::vector<Pixel2>& cloud) const
	{
		auto mind = sqdist(cloud[0]), 
			maxd = mind;

		for (int i = 1; i < cloud.size(); i++)
		{
			auto dist = sqdist(cloud[i]);
			if (dist < mind)
				mind = dist;
			if (dist > maxd)
				maxd = dist;
		}
		return { mind, maxd };
	}

	double min_sqdist (const std::vector<Pixel2>& cloud) const
	{
		auto mind = sqdist (cloud[0]);

		for (int i = 1; i < cloud.size(); i++)
		{
			auto dist = sqdist(cloud[i]);
			if (dist < mind)
				mind = dist;
		}
		return mind;
	}


	double angle(const Pixel2& other) const
	{
		double dotProd = double(this->x * other.x) + double(this->y * other.y),
			magThis = std::sqrt(this->x * this->x + this->y * this->y),
			magOther = std::sqrt(other.x * other.x + other.y * other.y),
			cosVal = dotProd / (magThis * magOther),
			ang = std::acos(cosVal);
		return ang;
	}
};

