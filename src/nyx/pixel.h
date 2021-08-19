#pragma once

using PixIntens = unsigned int;
using StatsInt = long;
using StatsReal = double;

template <typename T>
struct Point2
{
	T x, y;
	Point2(const T x_, const T y_) : x(x_), y(y_) {}
	Point2() : x(0), y(0) {}

	double normL2() const { return sqrt(x * x + y * y); }

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
inline double normL2(const Point2f& p) { return p.normL2(); }

struct Pixel2 : public Point2i
{
	PixIntens inten;
	Pixel2(StatsInt x_, StatsInt y_, PixIntens i_) : Point2(x_, y_), inten(i_) {}

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
};

