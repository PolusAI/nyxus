#pragma once

#include <cfloat>
#include <string>
#include <vector>
#include "pixel.h"
#include "aabb.h"

// Depends:
#define MIN_VAL -FLT_MAX
#define MAX_VAL FLT_MAX
//
class Moments2 {
private:
	double _min, _max, _mean, M2;
	size_t _n;

public:
	Moments2() { reset(); }
	void reset() { _mean = M2 = 0.0; _min = DBL_MAX; _max = -DBL_MAX; _n = 0; }
	inline double add(const double x) {
		size_t n1;
		double delta, delta_n, term1;
		if (std::isnan(x) || x > MAX_VAL || x < MIN_VAL) 
			return (x);

		n1 = _n;
		_n = _n + 1;
		delta = x - _mean;
		delta_n = delta / _n;
		term1 = delta * delta_n * n1;
		_mean = _mean + delta_n;
		M2 += term1;

		if (x > _max) _max = x;
		if (x < _min) _min = x;
		return (x);
	}

	size_t n()    const { return _n; }
	double min__()  const { return _min; }
	double max__()  const { return _max; }
	double mean() const { return _mean; }
	double std() const { return (_n > 2 ? sqrt(M2 / (_n - 1)) : 0.0); }
	double var() const { return (_n > 2 ? (M2 / (_n - 1)) : 0.0); }
	void momentVector(double* z) const { z[0] = mean(); z[1] = std(); }
};

// functor to call add on a reference using the () operator
// for example, using Eigen: ReadablePixels().unaryExpr (Moments4func(stats)).sum();
// N.B.: The sum() in the expression above is to force evaluation of all of the coefficients.
// The return value of Eigen's unaryExpr is a unaryExpr, which doesn't actually do anything until its assigned to something.
class Moments2func {
	Moments2& moments;
public:
	Moments2func(Moments2& in_moments) : moments(in_moments) { in_moments.reset(); }
	const double operator()(const double& x) const {
		return (moments.add(x));
	}
};

template <class T>
class SimpleMatrix : public std::vector<T>
{
public:
	SimpleMatrix(int _w, int _h) : W(_w), H(_h) 
	{ 
		this.resize (W*H, 0);
	}

	SimpleMatrix() {}

	void allocate(int _w, int _h)
	{
		W = _w;
		H = _h;
		this->resize (W*H, 0);
	}

	T& operator() (int x, int y)
	{
		if (x >= W || y >= H)
		{
			throw "subscript out of bounds";
		}
		return this->at(W * y + x);
	}
	T operator() (int x, int y) const
	{
		if (x >= W || y >= H)
		{
			throw "subscript out of bounds";
			return -1;	// Special value indicating invalid intensity
		}
		T val = this->at(W * y + x);
		return val;
	}

	// 1-based x and y
	T matlab (int y, int x) const
	{
		T t = operator() (x-1,y-1);
		return t;
	}

	bool safe(int x, int y)
	{
		if (x >= W || y >= H)
			return false;
		else
			return true;
	}

	int width() { return W; }
	int height() { return H; }

	void print (const std::string& head, const std::string& tail);

protected:
	int W = 0, H = 0;
};


class pixData : public std::vector<PixIntens>
{
public:
	pixData(int _w, int _h) : W(_w), H(_h) {}

	void resize (int width, int height, PixIntens val)
	{
		W = width;
		H = height;
		std::vector<PixIntens>::resize (width * height, val);
	}

	PixIntens & operator() (int y, int x)
	{
		if (x >= W || y >= H)
		{
			throw "subscript out of bounds";
		}
		return this->at(W * y + x);
	}
	PixIntens operator() (int y, int x) const
	{
		if (x >= W || y >= H)
		{
			throw "subscript out of bounds";
			return -1;	// Special value indicating invalid intensity
		}
		PixIntens val = this->at (W * y + x);
		return val;
	}
	
	bool safe (int y, int x) const
	{
		if (x < 0 || x >= W || y < 0 || y >= H)
			return false;
		else
			return true;
	}

	int width() const { return W; }
	int height() const { return H; }

protected:
	int W, H;
};

typedef const pixData & readOnlyPixels;
typedef pixData & writeablePixels;

class ImageMatrix
{
public:
	ImageMatrix(): _pix_plane(0,0) {}

	ImageMatrix(const std::vector <Pixel2>& labels_raw_pixels, AABB & aabb) :
		original_aabb (aabb),
		_pix_plane(aabb.get_width(), aabb.get_height())
	{
		// Dimensions
		width = aabb.get_width();
		height = aabb.get_height();
		auto n = height * width;

		// Zero the matrix
		_pix_plane.reserve(n);
		for (auto i = 0; i < n; i++)
			_pix_plane.push_back(0);

		// Read pixels
		for (auto& pxl : labels_raw_pixels)
		{
			auto x = pxl.x - aabb.get_xmin(),
				y = pxl.y - aabb.get_ymin();
			_pix_plane[y * width + x] = pxl.inten;
		}
	}

	void allocate(int w, int h)
	{
		width = w;
		height = h;
		_pix_plane.resize (width, height, 0);
	}

	void GetStats(Moments2& moments2) const
	{
		// Feed all the image pixels into the Moments2 object:
		moments2.reset();
		for (auto intens : _pix_plane)
			moments2.add(intens);
	}

	inline const pixData& /*readOnlyPixels*/ ReadablePixels() const
	{
		return _pix_plane;
	}

	inline pixData& WriteablePixels()
	{
		return _pix_plane;
	}

	PixIntens* writable_data_ptr() { return _pix_plane.data(); }

	// by default, based on computed min and max
	void histogram(double* bins, unsigned short nbins, bool imhist = false, const Moments2& in_stats = Moments2()) const; 

	// Otsu grey threshold
	double Otsu (bool dynamic_range = true) const;

	// min, max, mean, std computed in single pass, median in separate pass
	Moments2 stats;


	StatsInt height = 0, width = 0;
	AABB original_aabb;
	
	//std::vector<PixIntens> _pix_plane;	// [ height * width ]
	pixData _pix_plane;

	void print(const std::string& head = "", const std::string& tail = "");
};
