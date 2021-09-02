#pragma once

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


//typedef std::vector<PixIntens> pixData;

class pixData : public std::vector<PixIntens>
{
public:
	pixData(int _w, int _h) : w(_w), h(_h) {}
	PixIntens & operator() (int x, int y)
	{
	if (x >= w || y >= h)
		throw "subscript out of bounds";
	return this->at (w * y + x);	
	}
	PixIntens operator() (int y, int x) const
	{
		if (x >= w || y >= h)
			throw "subscript out of bounds";
		PixIntens val = this->at(w * y + x);
		return val;
	}
protected:
	int w, h;
};

typedef const pixData& readOnlyPixels;


class ImageMatrix
{
public:

	ImageMatrix(const std::vector <Pixel2>& labels_raw_pixels, AABB & aabb)
		:_pix_plane(aabb.get_width(), aabb.get_height())
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

	void GetStats(Moments2& moments2) const
	{
		// Feed all the image pixels into the Moments2 object:
		moments2.reset();
		for (auto intens : _pix_plane)
			moments2.add(intens);
	}

	inline readOnlyPixels ReadablePixels() const {
		return _pix_plane;
	}

	StatsInt height = 0, width = 0;
	
	//std::vector<PixIntens> _pix_plane;	// [ height * width ]
	pixData _pix_plane;
};
