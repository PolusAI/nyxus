#pragma once

#include <cfloat>
#include <string>
#include <vector>
#include "pixel.h"
#include "aabb.h"
#include "moments.h"
#include "../helpers/helpers.h"

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
		this->resize (W*H, 0);
	}

	SimpleMatrix() {}

	void allocate(int _w, int _h, T inival=0)
	{
		W = _w;
		H = _h;
		this->resize (W*H, inival);
	}
	// X,Y operator
	T& operator() (int x, int y)
	{
		if (x >= W || y >= H)
		{
			throw "subscript out of bounds";
		}
		return this->at(W * y + x);
	}
	// X,y operator
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

	bool safe(int x, int y) const
	{
		if (x >= W || y >= H)
			return false;
		else
			return true;
	}

	void fill (T val)
	{
		auto n = W * H;
		for (int i = 0; i < n; i++)
			this->at(i) = val;
	}

	int width() const { return W; }
	int height() const { return H; }

	void print (const std::string& head, const std::string& tail);

private:
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

private:
	int W, H;
};

typedef const pixData & readOnlyPixels;
typedef pixData & writeablePixels;

class ImageMatrix
{
public:
	ImageMatrix(): _pix_plane(0,0) {}

	ImageMatrix(const ImageMatrix & I): _pix_plane(0,0) 
	{
		this->allocate(I.width, I.height);
		this->_pix_plane = I._pix_plane;
	}

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

	ImageMatrix(const std::vector <Pixel2>& labels_raw_pixels):
		original_aabb(labels_raw_pixels), 
		_pix_plane(original_aabb.get_width(), original_aabb.get_height())
	{

		// Dimensions
		width = original_aabb.get_width();
		height = original_aabb.get_height();
		auto n = height * width;

		// Zero the matrix
		_pix_plane.reserve(n);
		for (auto i = 0; i < n; i++)
			_pix_plane.push_back(0);

		// Read pixels
		for (auto& pxl : labels_raw_pixels)
		{
			auto x = pxl.x - original_aabb.get_xmin(),
				y = pxl.y - original_aabb.get_ymin();
			_pix_plane[y * width + x] = pxl.inten;
		}
	}

	void use_roi (const std::vector <Pixel2>& labels_raw_pixels, const AABB& aabb)
	{
		original_aabb = aabb;

		// Dimensions
		width = original_aabb.get_width();
		height = original_aabb.get_height();

		// Zero the matrix
		_pix_plane.resize (width, height, 0);

		// Read pixels
		auto xmin = original_aabb.get_xmin(),
			ymin = original_aabb.get_ymin();
		for (auto& pxl : labels_raw_pixels)
		{
			auto x = pxl.x - xmin,
				y = pxl.y - ymin;
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

	void erode();

	// Based on X.Shu, Q.Zhang, J.Shi and Y.Qi - "A Comparative Study on Weighted Central Moment and Its Application in 2D Shape Retrieval" (2016) https://pdfs.semanticscholar.org/8927/2bef7ba9496c59081ae102925ebc0134bceb.pdf
	void apply_distance_to_contour_weights(const std::vector<Pixel2>& raw_pixels, const std::vector<Pixel2>& contour_pixels);

	// Returns chord length at x
	int get_chlen(int col);

	// Support of fractal dimension calculation
	bool tile_contains_signal (int tile_row, int tile_col, int tile_side);

	// min, max, mean, std computed in single pass, median in separate pass
	Moments2 stats;


	StatsInt height = 0, width = 0;
	AABB original_aabb;
	
	//std::vector<PixIntens> _pix_plane;	// [ height * width ]
	pixData _pix_plane;

	// hilight_x|y = -1 means no gilight
	using PrintablePoint = std::tuple<int, int, std::string>;
	void print(const std::string& head = "", const std::string& tail = "", std::vector<PrintablePoint> special_points = {});
};

class Power2PaddedImageMatrix : public ImageMatrix
{
public:
	Power2PaddedImageMatrix(const std::vector <Pixel2>& labels_raw_pixels, const AABB& aabb):
		ImageMatrix ()
	{
		original_aabb = aabb;

		int bigSide = std::max(aabb.get_width(), aabb.get_height());
		StatsInt paddedSide = closest_pow2 (bigSide);
		allocate (paddedSide, paddedSide);

		int padOffsetX = (paddedSide - original_aabb.get_width()) / 2;
		int padOffsetY = (paddedSide - original_aabb.get_height()) / 2;

		// Read pixels
		for (auto& pxl : labels_raw_pixels)
		{
			auto x = pxl.x - original_aabb.get_xmin() + padOffsetX,
				y = pxl.y - original_aabb.get_ymin() + padOffsetY;
			_pix_plane[y * width + x] = pxl.inten;
		}
	}
};
