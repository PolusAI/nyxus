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
	double operator()(const double& x) const 
	{
		return (moments.add(x));
	}
};

/// @brief Generic pixel matrix class
/// @tparam T - pixel intensity class (int, uint, float, etc)
template <class T>
class SimpleMatrix : public std::vector<T>
{
public:
	SimpleMatrix(int _w, int _h)
	{ 
		allocate(_w, _h, 0);
	}

	SimpleMatrix() {}

	void allocate(int _w, int _h, T inival=0)
	{
		W = _w;
		H = _h;
		this->resize (W*H, inival);
	}

	inline T& xy(int x, int y)
	{
		#ifdef NYX_CHECK_BUFFER_BOUNDS
		if (x >= W || y >= H)
		{
			std::string msg = "index ";
			msg += std::to_string(x) + ",";
			msg += std::to_string(y) + " is out of range ";
			msg += std::to_string(W) + ",";
			msg += std::to_string(H) + " at ";
			msg += __FILE__ ":" + std::to_string(__LINE__);
			throw std::out_of_range(msg.c_str());
		}
		#endif

		return this->operator[] (W * y + x);
	}

	inline T xy (int x, int y) const
	{
		#ifdef NYX_CHECK_BUFFER_BOUNDS
		if (x >= W || y >= H)
		{
			std::string msg = "index ";
			msg += std::to_string(x) + ",";
			msg += std::to_string(y) + " is out of range ";
			msg += std::to_string(W) + ",";
			msg += std::to_string(H) + " at ";
			msg += __FILE__ ":" + std::to_string(__LINE__);
			throw std::out_of_range(msg.c_str());
			return -1;	// Special value indicating invalid intensity
		}
		#endif

		T val = this->operator[] (W * y + x);
		return val;
	}

	// y - strided index, x - nonstrided; 1-based x and y
	T matlab (int y, int x) const
	{
		T t = xy(x-1,y-1);		//--formerly--> operator() (x-1,y-1);
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
			this->operator[](i) = val;
	}

	int width() const { return W; }
	int height() const { return H; }

	void print(const std::string& head, const std::string& tail);

private:
	int W = 0, H = 0;
};

/// @brief Pixel plane of image matrices
class pixData : public std::vector<PixIntens>
{
public:
	pixData(int _w, int _h) 
	{
		set_dimensions (_w, _h);
	}

	void set_dimensions (int _w, int _h)
	{
		W = _w;
		H = _h;
	}

	/// @brief Allocates and initializes
	void allocate_and_initialize (int width, int height, PixIntens val)
	{
		set_dimensions(width, height);
		std::vector<PixIntens>::resize (width * height, val);
	}

	/// @brief Only initializes the image matrix buffer but does not allocate it. 
	void initialize_without_allocation (int width, int height, PixIntens val)
	{
		set_dimensions(width, height);	// sets W and H
		for (size_t n=W*H, i=0; i<n; i++)
			this->operator[](i) = val;
	}
	
	// = W * y + x
	inline PixIntens & yx /*operator()*/ (int y, int x)
	{
		#ifdef NYX_CHECK_BUFFER_BOUNDS
		if (W>0 || H>0)
		{
			throw std::out_of_range("pixData::W or H is equal to 0");
		}
		if (x >= W || y >= H)
		{
			std::string msg = "index ";
			msg += std::to_string(x) + ",";
			msg += std::to_string(y) + " is out of range ";
			msg += std::to_string(W) + ",";
			msg += std::to_string(H) + " at ";
			msg += __FILE__ ":" + std::to_string(__LINE__);
			throw std::out_of_range(msg.c_str());
		}
		#endif

		return this->operator[] (W * y + x);
	}
	// = W * y + x
	inline PixIntens yx /*operator()*/ (int y, int x) const
	{
		#ifdef NYX_CHECK_BUFFER_BOUNDS
		if (W > 0 || H > 0)
		{
			throw std::out_of_range("pixData::W or H is equal to 0");
		}		
		if (x >= W || y >= H)
		{
			std::string msg = "index ";
			msg += std::to_string(x) + ",";
			msg += std::to_string(y) + " is out of range ";
			msg += std::to_string(W) + ",";
			msg += std::to_string(H) + " at ";
			msg += __FILE__ ":" + std::to_string(__LINE__);
			throw std::out_of_range(msg.c_str());
			return -1;	// Special value indicating invalid intensity
		}
		#endif

		PixIntens val = this->operator[] (W * y + x);
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

/// @brief General purpose matrix of single-channel pixel intensities
class ImageMatrix
{
public:
	ImageMatrix(): _pix_plane(0,0) {}

	ImageMatrix(const ImageMatrix & I, const AABB& aabb): 
		original_aabb(aabb),
		_pix_plane(0,0)
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

	void bind_to_buffer (PixIntens* startitem_ptr, PixIntens* enditem_ptr, int _width, int _height)
	{
		_pix_plane.assign (startitem_ptr, enditem_ptr);
		_pix_plane.set_dimensions (_width, _height);
		this->width = _width;
		this->height = _height;
	}

	void calculate_from_pixelcloud (const std::vector <Pixel2>& labels_raw_pixels, const AABB& aabb)
	{
		original_aabb = aabb;

		// Dimensions
		width = original_aabb.get_width();
		height = original_aabb.get_height();

		// Zero the matrix
		_pix_plane.initialize_without_allocation (width, height, 0);

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
		_pix_plane.allocate_and_initialize (width, height, 0);
	}

	void clear()
	{
		_pix_plane.clear();
		width = height = 0;
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

	/// Diagnostic methods not meant for performance
	using PrintablePoint = std::tuple<int, int, std::string>;
	void print(const std::string& head = "", const std::string& tail = "", std::vector<PrintablePoint> special_points = {});
	void print(std::ofstream& f, const std::string& head = "", const std::string& tail = "", std::vector<PrintablePoint> special_points = {});
};

/// @brief Padded image matrix
class Power2PaddedImageMatrix : public ImageMatrix
{
public:
	Power2PaddedImageMatrix(const std::vector <Pixel2>& labels_raw_pixels, const AABB& aabb):
		ImageMatrix ()
	{
		original_aabb = aabb;

		int bigSide = std::max(aabb.get_width(), aabb.get_height());
		StatsInt paddedSide = Nyxus::closest_pow2 (bigSide);
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
