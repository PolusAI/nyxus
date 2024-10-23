#pragma once

#include <vector>
#include "pixel.h"

/// @brief Generic pixel matrix class
/// @tparam T - pixel intensity class (int, uint, float, etc)
template <class T>
class SimpleCube : public std::vector<T>
{
public:
	SimpleCube(int _w, int _h, int _d) : W(_w), H(_h), D(_d)
	{
		this->resize (W * H * D, 0);
	}

	SimpleCube() {}

	// Updates matrice's width and geight from the parapeters
	void allocate(int _w, int _h, int _d)
	{
		W = _w;
		H = _h;
		D = _d;
		this->resize (size_t(W) * size_t(H) * size_t(D));
	}

	void calculate_from_pixelcloud (const std::vector <Pixel3> & pixelcloud, const AABB & aabb)
	{
		// Dimensions
		W = aabb.get_width();
		H = aabb.get_height();
		D = aabb.get_z_depth();

		// Zero the matrix
		allocate (W, H, D);
		std::fill (this->begin(), this->end(), 0);

		// Read pixels
		auto xmin = aabb.get_xmin(),
			ymin = aabb.get_ymin(),
			zmin = aabb.get_zmin();

		for (auto& pxl : pixelcloud)
		{
			auto x = pxl.x - xmin,
				y = pxl.y - ymin,
				z = pxl.z - zmin;
			this->xyz (x, y, z) = pxl.inten;
		}
	}

	// Returns size in bytes
	size_t szb()
	{
		return sizeof(T) * W * H * D;
	}

	// = W*H*z + W*y + x
	inline T& xyz (int x, int y, int z)
	{
#ifdef NYX_CHECK_BUFFER_BOUNDS
		if (x >= W || y >= H)
		{
			std::string msg = "index (" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ") is out of range (" + 
				std::to_string(W) + "," + std::to_string(H) + std::to_string(D) + ") at " + __FILE__ ":" + std::to_string(__LINE__);
#ifdef WITH_PYTHON_H
			throw std::out_of_range(msg.c_str());
#else
			std::cerr << "\nError: " << msg << '\n';
			std::exit(1);
#endif
		}
#endif

		return this->at (W*H*z + W*y + x);
	}

	// = W*H*z + W*y + x
	inline T xyz (int x, int y, int z) const
	{
#ifdef NYX_CHECK_BUFFER_BOUNDS
		if (x >= W || y >= H)
		{
			std::string msg = "index (" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ") is out of range (" +
				std::to_string(W) + "," + std::to_string(H) + std::to_string(D) + ") at " + __FILE__ ":" + std::to_string(__LINE__);
			throw std::out_of_range(msg.c_str());
			return -1;	// invalid intensity
		}
#endif

		T val = this->at (W*H*z + W*y + x);
		return val;
	}

	inline T zyx (int z, int y, int x) const
	{
		return xyz (x, y, z);
	}

	inline T& zyx (int z, int y, int x)
	{
		return xyz (x, y, z);
	}

	// 1-based coordinates
	inline T matlab (int z, int y, int x) const
	{
		return xyz (x-1, y-1, z-1);
	}

	bool safe (int z, int y, int x) const
	{
		if (x<0 || x >= W || y < 0 || y >= H || z < 0 || z >= D)
			return false;
		else
			return true;
	}

	void fill (T val)
	{
		std::fill (this->begin(), this->end(), val);
	}

	int width() const { return W; }
	int height() const { return H; }
	int depth() const { return D; }

private:
	int W = 0, H = 0, D = 0;
};

