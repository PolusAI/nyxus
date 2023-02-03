#pragma once

#include <string>
#include "aabb.h"
#include "pixel.h"
#include "../image_loader.h"

/// @brief Writeable out of memory pixel cloud
class OutOfRamPixelCloud
{
public:
	OutOfRamPixelCloud();
	OutOfRamPixelCloud(const OutOfRamPixelCloud&) = delete;	// Instead, use init() and copy()
	~OutOfRamPixelCloud();
	void init (unsigned int _roi_label, std::string name);
	void clear();
	void add_pixel (const Pixel2& p);
	void close();
	size_t size() const;
	Pixel2 get_at (size_t idx) const;
	Pixel2 operator[] (size_t idx) const { return get_at(idx); }

	struct iterator 
	{
		public:
			iterator (const OutOfRamPixelCloud& obj, std::size_t idx)
				: m_object(obj), 
				m_index(idx)
			{}

			Pixel2 operator * () const 
			{
				return m_object.get_at(m_index);
			}

			bool operator == (iterator const& it) const 
			{
				return (&m_object == &it.m_object) && (m_index == it.m_index);
			}

			bool operator != (iterator const& it) const 
			{
				return (&m_object != &it.m_object) || (m_index != it.m_index);
			}

			iterator& operator ++ () 
			{
				++m_index;
				return *this;
			}
		private:
			const OutOfRamPixelCloud& m_object;
			std::size_t m_index;
	};

	iterator begin() const
	{
		return iterator(*this, 0);
	}

	iterator end() const
	{
		return iterator(*this, size());
	}

private:
	size_t n_items = 0;
	std::string filepath;
	FILE* pF = nullptr;
	size_t item_size = sizeof(Pixel2::x) + sizeof(Pixel2::y) + sizeof(Pixel2::inten);
};

/// @brief Read-only out of memory pixel matrix browsable via ImageLoader
class OOR_ReadMatrix
{
public:
	OOR_ReadMatrix (ImageLoader& _imloader, const AABB& _aabb) : imloader(_imloader), aabb(_aabb) {}
	size_t get_width() const;
	size_t get_height() const;
	size_t get_size() const;
	bool safe(size_t x, size_t y) const;

	/// @brief Helps constructing a Pixel2 instance at index 'idx' in intensity matrix scenarios
	/// 
	/// Example:
	/// 	auto [y0, x0] = matrix.idx_2_rc(idx);
	/// 	double inten = matrix.get_at(imlo, idx);
	/// 	Pixel2 p0(x0, y0, inten);
	/// 
	/// @param idx 0-based pixel index
	/// @return 0-based row and column 
	std::tuple<size_t, size_t> idx_2_rc (size_t idx) const;

	double get_at (size_t row, size_t col) const;
	double get_at (size_t idx) const;

	// Normalization
	void apply_normalizing_range (double _minval, double _maxval, double _normalization_ceil) 
	{ 
		minval = _minval; 
		maxval = _maxval; 
		normalization_ceil = _normalization_ceil;
		scale = normalization_ceil * (maxval - minval);
	}
	double get_normed_at(size_t row, size_t col) const { return (get_at(row, col) - minval)* scale;  }
	double get_normed_at(size_t idx) const { return (get_at(idx) - minval) * scale; }

private:
	ImageLoader& imloader;
	AABB aabb;

	// Retrieving normalized elements
	double normalization_ceil = 255.0, minval = 0.0, maxval = 1.0, scale = 255.0;
};

/// @brief Readable out of RAM version of class ImageMatrix
class ReadImageMatrix_nontriv
{
public:
	ReadImageMatrix_nontriv (const AABB & aabb);
	double get_at (ImageLoader& imloader, size_t row, size_t col);
	double get_at (ImageLoader& imloader, size_t idx);
	size_t get_width() const;
	size_t get_height() const;
	size_t get_size() const;

	/// @brief Helps constructing a Pixel2 instance at index 'idx' in intensity matrix scenarios
	/// 
	/// Example:
	/// 	auto [y0, x0] = matrix.idx_2_rc(idx);
	/// 	double inten = matrix.get_at(imlo, idx);
	/// 	Pixel2 p0(x0, y0, inten);
	/// 
	/// @param idx 0-based pixel index
	/// @return 0-based row and column 
	std::tuple<size_t, size_t> idx_2_rc (size_t idx) const;

	bool safe(size_t x, size_t y) const;

private:
	AABB aabb;
};

/// @brief Writable out of RAM version of class ImageMatrix
class WriteImageMatrix_nontriv
{
public:
	WriteImageMatrix_nontriv (const std::string&  _name, unsigned int _roi_label);
	~WriteImageMatrix_nontriv();
	
	// Initialization
	void allocate (int w, int h, double ini_value);
	void allocate_from_cloud (const OutOfRamPixelCloud& cloud, const AABB& aabb, bool mask_image);
	void allocate_from_cloud_coarser_grayscale (const OutOfRamPixelCloud& cloud, const AABB& aabb, PixIntens min_inten, PixIntens inten_range, unsigned int n_grays);
	void copy (WriteImageMatrix_nontriv & other);
	void set_at(int row, int col, double val);
	void set_at(size_t idx,  double val);
	double yx (size_t row, size_t col);
	double get_at (size_t idx);
	double get_max();
	size_t size();
	size_t get_width();
	size_t get_height();
	size_t get_chlen(size_t x);
	bool safe(size_t y, size_t x) const;
	PixIntens operator[] (size_t idx) { return (PixIntens) get_at(idx); }
	void check_non_empty();
	std::string info();

protected:
	std::string name;
	std::string filepath;
	FILE* pF = nullptr;
	int width = 0, height = 0;
	size_t item_size = sizeof(double);
	AABB original_aabb;
};

/// @brief Padded image matrix
class Power2PaddedImageMatrix_NT : public WriteImageMatrix_nontriv
{
public:
	/// @brief Use base_level=0 and attenuation >0 and <1 e.g. 0.5 to build an imag of a specific intensity distribution. Or base_level=1 and attenuation 1 to build an image of the mask
	/// @param labels_raw_pixels ROI pixel cloud
	/// @param aabb ROI axis aligned bounding box
	/// @param base_level Set {0,1}
	/// @param attenuation Value in the interval (0,1]
	Power2PaddedImageMatrix_NT (const std::string& _name, unsigned int _roi_label, const OutOfRamPixelCloud& raw_pixels, const AABB& aabb, PixIntens base_level, double attenuation);

	// Support of erosion features
	bool tile_contains_signal (int tile_row, int tile_col, int tile_side);
};

