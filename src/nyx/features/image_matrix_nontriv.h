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
	void init (unsigned int _roi_label, std::string name);
	void clear();
	void add_pixel (const Pixel2& p);
	size_t get_size() const;
	Pixel2 get_at (size_t idx) const;

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
	void allocate (int w, int h=1, double ini_value=0.0);
	void init_with_cloud (const OutOfRamPixelCloud& cloud, const AABB& aabb);
	void init_with_cloud_distance_to_contour_weights (const OutOfRamPixelCloud& cloud, const AABB& aabb, std::vector<Pixel2>& contour);
	void copy(WriteImageMatrix_nontriv & other);

	void set_at(int row, int col, double val);
	void set_at(size_t idx,  double val);
	double get_at(int row, int col);
	double get_at(size_t idx);
	double get_max();
	size_t size();
	size_t get_width();
	size_t get_height();
	size_t get_chlen(size_t x);
	bool safe(size_t x, size_t y) const;

private:
	std::string filepath;
	FILE* pF = nullptr;
	int width, height;
	size_t item_size = sizeof(double);
	AABB original_aabb;
};


