#include <errno.h>
#include <iostream>
#include <sstream>
#include <thread>
#include "../environment.h"
#include "../helpers/fsystem.h"
#include "../helpers/helpers.h"
#include "image_matrix_nontriv.h"

OutOfRamPixelCloud::OutOfRamPixelCloud() {}


OutOfRamPixelCloud::~OutOfRamPixelCloud()
{
	if (pF)
	{
		fclose(pF);
		pF = nullptr;
		fs::remove(filepath);
	}
}

void OutOfRamPixelCloud::check_io_ok() const
{
	if (!pF)
		throw (std::runtime_error("ERROR in add_pixel() - file might not be open with init()"));
}

void OutOfRamPixelCloud::init (unsigned int _roi_label, std::string name)
{
	n_items = 0;	
	
	auto tid = std::this_thread::get_id();

	std::stringstream ssPath;
	ssPath << Nyxus::theEnvironment.get_temp_dir_path() << name << "_roi" << _roi_label << "_tid" << tid << ".vec.nyxus";
	filepath = ssPath.str();
	
	errno = 0;	
	pF = fopen(filepath.c_str(), "w+b");
	if (!pF)
		throw (std::runtime_error("Error creating file " + filepath + " Error code:" + std::to_string(errno)));

	if (std::setvbuf(pF, nullptr, _IOFBF, 32768) != 0)
		std::cout << "setvbuf failed\n";
}

void OutOfRamPixelCloud::close()
{
	if (pF)
	{
		fclose(pF);
		pF = nullptr;
	}
}

void OutOfRamPixelCloud::clear()
{
	if (pF)
	{
		close();
		fs::remove (filepath);
		n_items = 0;
	}
}

void OutOfRamPixelCloud::add_pixel(const Pixel2& p)
{
	check_io_ok();

	fwrite((const void*) &(p.x), sizeof(p.x), 1, pF);
	fwrite((const void*)&(p.y), sizeof(p.y), 1, pF);
	fwrite((const void*)&(p.inten), sizeof(p.inten), 1, pF);

	n_items++;
}

size_t OutOfRamPixelCloud::size() const
{
	return n_items;
}

Pixel2 OutOfRamPixelCloud::get_at(size_t idx) const
{
	check_io_ok();

	size_t offs = idx * item_size;
	fseek(pF, offs, SEEK_SET);
	Pixel2 px;
	fread((void*)&(px.x), sizeof(px.x), 1, pF);
	fread((void*)&(px.y), sizeof(px.y), 1, pF);
	fread((void*)&(px.inten), sizeof(px.inten), 1, pF);
	return px;
}

WriteImageMatrix_nontriv::WriteImageMatrix_nontriv (const std::string& _name, unsigned int _roi_label):
	name(_name)
{
	auto tid = std::this_thread::get_id();

	std::stringstream ssPath;
	ssPath << Nyxus::theEnvironment.get_temp_dir_path() << name << "_roi" << _roi_label << "_tid" << tid << ".mat.nyxus";
	filepath = ssPath.str();

	errno = 0;
	pF = fopen (filepath.c_str(), "w+b");
	if (!pF)
		throw (std::runtime_error("Error creating file " + filepath + " Error code:" + std::to_string(errno)));

	if (std::setvbuf (pF, nullptr, _IOFBF, 32768) != 0) 
		std::cout << "setvbuf failed\n";
}

WriteImageMatrix_nontriv::~WriteImageMatrix_nontriv()
{
	if (pF)
	{
		fclose(pF);
		pF = nullptr;
		fs::remove (filepath);
	}
}

void WriteImageMatrix_nontriv::allocate (int w, int h, double ini_value)
{
	width = w;
	height = h;

	// Fill the file with 0-s
	auto n = width * height;
	double buf = ini_value;
	for (size_t i=0; i<n; i++)
		fwrite((const void*)&buf, sizeof(buf), 1, pF);
	fflush(pF);
}

void WriteImageMatrix_nontriv::allocate_from_cloud (const OutOfRamPixelCloud & cloud, const AABB & aabb, bool mask_image)
{
	// Allocate space
	allocate(aabb.get_width(), aabb.get_height(), 0);

	// '+1' does the job: wherever intensity pixels are defined in 'cloud', 
	// in the image we will have at least 1 even if the pixel intensity is 0
	int maskMagic = mask_image ? 1 : 0;
	
	// Fill it with cloud pixels 
	for (size_t i = 0; i < cloud.size(); i++)
	{
		const Pixel2 p = cloud.get_at(i);
		auto y = p.y - aabb.get_ymin(),
			x = p.x - aabb.get_xmin();
		set_at (y, x, p.inten + maskMagic);
	}

	// Flush the buffer
	fflush(pF);
}

void WriteImageMatrix_nontriv::allocate_from_cloud_coarser_grayscale (const OutOfRamPixelCloud& cloud, const AABB& aabb, PixIntens min_inten, PixIntens inten_range, unsigned int n_grays)
{
	// Allocate space
	allocate(aabb.get_width(), aabb.get_height(), 0);

	// Fill it with cloud pixels 
	for (size_t i = 0; i < cloud.size(); i++)
	{
		const Pixel2 p = cloud.get_at(i);
		auto y = p.y - aabb.get_ymin(),
			x = p.x - aabb.get_xmin();
		PixIntens newI = Nyxus::to_grayscale (p.inten, min_inten, inten_range, n_grays);
		set_at (y, x, newI);
	}

	// Flush the buffer
	fflush(pF);
}

void WriteImageMatrix_nontriv::copy (WriteImageMatrix_nontriv& other)
{
	this->check_non_empty();
	other.check_non_empty();

	if (this->height != other.height || this->width != other.width)
	{
		std::string msg = "Error: size mismatch while copying " + other.info() + " to " + this->info();
		throw (std::runtime_error(msg));	
	}

	for (size_t i = 0; i < this->size(); i++)
	{
		auto val = other[i];
		this->set_at(i, val);
	}

	fflush(pF);
}

void WriteImageMatrix_nontriv::set_at(size_t idx, double val)
{
	size_t offs = idx * item_size;
	fseek(pF, offs, SEEK_SET);
	fwrite((const void*)&val, sizeof(val), 1, pF);
	fflush(pF);
}

void WriteImageMatrix_nontriv::set_at (int row, int col, double val)
{
	size_t idx = row * width + col;
	set_at(idx, val);
}

double WriteImageMatrix_nontriv::get_at(size_t idx)
{
	size_t offs = idx * item_size;
	fseek(pF, offs, SEEK_SET);
	double val;
	fread ((void*)&val, sizeof(val), 1, pF);
	return val;
}

double WriteImageMatrix_nontriv::yx (size_t row, size_t col)
{
	size_t idx = row * width + col;
	double val = get_at (idx);
	return val;
}

double WriteImageMatrix_nontriv::get_max()
{
	bool blank = true;
	double retval = NAN;

	auto n = size();
	double buf = 0.0;
	for (size_t i = 0; i < n; i++)
	{
		buf = get_at(i);
		if (blank)
		{
			blank = false;
			retval = buf;
		}
		else
			retval = std::max(retval, buf);
	}
	return retval;
}

size_t WriteImageMatrix_nontriv::size()
{
	return width * height;
}

size_t WriteImageMatrix_nontriv::get_width()
{
	return width;
}

size_t WriteImageMatrix_nontriv::get_height()
{
	return height;
}

// Returns chord length at x
size_t WriteImageMatrix_nontriv::get_chlen (size_t col)
{
	bool noSignal = true;
	int chlen = 0, maxChlen = 0;	// We will find the maximum chord in case ROI has holes

	for (int row = 0; row < height; row++)
	{
		if (noSignal)
		{
			if (yx(row, col) != 0)
			{
				// begin tracking a new chord
				noSignal = false;
				chlen = 1;
			}
		}
		else // in progress tracking a signal
		{
			if (yx(row, col) != 0)
				chlen++;	// signal continues
			else
			{
				// signal has ended
				maxChlen = std::max(maxChlen, chlen);
				chlen = 0;
				noSignal = true;
			}
		}
	}

	return maxChlen;
}

bool WriteImageMatrix_nontriv::safe(size_t y, size_t x) const
{
	if (x >= width || y >= height)
		return false;
	else
		return true;
}

void WriteImageMatrix_nontriv::check_non_empty()
{
	if (height == 0 || width == 0)
	{
		std::string msg = "Error: attempt to perform action on uninitialized " + this->name;
		throw (std::runtime_error(msg));
	}
}

std::string WriteImageMatrix_nontriv::info()
{
	std::string retval = this->name + " (width=" + std::to_string(width) + " height=" + std::to_string(height) + ")";
	return retval;
}

double OOR_ReadMatrix::get_at (size_t pixel_row, size_t pixel_col) const
{
	size_t tile_x = imloader.get_tile_x(pixel_col),
		tile_y = imloader.get_tile_y(pixel_row);
	imloader.load_tile(tile_y, tile_x);
	auto& dataI = imloader.get_int_tile_buffer();
	size_t idx = imloader.get_within_tile_idx(pixel_row, pixel_col);
	auto val = dataI[idx];
	return (double)val;
}

double OOR_ReadMatrix::get_at(size_t idx) const
{
	auto y = aabb.get_ymin() + idx / aabb.get_width(),
		x = aabb.get_xmin() + idx % aabb.get_width();
	double retval = get_at(y, x);
	return retval;
}

size_t OOR_ReadMatrix::get_width() const
{
	return aabb.get_width();
}

size_t OOR_ReadMatrix::get_height() const
{
	return aabb.get_height();
}

size_t OOR_ReadMatrix::get_size() const
{
	return get_width() * get_height();
}

std::tuple<size_t, size_t> OOR_ReadMatrix::idx_2_rc(size_t idx) const
{
	auto width = get_width();
	size_t row = idx % width,
		col = idx / width;
	return { row, col };
}

bool OOR_ReadMatrix::safe(size_t x, size_t y) const
{
	if (x >= aabb.get_width() || y >= aabb.get_height())
		return false;
	else
		return true;
}

ReadImageMatrix_nontriv::ReadImageMatrix_nontriv (const AABB & _aabb)
{
	aabb = _aabb;
}

double ReadImageMatrix_nontriv::get_at (ImageLoader& imloader, size_t pixel_row, size_t pixel_col)
{
	size_t tile_x = imloader.get_tile_x (pixel_col),
		tile_y = imloader.get_tile_y (pixel_row);
	imloader.load_tile (tile_y, tile_x);
	auto& dataI = imloader.get_int_tile_buffer();
	size_t idx = imloader.get_within_tile_idx (pixel_row, pixel_col);
	auto val = dataI[idx];
	return (double)val;
}

double ReadImageMatrix_nontriv::get_at (ImageLoader& imloader, size_t idx)
{
	auto y = aabb.get_ymin() + idx / aabb.get_width(),
		x = aabb.get_xmin() + idx % aabb.get_width();
	double retval = get_at(imloader, y, x);
	return retval;
}

size_t ReadImageMatrix_nontriv::get_width() const
{
	return aabb.get_width();
}

size_t ReadImageMatrix_nontriv::get_height() const
{
	return aabb.get_height();
}

size_t ReadImageMatrix_nontriv::get_size() const
{
	return get_width() * get_height();
}

std::tuple<size_t, size_t> ReadImageMatrix_nontriv::idx_2_rc (size_t idx) const
{
	auto width = get_width();
	size_t row = idx % width,
		col = idx / width;
	return {row, col};
}

bool ReadImageMatrix_nontriv::safe(size_t x, size_t y) const
{
	if (x >= aabb.get_width() || y >= aabb.get_height())
		return false;
	else
		return true;
}

/// @brief Use base_level=0 and attenuation >0 and <1 e.g. 0.5 to build an imag of a specific intensity distribution. Or base_level=1 and attenuation 1 to build an image of the mask
/// @param labels_raw_pixels ROI pixel cloud
/// @param aabb ROI axis aligned bounding box
/// @param base_level Set {0,1}
/// @param attenuation Value in the interval (0,1]
Power2PaddedImageMatrix_NT::Power2PaddedImageMatrix_NT (const std::string& _name, unsigned int _roi_label, const OutOfRamPixelCloud& raw_pixels, const AABB& aabb, PixIntens base_level, double attenuation) :
	WriteImageMatrix_nontriv(_name, _roi_label)
{
	// Cache AABB
	original_aabb = aabb;

	// Figure out the padded size and allocate
	int bigSide = std::max(aabb.get_width(), aabb.get_height());
	StatsInt paddedSide = Nyxus::closest_pow2(bigSide);
	allocate(paddedSide, paddedSide, 0.0);

	// Copy pixels
	int padOffsetX = (paddedSide - original_aabb.get_width()) / 2;
	int padOffsetY = (paddedSide - original_aabb.get_height()) / 2;

	for (auto pxl : raw_pixels)
	{
		auto x = pxl.x - original_aabb.get_xmin() + padOffsetX,
			y = pxl.y - original_aabb.get_ymin() + padOffsetY;
		set_at(y * width + x, pxl.inten * attenuation + base_level);
	}
}

bool Power2PaddedImageMatrix_NT::tile_contains_signal (int tile_row, int tile_col, int tile_side)
{
	int r1 = tile_row * tile_side,
		r2 = r1 + tile_side,
		c1 = tile_col * tile_side,
		c2 = c1 + tile_side;
	for (int r = r1; r < r2; r++)
		for (int c = c1; c < c2; c++)
			if (yx(r, c) != 0)
				return true;
	return false;
}

