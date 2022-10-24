#include <fstream>
#include <iostream>
#include <iomanip>
#include "../helpers/helpers.h"
#include "image_matrix.h"
#include "moments.h"

template<>
void SimpleMatrix<int>::print(const std::string& head, const std::string& tail)
{
	const int Wd = 6;	// data
	const int Wi = 5;		// index

	std::cout << head << "\n";
	std::cout << std::string(Wi + Wd * this->width(), '-') << std::endl;		// Upper solid line
	std::cout << "w=" << this->width() << " h=" << this->height() << "\n";

	for (int row = 0; row < this->height(); row++)
	{
		// Hdr
		if (row == 0)
		{
			std::cout << std::setw(Wi + 2) << "";	// Wi+2 because '[' + Wi + ']'
			for (int col = 0; col < this->width(); col++)
			{
				std::cout << std::setw(Wd) << col;
			}
			std::cout << "\n";
		}

		// Row
		std::cout << "[" << std::setw(Wi) << row << "]";
		for (int col = 0; col < this->width(); col++)
		{
			std::cout << std::setw(Wd) << (int) this->xy(col, row);
		}
		std::cout << "\n";
	}

	std::cout << std::string(Wi + Wd * this->width(), '-') << std::endl;	// Lower solid line
	std::cout << tail;
}

void ImageMatrix::print (const std::string& head, const std::string& tail, std::vector<PrintablePoint> special_points)
{
	const int Wd = 6;	// data
	const int Wi = 5;		// index

	readOnlyPixels D = ReadablePixels();

	std::cout << head << "\n";
	std::cout << std::string(Wi + Wd * this->width, '-') << std::endl;	// Upper solid line
	std::cout << "w=" << this->width << " h=" << this->height << "\n";

	for (int row = 0; row < this->height; row++)
	{
		// Header
		if (row == 0)
		{
			std::cout << std::setw(Wi + 2) << "";	// Wi+2 because '[' + Wi + ']'
			for (int col = 0; col < this->width; col++)
			{
				std::cout << std::setw(Wd) << col;
			}
			std::cout << "\n";
		}

		// Row
		std::cout << "[" << std::setw(Wi) << row << "]";
		for (int col = 0; col < this->width; col++)
		{
			// Any special pixel at location () ?
			bool haveSpecPix = false;
			for (auto& p : special_points)
			{
				int x = std::get<0>(p),
					y = std::get<1>(p);
				int loc_x = x - original_aabb.get_xmin(),
					loc_y = y - original_aabb.get_ymin();
				if (col == loc_x && row == loc_y)
				{
					haveSpecPix = true;
					std::string txt = std::get<2>(p);
					std::cout << std::setw(Wd) << txt;
					break;	// No need to consider other special pixels -- the rule is to have only 1 pixel per location
				}
			}
			if (haveSpecPix)
				continue;

			// Regular pixel
			auto I = D.yx (row, col);
			if (I == 0)
				std::cout << std::setw(Wd) << '.';
			else
				std::cout << std::setw(Wd) << I;
		}
		std::cout << "\n";
	}

	std::cout << std::string(Wi + Wd * this->width, '-') << std::endl;	// Lower solid line
	std::cout << tail;
}

void ImageMatrix::print (std::ofstream& f, const std::string& head, const std::string& tail, std::vector<PrintablePoint> special_points)
{
	const int Wd = 6;	// data
	const int Wi = 5;		// index

	readOnlyPixels D = ReadablePixels();

	f << head << "\n";
	f << std::string(Wi + Wd * this->width, '-') << std::endl;	// Upper solid line
	f << "w=" << this->width << " h=" << this->height << "\n";

	for (int row = 0; row < this->height; row++)
	{
		// Header
		if (row == 0)
		{
			f << std::setw(Wi + 2) << "";	// Wi+2 because '[' + Wi + ']'
			for (int col = 0; col < this->width; col++)
			{
				f << std::setw(Wd) << col;
			}
			f << "\n";
		}

		// Row
		f << "[" << std::setw(Wi) << row << "]";
		for (int col = 0; col < this->width; col++)
		{
			// Any special pixel at location () ?
			bool haveSpecPix = false;
			for (auto& p : special_points)
			{
				int x = std::get<0>(p),
					y = std::get<1>(p);
				int loc_x = x - original_aabb.get_xmin(),
					loc_y = y - original_aabb.get_ymin();
				if (col == loc_x && row == loc_y)
				{
					haveSpecPix = true;
					std::string txt = std::get<2>(p);
					f << std::setw(Wd) << txt;
					break;	// No need to consider other special pixels -- the rule is to have only 1 pixel per location
				}
			}
			if (haveSpecPix)
				continue;

			// Regular pixel
			auto I = D.yx(row, col);
			if (I == 0)
				f << std::setw(Wd) << '.';
			else
				f << std::setw(Wd) << I;
		}
		f << "\n";
	}

	f << std::string(Wi + Wd * this->width, '-') << std::endl;	// Lower solid line
	f << tail;
}

// Binarization by Otsu's method based on maximization of inter-class variance 
double ImageMatrix::Otsu(bool dynamic_range) const {
	#define OTSU_LEVELS 256	
	double hist[OTSU_LEVELS];
	double omega[OTSU_LEVELS];
	double myu[OTSU_LEVELS];
	double max_sigma, sigma[OTSU_LEVELS]; // inter-class variance
	int i;
	int threshold;
	double min_val, max_val; // pixel range

	if (!dynamic_range) 
	{
		histogram(hist, OTSU_LEVELS, true);
		min_val = 0.0;
		int bits = sizeof(PixIntens) * 8;
		max_val = pow(2.0, bits) - 1;
	}
	else 
	{
		// to keep this const method from modifying the object, we use GetStats on a local Moments2 object
		Moments2 local_stats;
		GetStats(local_stats);
		min_val = local_stats.min__();
		max_val = local_stats.max__();
		histogram(hist, OTSU_LEVELS, false);
	}

	// omega & myu generation
	omega[0] = hist[0] / stats.n();

	myu[0] = 0.0;
	for (i = 1; i < OTSU_LEVELS; i++) 
	{
		omega[i] = omega[i - 1] + (hist[i] / stats.n());
		myu[i] = myu[i - 1] + i * (hist[i] / stats.n());
	}

	// maximization of inter-class variance
	threshold = 0;
	max_sigma = 0.0;
	for (i = 0; i < OTSU_LEVELS - 1; i++) 
	{
		if (omega[i] != 0.0 && omega[i] != 1.0)
			sigma[i] = pow(myu[OTSU_LEVELS - 1] * omega[i] - myu[i], 2) / (omega[i] * (1.0 - omega[i]));
		else
			sigma[i] = 0.0;
		if (sigma[i] > max_sigma) 
		{
			max_sigma = sigma[i];
			threshold = i;
		}
	}

	// threshold is a histogram index - needs to be scaled to a pixel value.
	return ((((double)threshold / (double)(OTSU_LEVELS - 1)) * (max_val - min_val)) + min_val);
}

// get image histogram 
void ImageMatrix::histogram(double* bins, unsigned short nbins, bool imhist, const Moments2& in_stats) const 
{
	unsigned long a, bin, num = width * height;
	double val, h_min = INF, h_max = -INF, h_scale;
	readOnlyPixels pix_plane = ReadablePixels();

	// find the minimum and maximum 
	if (imhist) 
	{
		// similar to the Matlab imhist 
		h_min = 0;
		int bits = sizeof(PixIntens) * 8;
		h_max = pow((double)2, bits) - 1;
	}
	else if (in_stats.n() > 0) 
	{
		h_min = in_stats.min__();
		h_max = in_stats.max__();
	}
	else 
	{
		// to keep this const method from modifying the object, we use GetStats on a local Moments2 object
		Moments2 local_stats;
		GetStats(local_stats);
		h_min = local_stats.min__();
		h_max = local_stats.max__();
	}
	if (h_max - h_min > 0) h_scale = (double)nbins / double(h_max - h_min);
	else h_scale = 0;

	// initialize the bins, invicem "memset(bins, 0, nbins * sizeof(double))"
	for (int i = 0; i < nbins; i++)
		bins[i] = 0.0;

	// build the histogram
	for (a = 0; a < num; a++) 
	{
		val = pix_plane[a];  
		if (std::isnan(val)) 
			continue;
		bin = (unsigned long)(((val - h_min) * h_scale));
		if (bin >= nbins) bin = nbins - 1;
		bins[bin] += 1.0;
	}

	return;
}

void ImageMatrix::apply_distance_to_contour_weights (const std::vector<Pixel2>& raw_pixels, const std::vector<Pixel2>& contour_pixels)
{
	const double epsilon = 0.1;

	for (auto& p : raw_pixels)
	{
		auto mind = p.min_sqdist (contour_pixels);
		double dist = std::sqrt(mind);

		// (row, column) coordinates in the image matrix
		auto c = p.x - original_aabb.get_xmin(),
			r = p.y - original_aabb.get_ymin();
		
		// Weighted intensity
		PixIntens wi = _pix_plane.yx(r, c) / (dist + epsilon) + 0.5/*rounding*/;
		
		_pix_plane.yx(r,c) = wi;
	}
}

// Returns chord length at x
int ImageMatrix::get_chlen (int col)
{
	bool noSignal = true;
	int chlen = 0, maxChlen = 0;	// We will find the maximum chord in case ROI has holes

	for (int row = 0; row < height; row++)
	{
		if (noSignal)
		{
			if (_pix_plane.yx(row, col) != 0)
			{
				// begin tracking a new chord
				noSignal = false;
				chlen = 1;
			}
		}
		else // in progress tracking a signal
		{
			if (_pix_plane.yx(row, col) != 0)
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

bool ImageMatrix::tile_contains_signal (int tile_row, int tile_col, int tile_side)
{
	int r1 = tile_row * tile_side,
		r2 = r1 + tile_side,
		c1 = tile_col * tile_side,
		c2 = c1 + tile_side;
	for (int r = r1; r < r2; r++)
		for (int c = c1; c < c2; c++)
			if (_pix_plane.yx(r, c) != 0)
				return true;
	return false;
}
