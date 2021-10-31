#include <iostream>
#include <iomanip>
#include "image_matrix.h"
#include "sensemaker.h"

template <>
void SimpleMatrix<int>::print(const std::string& head, const std::string& tail)
{
	const int Wd = 6;	// data
	const int Wi = 5;	// index

	std::cout << head << "\n";
	std::cout << std::string(Wi + Wd * this->width(), '-') << std::endl;	// Upper solid line
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
			std::cout << std::setw(Wd) << (int) this->operator()(row, col);
		}
		std::cout << "\n";
	}

	std::cout << std::string(Wi + Wd * this->width(), '-') << std::endl;	// Lower solid line
	std::cout << tail;
}

void ImageMatrix::print (const std::string& head, const std::string& tail, std::vector<PrintablePoint> special_points)
{
	const int Wd = 6;	// data
	const int Wi = 5;	// index

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
			/*--- Hilighted pixel v.1
			// Print a regular or highlighted pixel
			if (col == local_hilight_x && row == local_hilight_y)
			{
				// Highlighted pixel
				std::cout << std::setw(Wd) << hilight_text;
			}
			else
			{
				// Regular pixel
				auto I = D(row, col);
				if (I == 0)
					std::cout << std::setw(Wd) << '.';
				else
					std::cout << std::setw(Wd) << I;
			}
			*/

			//--- Hilighted pixel v.2
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
			auto I = D(row, col);
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

//-----------------------------------------------------------------------------------
/* Otsu
   Find otsu threshold
*/
double ImageMatrix::Otsu(bool dynamic_range) const {
	/* binarization by Otsu's method
	based on maximization of inter-class variance */
#define OTSU_LEVELS 1024
	double hist[OTSU_LEVELS];
	double omega[OTSU_LEVELS];
	double myu[OTSU_LEVELS];
	double max_sigma, sigma[OTSU_LEVELS]; // inter-class variance
	int i;
	int threshold;
	double min_val, max_val; // pixel range

	if (!dynamic_range) {
		histogram(hist, OTSU_LEVELS, true);
		min_val = 0.0;
		int bits = sizeof(PixIntens) * 8;
		max_val = pow(2.0, bits) - 1;
	}
	else {
		// to keep this const method from modifying the object, we use GetStats on a local Moments2 object
		Moments2 local_stats;
		GetStats(local_stats);
		min_val = local_stats.min__();
		max_val = local_stats.max__();
		histogram(hist, OTSU_LEVELS, false);
	}

	// omega & myu generation
	//MM omega[0] = hist[0] / (width * height);
	omega[0] = hist[0] / stats.n();

	myu[0] = 0.0;
	for (i = 1; i < OTSU_LEVELS; i++) {
		//MM  omega[i] = omega[i-1] + (hist[i] / (width * height));
		//MM  myu[i] = myu[i-1] + i*(hist[i] / (width * height));
		omega[i] = omega[i - 1] + (hist[i] / stats.n());
		myu[i] = myu[i - 1] + i * (hist[i] / stats.n());
	}

	// maximization of inter-class variance
	threshold = 0;
	max_sigma = 0.0;
	for (i = 0; i < OTSU_LEVELS - 1; i++) {
		if (omega[i] != 0.0 && omega[i] != 1.0)
			sigma[i] = pow(myu[OTSU_LEVELS - 1] * omega[i] - myu[i], 2) /
			(omega[i] * (1.0 - omega[i]));
		else
			sigma[i] = 0.0;
		if (sigma[i] > max_sigma) {
			max_sigma = sigma[i];
			threshold = i;
		}
	}

	// threshold is a histogram index - needs to be scaled to a pixel value.
	return ((((double)threshold / (double)(OTSU_LEVELS - 1)) * (max_val - min_val)) + min_val);
}

/* get image histogram */
void ImageMatrix::histogram(double* bins, unsigned short nbins, bool imhist, const Moments2& in_stats) const {
	unsigned long a, bin, num = width * height;
	double val, h_min = INF, h_max = -INF, h_scale;
	readOnlyPixels pix_plane = ReadablePixels();

	/* find the minimum and maximum */
	if (imhist) {    /* similar to the Matlab imhist */
		h_min = 0;
		int bits = sizeof(PixIntens) * 8;
		h_max = pow((double)2, bits) - 1;
	}
	else if (in_stats.n() > 0) {
		h_min = in_stats.min__();
		h_max = in_stats.max__();
	}
	else {
		// to keep this const method from modifying the object, we use GetStats on a local Moments2 object
		Moments2 local_stats;
		GetStats(local_stats);
		h_min = local_stats.min__();
		h_max = local_stats.max__();
	}
	if (h_max - h_min > 0) h_scale = (double)nbins / double(h_max - h_min);
	else h_scale = 0;

	// initialize the bins
	//memset(bins, 0, nbins * sizeof(double));
	for (int i = 0; i < nbins; i++)
		bins[i] = 0.0;

	// build the histogram
	for (a = 0; a < num; a++) {
		val = pix_plane[a];  // pix_plane.array().coeff(a);
		if (std::isnan(val)) continue; //MM
		bin = (unsigned long)(((val - h_min) * h_scale));
		if (bin >= nbins) bin = nbins - 1;
		bins[bin] += 1.0;
	}

	return;
}

