#pragma once

#include <vector>
#include "aabb.h"
#include "pixel.h"
#include "../feature_method.h"


/// @brief Zernike features characterize the distribution of intensity across the object. Code originally written by Michael Boland and adapted by Ilya Goldberg

class ZernikeFeature: public FeatureMethod
{
public:
	ZernikeFeature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	static const short ZERNIKE2D_ORDER = 9, NUM_FEATURE_VALS = 30;

	// Compatibility with manual reduce
	static bool required(const FeatureSet& fs) { return fs.isEnabled(Nyxus::Feature2D::ZERNIKE2D); }

private:

	/// @brief Zernike moment generating function.  The moment of degree n and
	/// angular dependence l for the pixels defined by coordinate vectors
	/// X and Y and intensity vector P.  X, Y, and P must have the same length
	void mb_Znl(double* X, double* Y, double* P, int size, double D, double m10_m00, double m01_m00, double R, double psum, double* zvalues, long* output_size);

	/// @brief Algorithms for fast computation of Zernike momentsand their numerical stability
	/// Chandan Singhand Ekta Walia, Imageand Vision Computing 29 (2011) 251–259 implemented from 
	/// pseudo-code by Ilya Goldberg
	void mb_zernike2D (const ImageMatrix& Im, double order, double rad, double* zvalues);

	/// @brief Algorithms for fast computation of Zernike momentsand their numerical stability
	/// Chandan Singhand Ekta Walia, Imageand Vision Computing 29 (2011) 251–259 implemented from 
	/// pseudo-code by Ilya Goldberg
	void mb_zernike2D_nontriv (WriteImageMatrix_nontriv& I, double order, double rad, double* zvalues);

	std::vector<double> coeffs;
};

#define MAX_L 32

// This sets the maximum D parameter (15)
// The D parameter has to match MAX_D. See mb_Znl() below.
#define MAX_D 15

// This is based on the maximum D parameter being 15, which sets the number of returned values.
#define MAX_Z 72

// This is also based on the maximum D parameter - contains pre-computed factorials
#define MAX_LUT 240


