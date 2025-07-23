#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "3d_ngtdm.h"
#include "image_matrix_nontriv.h"
#include "../environment.h"

using namespace Nyxus;

int D3_NGTDM_feature::n_levels = 0;

D3_NGTDM_feature::D3_NGTDM_feature() : FeatureMethod("D3_NGTDM_feature")
{
	provide_features(D3_NGTDM_feature::featureset);
}

void D3_NGTDM_feature::clear_buffers()
{
	bad_roi_data = false;
	Ng = 0;
	Ngp = 0;
	Nvp = 0;
	Nd = 0;
	Nz = 0;
	Nvc = 0;
	P.clear();
	S.clear();
	N.clear();
}

void D3_NGTDM_feature::calculate(LR& r)
{
	// Clear variables
	clear_buffers();

	// grey-bin intensities
	int w = r.aux_image_cube.width(),
		h = r.aux_image_cube.height(),
		d = r.aux_image_cube.depth();

	SimpleCube<PixIntens> D;
	D.allocate(w, h, d);

	auto greyInfo = Nyxus::theEnvironment.get_coarse_gray_depth();
	auto greyInfo_localFeature = D3_NGTDM_feature::n_levels;
	if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
		greyInfo = greyInfo_localFeature;
	if (Nyxus::theEnvironment.ibsi_compliance)
		greyInfo = 0;

	bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

	// zero (backround) intensity at given grey binning method
	PixIntens zeroI = matlab_grey_binning(greyInfo) ? 1 : 0;

	// unique intensities
	std::unordered_set<PixIntens> U(D.begin(), D.end());
	U.erase(0);	// discard intensity '0'

	if (ibsi_grey_binning(greyInfo))
	{
		// intensities 0-max
		auto max_I = *std::max_element(U.begin(), U.end());
		for (PixIntens i = 0; i <= max_I; i++)
			I.push_back(i);
	}
	else
		// only unique intensities
		I.assign(U.begin(), U.end());

	std::sort(I.begin(), I.end());

	// is binned data informative?
	if (I.size() < 2)
	{
		_coarseness =
			_contrast =
			_busyness =
			_complexity =
			_strength = theEnvironment.resultOptions.noval();
		return;
	}

	// Define voxel neighborhood
	struct ShiftToNeighbor
	{
		int dx, dy, dz;
	};
	const static ShiftToNeighbor shifts[] =
	{
		{-1,		0,			0},		// West
		{-1,		-1,		0},		// North-West
		{0,		-1,		0},		// North
		{+1,		-1,		0},		// North-East
		{+1,		0,			0},		// East
		{+1,		+1,		0},		// South-East
		{0,		+1,		0},		// South
		{-1,		+1,		0}	,		// South-West

		{-1,		0,			+1},		// West
		{-1,		-1,		+1},		// North-West
		{0,		-1,		+1},		// North
		{+1,		-1,		+1},		// North-East
		{+1,		0,			+1},		// East
		{+1,		+1,		+1},		// South-East
		{0,		+1,		+1},		// South
		{-1,		+1,		+1},		// South-West	

		{-1,		0,			-1},		// West
		{-1,		-1,		-1},		// North-West
		{0,		-1,		-1},		// North
		{+1,		-1,		-1},		// North-East
		{+1,		0,			-1},		// East
		{+1,		+1,		-1},		// South-East
		{0,		+1,		-1},		// South
		{-1,		+1,		-1}		// South-West	
	};

	int nsh = sizeof(shifts) / sizeof(ShiftToNeighbor);

	// Gather zones
	using AveNeighborhoodInte = std::pair<PixIntens, double>;	// Pairs of (intensity, average intensity of all 8 neighbors)
	std::vector<AveNeighborhoodInte> Z;	// list of intensity clusters (zones)
	for (int zslice = 0; zslice < d; zslice++)
	{
		for (int row = 0; row < h; row++)
		{
			for (int col = 0; col < w; col++)
			{
				PixIntens pi = D.zyx (zslice, row, col);

				// Skip background voxels
				if (pi == zeroI)
					continue;

				// Evaluate the neighborhood
				double neigsI = 0;

				int nd = 0;	// Number of dependencies

				// shift z := 0

				if (D.safe(zslice, row-1, col) && D.zyx(zslice, row-1, col) != 0)	// North
				{
					neigsI += D.zyx (zslice, row - 1, col);
					nd++;
				}

				if (D.safe(zslice, row-1, col+1) && D.zyx(zslice, row-1, col+1) != 0)	// North-East
				{
					neigsI += D.zyx (zslice, row-1, col+1);
					nd++;
				}

				if (D.safe(zslice, row, col+1) && D.zyx(zslice, row, col+1) != 0)	// East
				{
					neigsI += D.zyx (zslice, row, col+1);
					nd++;
				}
				if (D.safe(zslice, row+1, col+1) && D.zyx(zslice, row+1, col+1) != 0)	// South-East
				{
					neigsI += D.zyx (zslice, row+1, col+1);
					nd++;
				}
				if (D.safe(zslice, row+1, col) && D.zyx(zslice, row+1, col) != 0) // South
				{
					neigsI += D.zyx (zslice, row+1, col);
					nd++;
				}
				if (D.safe(zslice, row+1, col-1) && D.zyx(zslice, row+1, col-1) != 0)	// South-West
				{
					neigsI += D.zyx (zslice, row+1, col-1);
					nd++;
				}
				if (D.safe(zslice, row, col-1) && D.zyx(zslice, row, col-1) != 0)	// West
				{
					neigsI += D.zyx (zslice, row, col-1);
					nd++;
				}
				if (D.safe(zslice, row-1, col-1) && D.zyx(zslice, row-1, col-1) != 0)	 // North-West
				{
					neigsI += D.zyx (zslice, row-1, col-1);
					nd++;
				}

				// shift z := +1
				int dz = 1;

				if (D.safe(zslice+dz, row - 1, col) && D.zyx(zslice, row - 1, col) != 0)	// North
				{
					neigsI += D.zyx(zslice, row - 1, col);
					nd++;
				}

				if (D.safe(zslice+dz, row - 1, col + 1) && D.zyx(zslice, row - 1, col + 1) != 0)	// North-East
				{
					neigsI += D.zyx(zslice, row - 1, col + 1);
					nd++;
				}

				if (D.safe(zslice+dz, row, col + 1) && D.zyx(zslice, row, col + 1) != 0)	// East
				{
					neigsI += D.zyx(zslice, row, col + 1);
					nd++;
				}
				if (D.safe(zslice+dz, row + 1, col + 1) && D.zyx(zslice, row + 1, col + 1) != 0)	// South-East
				{
					neigsI += D.zyx(zslice, row + 1, col + 1);
					nd++;
				}
				if (D.safe(zslice+dz, row + 1, col) && D.zyx(zslice, row + 1, col) != 0) // South
				{
					neigsI += D.zyx(zslice, row + 1, col);
					nd++;
				}
				if (D.safe(zslice+dz, row + 1, col - 1) && D.zyx(zslice, row + 1, col - 1) != 0)	// South-West
				{
					neigsI += D.zyx(zslice, row + 1, col - 1);
					nd++;
				}
				if (D.safe(zslice+dz, row, col - 1) && D.zyx(zslice, row, col - 1) != 0)	// West
				{
					neigsI += D.zyx(zslice, row, col - 1);
					nd++;
				}
				if (D.safe(zslice+dz, row - 1, col - 1) && D.zyx(zslice, row - 1, col - 1) != 0)	 // North-West
				{
					neigsI += D.zyx(zslice, row - 1, col - 1);
					nd++;
				}

				// shift z := -1
				dz = -1;

				if (D.safe(zslice + dz, row - 1, col) && D.zyx(zslice, row - 1, col) != 0)	// North
				{
					neigsI += D.zyx(zslice, row - 1, col);
					nd++;
				}

				if (D.safe(zslice + dz, row - 1, col + 1) && D.zyx(zslice, row - 1, col + 1) != 0)	// North-East
				{
					neigsI += D.zyx(zslice, row - 1, col + 1);
					nd++;
				}

				if (D.safe(zslice + dz, row, col + 1) && D.zyx(zslice, row, col + 1) != 0)	// East
				{
					neigsI += D.zyx(zslice, row, col + 1);
					nd++;
				}
				if (D.safe(zslice + dz, row + 1, col + 1) && D.zyx(zslice, row + 1, col + 1) != 0)	// South-East
				{
					neigsI += D.zyx(zslice, row + 1, col + 1);
					nd++;
				}
				if (D.safe(zslice + dz, row + 1, col) && D.zyx(zslice, row + 1, col) != 0) // South
				{
					neigsI += D.zyx(zslice, row + 1, col);
					nd++;
				}
				if (D.safe(zslice + dz, row + 1, col - 1) && D.zyx(zslice, row + 1, col - 1) != 0)	// South-West
				{
					neigsI += D.zyx(zslice, row + 1, col - 1);
					nd++;
				}
				if (D.safe(zslice + dz, row, col - 1) && D.zyx(zslice, row, col - 1) != 0)	// West
				{
					neigsI += D.zyx(zslice, row, col - 1);
					nd++;
				}
				if (D.safe(zslice + dz, row - 1, col - 1) && D.zyx(zslice, row - 1, col - 1) != 0)	 // North-West
				{
					neigsI += D.zyx(zslice, row - 1, col - 1);
					nd++;
				}

				// Save the intensity's average neighborhood intensity
				if (nd > 0)
				{
					neigsI /= nd;
					AveNeighborhoodInte z = { pi, neigsI };
					Z.push_back(z);
				}
			}
		}
	}

	// Fill the matrix
	// --dimensions
	Ng = (int)I.size();	//---pre 2024---> Ng = Environment::ibsi_compliance ? *std::max_element(std::begin(im.ReadablePixels()), std::end(im.ReadablePixels())) : (int) U.size();
	Ngp = (int)U.size();

	// --allocate the matrix
	P.resize(Ng + 1, 0);
	S.resize(Ng + 1, 0);
	N.resize(Ng + 1, 0);

	// --Calculate N and S
	for (auto& z : Z)
	{
		// row (grey level)
		auto inten = z.first;
		int row = -1;
		if (Environment::ibsi_compliance)
			row = inten;
		else
		{
			auto lower = std::lower_bound(I.begin(), I.end(), inten);	// enjoy sorted vector 'I'
			row = int(lower - I.begin());	// intensity index in array of unique intensities 'I'
		}

		// col
		int col = (int)z.second;	// 1-based
		// increment
		N[row]++;
		// --S
		PixIntens pi = I[row];
		double aveNeigI = z.second;
		S[row] += std::abs(pi - aveNeigI);
		// --Nvp
		if (aveNeigI > 0.0)
			Nvp++;
	}

	// --Calculate Nvc (sum of N)
	Nvc = 0;
	for (int i = 0; i < N.size(); i++)
		Nvc += N[i];

	// --Calculate P
	for (int i = 0; i < N.size(); i++)
		P[i] = (double)N[i] / Nvc;

	// Calculate features
	_coarseness = calc_Coarseness();
	_contrast = calc_Contrast();
	_busyness = calc_Busyness();
	_complexity = calc_Complexity();
	_strength = calc_Strength();
}

void D3_NGTDM_feature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature3D::NGTDM_COARSENESS][0] = _coarseness;
	fvals[(int)Feature3D::NGTDM_CONTRAST][0] = _contrast;
	fvals[(int)Feature3D::NGTDM_BUSYNESS][0] = _busyness;
	fvals[(int)Feature3D::NGTDM_COMPLEXITY][0] = _complexity;
	fvals[(int)Feature3D::NGTDM_STRENGTH][0] = _strength;
}

void D3_NGTDM_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting

void D3_NGTDM_feature::osized_calculate(LR& r, ImageLoader&)
{
	// Clear variables
	clear_buffers();

	// Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
	{
		bad_roi_data = true;
		return;
	}

	// Prepare ROI's intensity range for normalize_I()
	PixIntens piRange = r.aux_max - r.aux_min;

	// Make a list of intensity clusters (zones)
	using AveNeighborhoodInte = std::pair<PixIntens, double>;	// Pairs of (intensity, average intensity of all 8 neighbors)
	std::vector<AveNeighborhoodInte> Z;

	// While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	// ROI image
	WriteImageMatrix_nontriv D("D3_NGTDM_feature_osized_calculate_D", r.label);
	D.allocate_from_cloud(r.raw_pixels_NT, r.aabb, false);

	// Gather zones
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
	for (int row = 0; row < D.get_height(); row++)
		for (int col = 0; col < D.get_width(); col++)
		{
			// Find a non-blank pixel 
			PixIntens pi = Nyxus::to_grayscale(D.yx(row, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
			if (pi == 0)
				continue;

			// Update unique intensities
			U.insert(pi);

			// Evaluate the neighborhood
			double neigsI = 0;

			int nd = 0;	// Number of dependencies

			if (D.safe(row - 1, col) && D.yx(row - 1, col) != 0)	// North
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			if (D.safe(row - 1, col + 1) && D.yx(row - 1, col + 1) != 0)	// North-East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			if (D.safe(row, col + 1) && D.yx(row, col + 1) != 0)	// East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col + 1) && D.yx(row + 1, col + 1) != 0)	// South-East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col) && D.yx(row + 1, col) != 0)	// South
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col - 1) && D.yx(row + 1, col - 1) != 0)	// South-West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row, col - 1) && D.yx(row, col - 1) != 0)	// West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row - 1, col - 1) && D.yx(row - 1, col - 1) != 0)	// North-West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			// Save the intensity's average neighborhood intensity
			if (nd > 0)
			{
				neigsI /= nd;
				AveNeighborhoodInte z = { pi, neigsI };
				Z.push_back(z);
			}
		}

	// Fill the matrix

	Ng = (int)U.size();
	Ngp = (int)U.size();

	// --allocate the matrix
	P.resize(Ng + 1, 0);
	S.resize(Ng + 1, 0);
	N.resize(Ng + 1, 0);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --Calculate N and S
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = (Environment::ibsi_compliance) ?
			z.first : int(iter - I.begin());
		// col
		int col = (int)z.second;	// 1-based
		// increment
		N[row]++;
		// --S
		PixIntens pi = row;
		double aveNeigI = z.second;
		S[row] += std::abs(pi - aveNeigI);
		// --Nvp
		if (aveNeigI > 0.0)
			Nvp++;
	}

	// --Calculate Nvc (sum of N)
	Nvc = 0;
	for (int i = 0; i < N.size(); i++)
		Nvc += N[i];

	// --Calculate P
	for (int i = 0; i < N.size(); i++)
		P[i] = (double)N[i] / Nvc;

	// Calculate features
	_coarseness = calc_Coarseness();
	_contrast = calc_Contrast();
	_busyness = calc_Busyness();
	_complexity = calc_Complexity();
	_strength = calc_Strength();
}

// Coarseness
double D3_NGTDM_feature::calc_Coarseness()
{
	// Calculate the feature
	double sum = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum += P[i - 1] * S[i - 1];
	double retval = 1.0 / sum;
	return retval;
}

// Contrast
double D3_NGTDM_feature::calc_Contrast()
{
	// Calculate the feature
	// --term 1
	double sum = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double ival = (double)I[i - 1];
		for (int j = 1; j <= Ng; j++)
		{
			double jval = (double)I[j - 1];
			double tmp = P[i - 1] * P[j - 1] * (ival - jval) * (ival - jval);
			sum += tmp;
		}
	}
	int Ngp_p2 = Ngp > 1 ? Ngp * (Ngp - 1) : Ngp;
	double term1 = sum / double(Ngp_p2);

	// --term 2
	sum = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum += S[i - 1];
	double term2 = sum / Nvc;

	double retval = term1 * term2;
	return retval;
}

// Busyness
double D3_NGTDM_feature::calc_Busyness()
{
	// Trivial case?
	if (Ngp == 1)
		return 0.0;

	// Calculate the feature
	double sum1 = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum1 += P[i - 1] * S[i - 1];

	double sum2 = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double ival = (double)I[i - 1];
		for (int j = 1; j <= Ng; j++)
		{
			double jval = (double)I[j - 1];
			if (P[i - 1] != 0 && P[j - 1] != 0)
			{
				double tmp = P[i - 1] * ival - P[j - 1] * jval;
				sum2 += std::abs(tmp);
			}
		}
	}

	if (sum2 == 0)
		return 0;

	double retval = sum1 / sum2;
	return retval;
}

// Complexity
double D3_NGTDM_feature::calc_Complexity()
{
	// Calculate the feature
	double sum = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double ival = (double)I[i - 1];
		for (int j = 1; j <= Ng; j++)
		{
			double jval = (double)I[j - 1];
			if (P[i - 1] != 0 && P[j - 1] != 0)
			{
				sum += std::abs(ival - jval) * (P[i - 1] * S[i - 1] + P[j - 1] * S[j - 1]) / (P[i - 1] + P[j - 1]);
			}
		}
	}

	double retval = sum / double(Nvp);
	return retval;
}

// Strength
double D3_NGTDM_feature::calc_Strength()
{
	// Calculate the feature
	double sum1 = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double ival = (double)I[i - 1];
		for (int j = 1; j <= Ng; j++)
		{
			double jval = (double)I[j - 1];
			if (P[i - 1] != 0 && P[j - 1] != 0)
			{
				sum1 += (P[i - 1] + P[j - 1]) * (ival - jval) * (ival - jval);
			}
		}
	}

	double sum2 = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum2 += S[i - 1];

	double retval = sum1 / sum2;
	return retval;
}

void D3_NGTDM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		D3_NGTDM_feature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

/*static*/ void D3_NGTDM_feature::extract(LR& r)
{
	D3_NGTDM_feature f;
	f.calculate(r);
	f.save_value(r.fvals);
}


