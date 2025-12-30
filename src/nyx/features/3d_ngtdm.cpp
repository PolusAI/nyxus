
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "3d_ngtdm.h"
#include "image_matrix_nontriv.h"
#include "../environment.h"

using namespace Nyxus;

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

// Define voxel neighborhood
struct ShiftToNeighbor
{
	int dx, dy, dz;
};
const static ShiftToNeighbor shifts[] =
{
	{-1,	0,		0},		// West
	{-1,	-1,		0},		// North-West
	{0,     -1,		0},		// North
	{+1,	-1,		0},		// North-East
	{+1,	0,		0},		// East
	{+1,	+1,		0},		// South-East
	{0,     +1,		0},		// South
	{-1,	+1,		0},		// South-West

	{-1,	0,		+1},	// West
	{-1,	-1,		+1},	// North-West
	{0,	-1,		+1},	// North
	{+1,	-1,		+1},	// North-East
	{+1,	0,		+1},	// East
	{+1,	+1,		+1},	// South-East
	{0,	+1,		+1},	// South
	{-1,	+1,		+1},	// South-West	

	{-1,	0,		-1},	// West
	{-1,	-1,		-1},	// North-West
	{0,	-1,		-1},	// North
	{+1,	-1,		-1},	// North-East
	{+1,	0,		-1},	// East
	{+1,	+1,		-1},	// South-East
	{0,	+1,		-1},	// South
	{-1,	+1,		-1}		// South-West	
};

int nsh = sizeof(shifts) / sizeof(ShiftToNeighbor);

/*static*/ void D3_NGTDM_feature::gather_zones (std::vector<std::pair<PixIntens, double>> &Z, SimpleCube<PixIntens> &D, int cheby_radius, PixIntens zeroI)
{
	int w = D.width(),
		h = D.height(),
		d = D.depth();

	for (int zslice = 0; zslice < d; zslice++)
	{
		for (int row = 0; row < h; row++)
		{
			for (int col = 0; col < w; col++)
			{
				PixIntens pi = D.zyx (zslice, row, col);

				// skip background voxels
				if (pi == zeroI)
					continue;

				// examine the neighborhood
				double neigsI = 0;
				int nd = 0;	// number of dependencies

				// scan neighborhood of voxel (zslice, row, col)
				for (int dz=-cheby_radius; dz<= cheby_radius; dz++)
					for (int dy=-cheby_radius; dy<= cheby_radius; dy++)
						for (int dx=-cheby_radius; dx<= cheby_radius; dx++)
						{
							// skip the voxel of consideration
							int neig_z = zslice + dz,
								neig_y = row + dy, 
								neig_x = col + dx;
							if (neig_z == zslice && neig_y == row && neig_x == col)
								continue;

							// skip voxels outside the image
							if (!D.safe(neig_z, neig_y, neig_x))
								continue;

							// update neighborhood stats
							neigsI += D.zyx (neig_z, neig_y, neig_x);
							nd++;
						}

				// save voxel's average neighborhood intensity
				if (nd > 0)
				{
					std::pair <PixIntens, double> zo = { pi, neigsI/nd };
					Z.push_back (zo);
				}
			}
		}
	}
}

//
// returns Nvp
//

/*static*/ double D3_NGTDM_feature::calc_NGTDM(
	// out
	std::vector <int> &N,
	std::vector <double> &P,
	std::vector <double> &S,
	// in
	const std::vector<std::pair<PixIntens, double>> &Z,
	const std::vector<PixIntens> &I)
{
	size_t Ng = I.size();

	// --allocate the matrix
	P.resize (Ng);
	std::fill (P.begin(), P.end(), 0);
	S.resize (Ng);
	std::fill(S.begin(), S.end(), 0);
	N.resize(Ng);
	std::fill(N.begin(), N.end(), 0);

	double Nvp = 0;

	// --Calculate N and S
	for (auto& z : Z)
	{
		// row (grey level)
		auto inten = z.first;
		int row = -1;
		auto lower = std::lower_bound (I.begin(), I.end(), inten);	// enjoying sorted vector I
		row = int(lower - I.begin());

		// col
		int col = (int)z.second;	// 1-based
		// increment
		N[row]++;
		// --S
		double voxI = I[row],
			aveNeigI = z.second;
		S[row] += std::abs(voxI - aveNeigI);
		// --Nvp
		if (aveNeigI > 0.0)
			Nvp += 1;
	}

	// --Calculate Nvc (sum of N)
	double Nvc = 0;
	for (int i = 0; i < N.size(); i++)
		Nvc += (double) N[i];

	// --Calculate P
	for (int i = 0; i < N.size(); i++)
		P[i] = double(N[i]) / Nvc;

	return Nvp;
}

void D3_NGTDM_feature::calculate (LR& r, const Fsettings& s)
{
	// Clear variables
	clear_buffers();

	// grey-bin intensities
	int w = r.aux_image_cube.width(),
		h = r.aux_image_cube.height(),
		d = r.aux_image_cube.depth();

	SimpleCube<PixIntens> D;
	D.allocate(w, h, d);

	auto greyInfo = STNGS_NGTDM_GREYDEPTH(s);
	if (STNGS_IBSI(s))
		greyInfo = 0;

	bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

	// unique intensities (set)
	std::unordered_set<PixIntens> U (D.begin(), D.end());

	// unique intensities (sorted vector)
	if (STNGS_IBSI(s))
	{
		// ibsi expects a linspace [0 , max]
		auto max_I = *std::max_element (U.begin(), U.end());
		for (PixIntens i = 0; i <= max_I; i++)
			I.push_back(i);
	}
	else
	{
		// only unique intensities i.e. [1-based min , max]
		I.assign (U.begin(), U.end());
	}

	std::sort (I.begin(), I.end());

	// correct zero min for NGTDM
	if (I[0] == 0)
	{
		// fix unique intens
		std::for_each(I.begin(), I.end(), [](PixIntens& x) {x += 1;});
		// fix data
		std::for_each (D.begin(), D.end(), [](PixIntens& x) {x += 1;});
	}

	// zero (backround) intensity at given grey binning method
	PixIntens zeroI = matlab_grey_binning(greyInfo) ? 1 : 0;

	// is binned data informative?
	if (I.size() < 2)
	{
		_coarseness =
		_contrast =
		_busyness =
		_complexity =
		_strength = STNGS_NAN(s);
		return;
	}

	// gather zones
	int neig_r = STNGS_NGTDM_RADIUS (s);
	using AveNeighborhoodInte = std::pair<PixIntens, double>;	// Pairs of (intensity, average intensity of all 8 neighbors)
	std::vector<AveNeighborhoodInte> Z;	// list of intensity clusters (zones)
	D3_NGTDM_feature::gather_zones (Z, D, neig_r, zeroI);

	// fill the NGTD-matrix

	// --dimensions
	Ng = (int)I.size();	
	Ngp = (int)U.size();
	Nvp = D3_NGTDM_feature::calc_NGTDM (N, P, S, Z, I);

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

void D3_NGTDM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
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
	unsigned int nGrays = STNGS_NGTDM_GREYDEPTH(s);	 // former theEnvironment.get_coarse_gray_depth()
	for (int row = 0; row < D.get_height(); row++)
		for (int col = 0; col < D.get_width(); col++)
		{
			// Find a non-blank pixel 
			PixIntens pi = Nyxus::to_grayscale(D.yx(row, col), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
			if (pi == 0)
				continue;

			// Update unique intensities
			U.insert(pi);

			// Evaluate the neighborhood
			double neigsI = 0;

			int nd = 0;	// Number of dependencies

			if (D.safe(row - 1, col) && D.yx(row - 1, col) != 0)	// North
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
				nd++;
			}

			if (D.safe(row - 1, col + 1) && D.yx(row - 1, col + 1) != 0)	// North-East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col + 1), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
				nd++;
			}

			if (D.safe(row, col + 1) && D.yx(row, col + 1) != 0)	// East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row, col + 1), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
				nd++;
			}
			if (D.safe(row + 1, col + 1) && D.yx(row + 1, col + 1) != 0)	// South-East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col + 1), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
				nd++;
			}
			if (D.safe(row + 1, col) && D.yx(row + 1, col) != 0)	// South
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
				nd++;
			}
			if (D.safe(row + 1, col - 1) && D.yx(row + 1, col - 1) != 0)	// South-West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col - 1), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
				nd++;
			}
			if (D.safe(row, col - 1) && D.yx(row, col - 1) != 0)	// West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row, col - 1), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
				nd++;
			}
			if (D.safe(row - 1, col - 1) && D.yx(row - 1, col - 1) != 0)	// North-West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col - 1), r.aux_min, piRange, nGrays, STNGS_IBSI(s));
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

	Ng = (int) U.size();
	Ngp = (int) U.size();

	// --allocate the matrix
	P.resize (Ng, 0);
	S.resize (Ng, 0);
	N.resize (Ng, 0);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --Calculate N and S
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = (STNGS_IBSI(s)) ?
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

double D3_NGTDM_feature::calc_Coarseness()
{
	double sum = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum += P[i - 1] * S[i - 1];
	double retval = 1.0 / sum;
	return retval;
}

double D3_NGTDM_feature::calc_Contrast()
{
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

	// --Nvc (sum of N)
	Nvc = 0;
	for (int i = 0; i < N.size(); i++)
		Nvc += N[i];

	// --term 2
	sum = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum += S[i - 1];
	double term2 = sum / Nvc;

	double retval = term1 * term2;
	return retval;
}

double D3_NGTDM_feature::calc_Busyness()
{
	if (Ngp == 1)
		return 0.0;

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

double D3_NGTDM_feature::calc_Complexity()
{
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

double D3_NGTDM_feature::calc_Strength()
{
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

void D3_NGTDM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		D3_NGTDM_feature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}

/*static*/ void D3_NGTDM_feature::extract (LR& r, const Fsettings& s)
{
	D3_NGTDM_feature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}


