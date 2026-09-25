
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "3d_ngtdm.h"
#include "3d_ooc_volume.h"
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
	// Out-of-core NGTDM. Streams the disk-backed voxel cloud through a (2*radius+1)-plane sliding
	// window of dense grey-binned planes, reproducing calculate() exactly: unique-over-cube grey
	// levels (incl background), the "+1 if min level is 0" shift, the radius neighbourhood average
	// (which includes background neighbours), then N/S/P and the shared feature math (calc_*).
	clear_buffers();
	I.clear();

	int greyInfo = STNGS_IBSI(s) ? 0 : STNGS_NGTDM_GREYDEPTH(s);
	bool ibsi = STNGS_IBSI(s);
	PixIntens mn = r.aux_min, mx = r.aux_max;
	const PixIntens bg = TextureFeature::bin_pixel (0, mn, mx, greyInfo);

	auto bin_of = [mn, mx, greyInfo](PixIntens v) { return TextureFeature::bin_pixel (v, mn, mx, greyInfo); };

	// --- unique binned levels over the whole cube (mask + background if the bbox has any)
	Nyxus::OocBinnedVolume scan (r, bin_of, bg, 1);
	const int W = scan.width(), H = scan.height(), Dz = scan.depth();
	PixIntens maxbin = 0;
	std::set<PixIntens> U = scan.levels (/*with_background=*/ true, /*drop_zero=*/ false, maxbin);
	Ngp = (int) U.size();

	// --- grey levels I: IBSI uses a linspace [0, max]; otherwise the unique set. Then sort.
	if (ibsi)
		for (PixIntens i = 0; i <= maxbin; i++)
			I.push_back (i);
	else
		I.assign (U.begin(), U.end());
	std::sort (I.begin(), I.end());

	// --- "+1 if min level is 0" shift (applied to I and every voxel value)
	const bool shift = (!I.empty() && I[0] == 0);
	if (shift)
		for (auto& x : I) x += 1;
	const PixIntens bgv = bg + (shift ? 1 : 0);				// background value in the shifted planes
	const PixIntens zeroI = matlab_grey_binning(greyInfo) ? 1 : 0;

	// is binned data informative?
	if (I.size() < 2)
	{
		_coarseness = _contrast = _busyness = _complexity = _strength = STNGS_NAN(s);
		return;
	}

	const int rad = STNGS_NGTDM_RADIUS(s);
	Ng = (int) I.size();
	N.assign (Ng, 0);
	S.assign (Ng, 0.0);
	P.assign (Ng, 0.0);
	Nvp = 0;

	// O(1) grey-level -> matrix row, replacing the per-voxel binary search in the scan below.
	// rowLUT[v] == lower_bound(I, v) - I.begin() for every possible (shifted) binned level (the
	// max is I's last, sorted, entry), so the row is identical to the search it replaces.
	std::vector<int> rowLUT = Nyxus::ooc_row_lut (I, I.back());

	// --- the binned cube with the shift baked into its levels, streamed through a window of
	//     (2*rad+1) planes -- the Z reach of the radius neighbourhood
	const PixIntens lift = shift ? 1 : 0;
	Nyxus::OocBinnedVolume vol (r, [bin_of, lift](PixIntens v) { return bin_of (v) + lift; }, bgv, 2 * rad + 1);

	for (int c = 0; c < Dz; c++)
	{
		const std::vector<PixIntens>& cur = vol.plane (c);

		for (int y = 0; y < H; y++)
			for (int x = 0; x < W; x++)
			{
				PixIntens pi = cur[(size_t) y * W + x];
				if (pi == zeroI)			// skip background/off-mask voxels
					continue;

				double neigsI = 0;
				int nd = 0;
				for (int dz = -rad; dz <= rad; dz++)
					for (int dy = -rad; dy <= rad; dy++)
						for (int dx = -rad; dx <= rad; dx++)
						{
							if (dz == 0 && dy == 0 && dx == 0)
								continue;
							int nz = c + dz, ny = y + dy, nx = x + dx;
							if (nz < 0 || nz >= Dz || ny < 0 || ny >= H || nx < 0 || nx >= W)
								continue;
							neigsI += vol.plane (nz)[(size_t) ny * W + nx];
							nd++;
						}

				if (nd > 0)
				{
					double aveNeigI = neigsI / nd;
					int row = rowLUT[pi];
					N[row]++;
					S[row] += std::abs ((double) I[row] - aveNeigI);
					if (aveNeigI > 0.0)
						Nvp++;
				}
			}
	}

	Nvc = 0;
	for (size_t i = 0; i < N.size(); i++)
		Nvc += N[i];
	for (size_t i = 0; i < N.size(); i++)
		P[i] = (Nvc > 0) ? (double) N[i] / Nvc : 0.0;

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


