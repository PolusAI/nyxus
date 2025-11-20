#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "3d_gldm.h"
#include "../environment.h"

using namespace Nyxus;

// Define the neighborhood 
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

const static int nsh = sizeof(shifts) / sizeof(ShiftToNeighbor);

D3_GLDM_feature::D3_GLDM_feature() : FeatureMethod("D3_GLDM_feature")
{
	provide_features (D3_GLDM_feature::featureset);
}

void D3_GLDM_feature::calculate (LR& r, const Fsettings& s)
{
	clear_buffers();

	// intercept blank ROIs
	if (r.aux_min == r.aux_max)
	{
		fv_SDE =
		fv_LDE =
		fv_GLN =
		fv_DN =
		fv_DNN =
		fv_GLV =
		fv_DV =
		fv_DE =
		fv_LGLE =
		fv_HGLE =
		fv_SDLGLE =
		fv_SDHGLE =
		fv_LDLGLE =
		fv_LDHGLE = STNGS_NAN(s);

		return;
	}

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;	// Pairs of (intensity,number_of_neighbors)
	std::vector<ACluster> Z;

	// grey-bin intensities
	int w = r.aux_image_cube.width(),
		h = r.aux_image_cube.height(),
		d = r.aux_image_cube.depth();

	SimpleCube<PixIntens> D;
	D.allocate(w, h, d);

	auto greyInfo = STNGS_NGREYS(s);	// former Nyxus::theEnvironment.get_coarse_gray_depth()
	if (STNGS_IBSI(s))	// former Nyxus::theEnvironment.ibsi_compliance
		greyInfo = 0;

	bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

	// allocate intensities matrix
	if (ibsi_grey_binning(greyInfo))
	{
		auto n_ibsi_levels = *std::max_element(D.begin(), D.end());

		I.resize(n_ibsi_levels);
		for (int i = 0; i < n_ibsi_levels; i++)
			I[i] = i + 1;
	}
	else // radiomics and matlab
	{
		std::unordered_set<PixIntens> U(D.begin(), D.end());
		U.erase(0);	// discard intensity '0'
		I.assign(U.begin(), U.end());
		std::sort(I.begin(), I.end());
	}

	// zero (backround) intensity at given grey binning method
	PixIntens zeroI = matlab_grey_binning(greyInfo) ? 1 : 0;

	// Gather zones
	for (int zslice = 0; zslice < d; zslice++)
	{
		for (int row = 0; row < h; row++)
		{
			for (int col = 0; col < w; col++)
			{
				PixIntens pi = D.zyx (zslice, row, col);

				// skip background
				if (pi == zeroI)
					continue;

				// count dependencies
				int nd = 1;	 // number of dependencies

				for (int i = 0; i < nsh; i++)
				{
					if (D.safe(zslice + shifts[i].dz, row + shifts[i].dy, col + shifts[i].dx))
					{
						PixIntens neig_pi = D.zyx(zslice + shifts[i].dz, row + shifts[i].dy, col + shifts[i].dx); // neighboring voxel
						if (pi == neig_pi)
							nd++;
					}
				}

				// save the intensity's dependency
				ACluster clu = { pi, nd };
				Z.push_back(clu);
			}
		}
	}

	//==== Fill the matrix
	Ng = greyInfo == 0 ? *std::max_element(I.begin(), I.end()) : (int)I.size();
	Nd = nsh + 1;	// (z +1, 0, -1) * (xy N, NE, E, SE, S, SW, W, NW) + zero

	// --allocate the matrix
	P.allocate(Nd + 1, Ng + 1);

	// maximum dependency (non-sparse, though)
	int max_Nd = 0;

	// --iterate zones and fill the matrix
	for (auto& z : Z)
	{
		// row (grey level)
		auto inten = z.first;
		int row = -1;
		if (Environment::ibsi_compliance)
			row = inten - 1;
		else
		{
			auto lower = std::lower_bound(I.begin(), I.end(), inten);	// enjoy sorted vector 'I'
			row = int(lower - I.begin());	// intensity index in array of unique intensities 'I'
		}

		// col
		int col = z.second - 1;	// 1-based
		// increment
		auto& k = P.xy(col, row);
		k++;

		// update max referenced dependency
		max_Nd = std::max(max_Nd, col + 1);
	}

	// If not IBSI mode, adjust Nd. No need to iterate the GLDM beyond 'max_Nd'
	if (greyInfo)
		Nd = max_Nd;

	// Number of dependency zones
	Nz = 0;
	for (auto p : P)
		Nz += p;

	// Calculate features
	if (Nz == 0)
	{
		// degenerate case
		fv_SDE =
		fv_LDE =
		fv_GLN =
		fv_DN =
		fv_DNN =
		fv_GLV =
		fv_DV =
		fv_DE =
		fv_LGLE =
		fv_HGLE =
		fv_SDLGLE =
		fv_SDHGLE =
		fv_LDLGLE =
		fv_LDHGLE = STNGS_NAN(s);
	}
	else
	{
		fv_SDE = calc_SDE();
		fv_LDE = calc_LDE();
		fv_GLN = calc_GLN();
		fv_DN = calc_DN();
		fv_DNN = calc_DNN();
		fv_GLV = calc_GLV();
		fv_DV = calc_DV();
		fv_DE = calc_DE();
		fv_LGLE = calc_LGLE();
		fv_HGLE = calc_HGLE();
		fv_SDLGLE = calc_SDLGLE();
		fv_SDHGLE = calc_SDHGLE();
		fv_LDLGLE = calc_LDLGLE();
		fv_LDHGLE = calc_LDHGLE();
	}
}

void D3_GLDM_feature::clear_buffers()
{
	bad_roi_data = false;
	int Ng = 0;
	int Nd = 0;
	int Nz = 0;

	double sum_p = 0;

	P.clear();
}

void D3_GLDM_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void D3_GLDM_feature::osized_calculate(LR& r, const Fsettings& s, ImageLoader&)
{
	calculate (r, s);
}

void D3_GLDM_feature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature3D::GLDM_SDE][0] = fv_SDE;
	fvals[(int)Feature3D::GLDM_LDE][0] = fv_LDE;
	fvals[(int)Feature3D::GLDM_GLN][0] = fv_GLN;
	fvals[(int)Feature3D::GLDM_DN][0] = fv_DN;
	fvals[(int)Feature3D::GLDM_DNN][0] = fv_DNN;
	fvals[(int)Feature3D::GLDM_GLV][0] = fv_GLV;
	fvals[(int)Feature3D::GLDM_DV][0] = fv_DV;
	fvals[(int)Feature3D::GLDM_DE][0] = fv_DE;
	fvals[(int)Feature3D::GLDM_LGLE][0] = fv_LGLE;
	fvals[(int)Feature3D::GLDM_HGLE][0] = fv_HGLE;
	fvals[(int)Feature3D::GLDM_SDLGLE][0] = fv_SDLGLE;
	fvals[(int)Feature3D::GLDM_SDHGLE][0] = fv_SDHGLE;
	fvals[(int)Feature3D::GLDM_LDLGLE][0] = fv_LDLGLE;
	fvals[(int)Feature3D::GLDM_LDHGLE][0] = fv_LDHGLE;
}

// 1. Small Dependence Emphasis(SDE)
double D3_GLDM_feature::calc_SDE()
{
	double sum = 0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
			sum += P.matlab(i, j) / (double(j) * double(j));
	}
	double retval = sum / double(Nz);
	return retval;
}

// 2. Large Dependence Emphasis (LDE)
double D3_GLDM_feature::calc_LDE()
{
	double sum = 0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
			sum += P.matlab(i, j) * (double(j) * double(j));
	}
	double retval = sum / double(Nz);
	return retval;
}

// 3. Gray Level Non-Uniformity (GLN)
double D3_GLDM_feature::calc_GLN()
{
	std::vector<double> si(Ng + 1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			si[i - 1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		f += si[i - 1] * si[i - 1];
	}

	double retval = f / double(Nz);
	return retval;
}

// 4. Dependence Non-Uniformity (DN)
double D3_GLDM_feature::calc_DN()
{
	std::vector<double> sj(Nd + 1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			sj[j - 1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int j = 1; j <= Nd; j++)
	{
		f += sj[j - 1] * sj[j - 1];
	}

	double retval = f / double(Nz);
	return retval;
}
// 5. Dependence Non-Uniformity Normalized (DNN)
double D3_GLDM_feature::calc_DNN()
{
	std::vector<double> sj(Nd + 1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			sj[j - 1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int j = 1; j <= Nd; j++)
	{
		f += sj[j - 1] * sj[j - 1];
	}
	double retval = f / (double(Nz) * double(Nz));
	return retval;
}

// 6. Gray Level Variance (GLV)
double D3_GLDM_feature::calc_GLV()
{
	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Nd; j++)
		{
			mu += P.matlab(i, j) / double(Nz) * inten;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Nd; j++)
		{
			double mu2 = (inten - mu) * (inten - mu);
			f += P.matlab(i, j) / double(Nz) * mu2;
		}
	}
	return f;
}

// 7. Dependence Variance (DV)
double D3_GLDM_feature::calc_DV()
{
	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			mu += P.matlab(i, j) / double(Nz) * j;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			double d = double(j) - mu,
				mu2 = d * d;
			f += P.matlab(i, j) / double(Nz) * mu2;
		}
	}
	return f;
}

// 8. Dependence Entropy (DE)
double D3_GLDM_feature::calc_DE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			double entrTerm = fast_log10(P.matlab(i, j) / double(Nz) + EPS) / LOG10_2;
			f += P.matlab(i, j) / double(Nz) * entrTerm;
		}
	}
	double retval = -f;
	return retval;
}

// 9. Low Gray Level Emphasis (LGLE)
double D3_GLDM_feature::calc_LGLE()
{
	double sum_i = 0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten2 = (double)I[i - 1];
		inten2 *= inten2;
		double sum_j = 0;
		for (int j = 1; j <= Nd; j++)
			sum_j += P.matlab(i, j);
		sum_i += sum_j / inten2;
	}
	double retval = sum_i / double(Nz);

	return retval;
}

// 10. High Gray Level Emphasis (HGLE)
double D3_GLDM_feature::calc_HGLE()
{
	std::vector<double> si(Ng + 1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			si[i - 1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		f += si[i - 1] * inten * inten;
	}

	double retval = f / double(Nz);
	return retval;
}

// 11. Small Dependence Low Gray Level Emphasis (SDLGLE)
double D3_GLDM_feature::calc_SDLGLE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) / double(inten * inten * j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 12. Small Dependence High Gray Level Emphasis (SDHGLE)
double D3_GLDM_feature::calc_SDHGLE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * (inten * inten) / double(j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 13. Large Dependence Low Gray Level Emphasis (LDLGLE)
double D3_GLDM_feature::calc_LDLGLE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * double(j * j) / (inten * inten);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 14. Large Dependence High Gray Level Emphasis (LDHGLE)
double D3_GLDM_feature::calc_LDHGLE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * double(inten * inten * j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

void D3_GLDM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		D3_GLDM_feature gldm;
		gldm.calculate (r, s);
		gldm.save_value (r.fvals);
	}
}

/*static*/ void D3_GLDM_feature::extract (LR& r, const Fsettings& s)
{
	D3_GLDM_feature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}
