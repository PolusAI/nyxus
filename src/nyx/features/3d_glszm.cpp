#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "3d_glszm.h"
#include "../environment.h"
#include "../helpers/timing.h"

#ifdef USE_GPU
#include "../gpu/glszm.cuh"
#endif

using namespace Nyxus;

int D3_GLSZM_feature::n_levels = 0;

void D3_GLSZM_feature::invalidate()
{
	fv_SAE =
		fv_LAE =
		fv_GLN =
		fv_GLNN =
		fv_SZN =
		fv_SZNN =
		fv_ZP =
		fv_GLV =
		fv_ZV =
		fv_ZE =
		fv_LGLZE =
		fv_HGLZE =
		fv_SALGLE =
		fv_SAHGLE =
		fv_LALGLE =
		fv_LAHGLE = theEnvironment.resultOptions.noval();
}

D3_GLSZM_feature::D3_GLSZM_feature() : FeatureMethod("D3_GLSZM_feature")
{
	provide_features(D3_GLSZM_feature::featureset);
}

void D3_GLSZM_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting

void D3_GLSZM_feature::osized_calculate(LR& r, ImageLoader&)
{
	calculate(r);
}

void D3_GLSZM_feature::calculate(LR& r)
{
	clear_buffers();

	// intercept blank ROIs (equal intensity)
	if (r.aux_min == r.aux_max)
	{
		invalidate();
		return;
	}

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	// Width of the intensity - zone area matrix 
	int maxZoneArea = 0;

	// grey-bin intensities
	int w = r.aux_image_cube.width(),
		h = r.aux_image_cube.height(),
		d = r.aux_image_cube.depth();

	SimpleCube<PixIntens> D;
	D.allocate(w, h, d);

	auto greyInfo = Nyxus::theEnvironment.get_coarse_gray_depth();
	auto greyInfo_localFeature = D3_GLSZM_feature::n_levels;
	if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
		greyInfo = greyInfo_localFeature;
	if (Nyxus::theEnvironment.ibsi_compliance)
		greyInfo = 0;

	bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

	// allocate the matrix of intensities
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

	{ //STOPWATCH("sz01/sz01/T/#raw", "\t=");

		// zero (backround) intensity at given grey binning method
		PixIntens zeroI = matlab_grey_binning(greyInfo) ? 1 : 0;
		
		// Number of zones
		const int VISITED = -1;
		for (int zslice = 0; zslice < d; zslice++)
		{
			for (int row = 0; row < h; row++)
			{
				for (int col = 0; col < w; col++)
				{
					// Find a non-blank pixel
					auto pi = D.zyx (zslice, row, col);
					if (pi == zeroI || int(pi) == VISITED)
						continue;

					// Found a gray pixel. Find same-intensity neighbourhood of it.
					std::vector<std::tuple<int, int, int>> history; // dimension order: x,y,z
					int x = col, y = row, z = zslice;
					int zoneArea = 1;
					D.zyx (z, y, x) = VISITED;

					for (;;)
					{
						// dz == 0
						if (D.safe(z, y, x + 1) && D.zyx(z, y, x + 1) != VISITED && D.zyx(z, y, x + 1) == pi)
						{
							D.zyx (z, y, x + 1) = VISITED;
							zoneArea++;

							// Remember this pixel
							history.push_back ({ x,y,z });
							// Advance
							x = x + 1;
							// Proceed
							continue;
						}
						if (D.safe(z, y + 1, x + 1) && D.zyx(z, y + 1, x + 1) != VISITED && D.zyx(z, y + 1, x + 1) == pi)
						{
							D.zyx (z, y + 1, x + 1) = VISITED;
							zoneArea++;

							history.push_back ({ x,y,z });
							x = x + 1;
							y = y + 1;
							continue;
						}
						if (D.safe(z, y + 1, x) && D.zyx(z, y + 1, x) != VISITED && D.zyx(z, y + 1, x) == pi)
						{
							D.zyx (z, y + 1, x) = VISITED;
							zoneArea++;

							history.push_back ({ x,y,z });
							y = y + 1;
							continue;
						}
						if (D.safe(z, y + 1, x - 1) && D.zyx(z, y + 1, x - 1) != VISITED && D.zyx(z, y + 1, x - 1) == pi)
						{
							D.zyx (z, y + 1, x - 1) = VISITED;
							zoneArea++;

							history.push_back ({ x,y,z });
							x = x - 1;
							y = y + 1;
							continue;
						}

						// dz == +1
						if (D.safe(z+1, y, x) && D.zyx(z+1, y, x) != VISITED && D.zyx(z+1, y, x) == pi)
						{
							D.zyx (z+1, y, x) = VISITED;
							zoneArea++;

							// Remember this pixel
							history.push_back ({ x,y,z });
							// Advance
							z = z + 1;
							// Proceed
							continue;
						}
						if (D.safe(z+1, y, x+1) && D.zyx(z+1, y, x+1) != VISITED && D.zyx(z+1, y, x+1) == pi)
						{
							D.zyx (z+1, y, x+1) = VISITED;
							zoneArea++;

							// Remember this pixel
							history.push_back ({ x,y,z });
							// Advance
							x = x + 1;
							z = z + 1;
							// Proceed
							continue;
						}
						if (D.safe(z+1, y+1, x+1) && D.zyx(z+1, y+1, x+1) != VISITED && D.zyx(z+1, y+1, x+1) == pi)
						{
							D.zyx (z+1, y+1, x+1) = VISITED;
							zoneArea++;

							history.push_back ({ x,y,z });
							x = x + 1;
							y = y + 1;
							z = z + 1;
							continue;
						}
						if (D.safe(z+1, y+1, x) && D.zyx(z+1, y+1, x) != VISITED && D.zyx(z+1, y+1, x) == pi)
						{
							D.zyx (z+1, y+1, x) = VISITED;
							zoneArea++;

							history.push_back ({ x,y,z });
							y = y + 1;
							z = z + 1;
							continue;
						}
						if (D.safe(z+1, y+1, x-1) && D.zyx(z+1, y+1, x-1) != VISITED && D.zyx(z+1, y+1, x-1) == pi)
						{
							D.zyx (z+1, y+1, x-1) = VISITED;
							zoneArea++;

							history.push_back ({ x,y,z });
							x = x - 1;
							y = y + 1;
							z = z + 1;
							continue;
						}

						// Return from the branch
						if (history.size() > 0)
						{
							// Recollect the coordinate where we diverted from
							std::tuple<int, int, int> prev = history[history.size() - 1];
							x = std::get<0>(prev);
							y = std::get<1>(prev);
							z = std::get<2>(prev);
							history.pop_back();
							continue;
						}

						// We are done exploring this cluster
						break;
					}

					// Done scanning a cluster. Perform 3 actions:
					// --1
					U.insert(pi);

					// --2
					maxZoneArea = std::max(maxZoneArea, zoneArea);

					// --3
					ACluster clu = { pi, zoneArea };
					Z.push_back(clu);
				}
			}
		}

	}//t

	// count non-zero pixels
	int count = 0;
	for (const auto& px : r.aux_image_cube)	// 'D' is dirty so use 'r.aux_image_cube'
	{
		if (px != 0)
			++count;
	}

	//==== Fill the SZ-matrix

	auto height = h;
	auto width = w;

	Ng = Environment::ibsi_compliance ? *std::max_element(I.begin(), I.end()) : I.size();
	Ns = height * width;
	Nz = (int)Z.size();
	Np = count;

	// --allocate the matrix
	P.allocate(Ns, Ng);

	{ //STOPWATCH("sz02/sz02/T/#raw", "\t=");

		// --iterate zones and fill the matrix
		int i = 0;
		for (auto& z : Z) // 'z' stands for zone, not z-slice
		{
			// row of P-matrix
			auto iter = std::find(I.begin(), I.end(), z.first);
			int row = (Environment::ibsi_compliance) ? z.first - 1 : int(iter - I.begin());
			// column of P-matrix
			int col = z.second - 1;	// 0-based => -1
			auto& k = P.xy(col, row);
			k++;
		}

		// Non-informative matrix?
		sum_p = 0;
		for (int i = 1; i <= Ng; ++i)
		{
			for (int j = 1; j <= Ns; ++j)
			{
				sum_p += P.matlab(i, j);
			}
		}

		if (sum_p == 0)
		{
			invalidate();
			return;
		}

	}//sz02

	{ //STOPWATCH("sz03/sz03/T/#raw", "\t=");

		// Precalculate sums of P
#ifdef USE_GPU
		if (theEnvironment.using_gpu())
		{
			if (! NyxusGpu::GLSZMfeature_calc(
				// out
				fv_SAE,
				fv_LAE,
				fv_GLN,
				fv_GLNN,
				fv_SZN,
				fv_SZNN,
				fv_ZP,
				fv_GLV,
				fv_ZV,
				fv_ZE,
				fv_LGLZE,
				fv_HGLZE,
				fv_SALGLE,
				fv_SAHGLE,
				fv_LALGLE,
				fv_LAHGLE,
				// in
				Ng, Ns, I.data(), P.data(), sum_p, Np, EPS))
			{
				std::cerr << "ERROR: D3_GLSZM_feature_calc_sums_of_P failed \n";
				invalidate();
				return;
			}
		}
		else
		{
			calc_sums_of_P();
		}
#else
		calc_sums_of_P();
#endif

	}//sz03

	{ //STOPWATCH("sz04/sz04/T/#raw", "\t=");

		// Calculate features
#ifdef USE_GPU
		if (theEnvironment.using_gpu())
		{
			// features are calculated in D3_GLSZM_feature_calc_sums_of_P
		}
		else
		{
			fv_SAE = calc_SAE();
			fv_LAE = calc_LAE();
			fv_GLN = calc_GLN();
			fv_GLNN = calc_GLNN();
			fv_SZN = calc_SZN();
			fv_SZNN = calc_SZNN();
			fv_ZP = calc_ZP();
			fv_GLV = calc_GLV();
			fv_ZV = calc_ZV();
			fv_ZE = calc_ZE();
			fv_LGLZE = calc_LGLZE();
			fv_HGLZE = calc_HGLZE();
			fv_SALGLE = calc_SALGLE();
			fv_SAHGLE = calc_SAHGLE();
			fv_LALGLE = calc_LALGLE();
			fv_LAHGLE = calc_LAHGLE();
		}
#else
		fv_SAE = calc_SAE();
		fv_LAE = calc_LAE();
		fv_GLN = calc_GLN();
		fv_GLNN = calc_GLNN();
		fv_SZN = calc_SZN();
		fv_SZNN = calc_SZNN();
		fv_ZP = calc_ZP();
		fv_GLV = calc_GLV();
		fv_ZV = calc_ZV();
		fv_ZE = calc_ZE();
		fv_LGLZE = calc_LGLZE();
		fv_HGLZE = calc_HGLZE();
		fv_SALGLE = calc_SALGLE();
		fv_SAHGLE = calc_SAHGLE();
		fv_LALGLE = calc_LALGLE();
		fv_LAHGLE = calc_LAHGLE();
#endif
	}//sz04
}

void D3_GLSZM_feature::calc_sums_of_P()
{
	// Zero specialized sums
	f_LAHGLE = 0;
	f_LALGLE = 0;
	f_SAHGLE = 0;
	f_SALGLE = 0;
	f_ZE = 0;
	mu_GLV = 0;
	mu_ZV = 0;

	// Reset by-level counters
	si.clear();
	si.resize(Ng + 1);
	std::fill(si.begin(), si.end(), 0.0);

	// Aggregate by grayscale level
	for (int i = 1; i <= Ng; ++i)
	{
		double inten = (double)I[i - 1];
		double sum = 0;
		for (int j = 1; j <= Ns; ++j)
		{
			double p = P.matlab(i, j);
			sum += p;

			// Once we're iterating matrix P, let's compute specialized sums
			double i2 = inten * inten,
				j2 = double(j) * double(j);

			f_LAHGLE += p * i2 * j2;
			f_LALGLE += p * j2 / i2;
			f_SAHGLE += p * i2 / j2;
			f_SALGLE += p / (i2 * j2);

			double entrTerm = fast_log10(p / sum_p + EPS) / LOG10_2;
			f_ZE += p / sum_p * entrTerm;

			mu_ZV += p / sum_p * double(j);
			mu_GLV += p / sum_p * double(inten);
		}
		si[i] = sum;
	}

	// Reset by-position counters
	sj.clear();
	sj.resize(Ns + 1);
	std::fill(sj.begin(), sj.end(), 0.0);
	for (int j = 1; j <= Ns; ++j)
	{
		double sum = 0;
		for (int i = 1; i <= Ng; ++i)
			sum += P.matlab(i, j);
		sj[j] = sum;
	}
}

bool D3_GLSZM_feature::need (Nyxus::Feature3D f)
{
	return theFeatureSet.isEnabled(f);
}

void D3_GLSZM_feature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature3D::GLSZM_SAE][0] = fv_SAE;
	fvals[(int)Feature3D::GLSZM_LAE][0] = fv_LAE;
	fvals[(int)Feature3D::GLSZM_GLN][0] = fv_GLN;
	fvals[(int)Feature3D::GLSZM_GLNN][0] = fv_GLNN;
	fvals[(int)Feature3D::GLSZM_SZN][0] = fv_SZN;
	fvals[(int)Feature3D::GLSZM_SZNN][0] = fv_SZNN;
	fvals[(int)Feature3D::GLSZM_ZP][0] = fv_ZP;
	fvals[(int)Feature3D::GLSZM_GLV][0] = fv_GLV;
	fvals[(int)Feature3D::GLSZM_ZV][0] = fv_ZV;
	fvals[(int)Feature3D::GLSZM_ZE][0] = fv_ZE;
	fvals[(int)Feature3D::GLSZM_LGLZE][0] = fv_LGLZE;
	fvals[(int)Feature3D::GLSZM_HGLZE][0] = fv_HGLZE;
	fvals[(int)Feature3D::GLSZM_SALGLE][0] = fv_SALGLE;
	fvals[(int)Feature3D::GLSZM_SAHGLE][0] = fv_SAHGLE;
	fvals[(int)Feature3D::GLSZM_LALGLE][0] = fv_LALGLE;
	fvals[(int)Feature3D::GLSZM_LAHGLE][0] = fv_LAHGLE;
}

// 1. Small Area Emphasis
double D3_GLSZM_feature::calc_SAE()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		f += sj[j] / (j * j);
	}
	double retval = f / sum_p;
	return retval;
}

// 2. Large Area Emphasis
double D3_GLSZM_feature::calc_LAE()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		f += sj[j] * (j * j);
	}
	double retval = f / sum_p;
	return retval;
}

// 3. Gray Level Non - Uniformity
double D3_GLSZM_feature::calc_GLN()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double x = si[i];
		f += x * x;
	}

	double retval = f / sum_p;
	return retval;
}

// 4. Gray Level Non - Uniformity Normalized
double D3_GLSZM_feature::calc_GLNN()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;

	for (int i = 1; i <= Ng; i++)
	{
		double x = si[i];
		f += x * x;
	}

	double retval = f / double(sum_p * sum_p);
	return retval;
}

// 5. Size - Zone Non - Uniformity
double D3_GLSZM_feature::calc_SZN()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		double x = sj[j];
		f += x * x;
	}

	double retval = f / sum_p;
	return retval;
}

// 6. Size - Zone Non - Uniformity Normalized
double D3_GLSZM_feature::calc_SZNN()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		double x = sj[j];
		f += x * x;
	}

	double retval = f / double(sum_p * sum_p);
	return retval;
}

// 7. Zone Percentage
double D3_GLSZM_feature::calc_ZP()
{
	double retval = sum_p / double(Np);
	return retval;
}

// 8. Gray Level Variance
double D3_GLSZM_feature::calc_GLV()
{
	// Calculate feature. 'mu_GLV' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Ns; j++)
		{
			double d2 = (inten - mu_GLV) * (inten - mu_GLV);
			f += P.matlab(i, j) / sum_p * d2;
		}
	}
	return f;
}

// 9. Zone Variance
double D3_GLSZM_feature::calc_ZV()
{
	// Calculate feature. 'mu_ZV' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			double mu2 = (double(j) - mu_ZV) * (double(j) - mu_ZV);
			f += P.matlab(i, j) / sum_p * mu2;
		}
	}
	return f;
}

// 10. Zone Entropy
double D3_GLSZM_feature::calc_ZE()
{
	// Calculate feature. 'f_ZE' is expected to have been initialized in calc_sums_of_P()
	double retval = -f_ZE;
	return retval;
}

// 11. Low Gray Level Zone Emphasis
double D3_GLSZM_feature::calc_LGLZE()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		f += si[i] / (inten * inten);
	}

	double retval = f / sum_p;
	return retval;
}

// 12. High Gray Level Zone Emphasis
double D3_GLSZM_feature::calc_HGLZE()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		f += si[i] * (inten * inten);
	}

	double retval = f / sum_p;
	return retval;
}

// 13. Small Area Low Gray Level Emphasis
double D3_GLSZM_feature::calc_SALGLE()
{
	// Calculate feature. 'f_SALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SALGLE / sum_p;
	return retval;
}

// 14. Small Area High Gray Level Emphasis
double D3_GLSZM_feature::calc_SAHGLE()
{
	// Calculate feature. 'f_SAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SAHGLE / sum_p;
	return retval;
}

// 15. Large Area Low Gray Level Emphasis
double D3_GLSZM_feature::calc_LALGLE()
{
	// Calculate feature. 'f_LALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LALGLE / sum_p;
	return retval;
}

// 16. Large Area High Gray Level Emphasis
double D3_GLSZM_feature::calc_LAHGLE()
{
	// Calculate feature. 'f_LAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LAHGLE / sum_p;
	return retval;
}

void D3_GLSZM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_GLSZM_feature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

/*static*/ void D3_GLSZM_feature::extract(LR& r)
{
	D3_GLSZM_feature f;
	f.calculate(r);
	f.save_value(r.fvals);
}

