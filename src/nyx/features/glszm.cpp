#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "glszm.h"
#include "../environment.h"
#include "../helpers/timing.h"

#ifdef USE_GPU
	#include "../gpu/glszm.cuh"
#endif

using namespace Nyxus;

int GLSZMFeature::n_levels = 0;

void GLSZMFeature::invalidate()
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
		fv_LAHGLE = theEnvironment.nan_substitute;
}

GLSZMFeature::GLSZMFeature() : FeatureMethod("GLSZMFeature")
{
	provide_features (GLSZMFeature::featureset);
}

void GLSZMFeature::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {} // Not supporting

void GLSZMFeature::osized_calculate (LR& r, ImageLoader&)
{
	clear_buffers();

	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	// Width of the intensity - zone area matrix 
	int maxZoneArea = 0;

	// Create an image matrix for this ROI
	WriteImageMatrix_nontriv M ("GLSZMFeature-osized_calculate-M", r.label);
	M.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);

	// Helpful temps
	auto height = M.get_height(), 
		width = M.get_width();

	// Copy the image matrix
	WriteImageMatrix_nontriv D ("GLSZMFeature-osized_calculate-D", r.label);
	D.allocate (width, height, 0);
	D.copy (M);

	// Squeeze the intensity range
	PixIntens piRange = r.aux_max - r.aux_min;		// Prepare ROI's intensity range
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();

	for (size_t i = 0; i < D.size(); i++)
		D.set_at(i, Nyxus::to_grayscale(D[i], r.aux_min, piRange, nGrays, Environment::ibsi_compliance));

	// Number of zones
	const int VISITED = -1;
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			// Find a non-blank pixel
			auto pi = D.yx(row, col);
			if (pi == 0 || int(pi) == VISITED)
				continue;

			// Found a gray pixel. Find same-intensity neighbourhood of it.
			std::vector<std::tuple<int, int>> history;
			int x = col, y = row;
			int zoneArea = 1;
			D.set_at(y, x, VISITED);

			for (;;)
			{
				if (D.safe(y, x + 1) && D.yx(y, x + 1) != VISITED && D.yx(y, x + 1) == pi)
				{
					D.set_at(y, x + 1, VISITED);
					zoneArea++;

					// Remember this pixel
					history.push_back({ x,y });
					// Advance
					x = x + 1;
					// Proceed
					continue;
				}
				if (D.safe(y + 1, x + 1) && D.yx(y + 1, x + 1) != VISITED && D.yx(y + 1, x + 1) == pi)
				{
					D.set_at(y + 1, x + 1, VISITED);
					zoneArea++;

					history.push_back({ x,y });
					x = x + 1;
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x) && D.yx(y + 1, x) != VISITED && D.yx(y + 1, x) == pi)
				{
					D.set_at(y + 1, x, VISITED);
					zoneArea++;

					history.push_back({ x,y });
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x - 1) && D.yx(y + 1, x - 1) != VISITED && D.yx(y + 1, x - 1) == pi)
				{
					D.set_at(y + 1, x - 1, VISITED);
					zoneArea++;

					history.push_back({ x,y });
					x = x - 1;
					y = y + 1;
					continue;
				}

				// Return from the branch
				if (history.size() > 0)
				{
					// Recollect the coordinate where we diverted from
					std::tuple<int, int> prev = history[history.size() - 1];
					x = std::get<0>(prev);
					y = std::get<1>(prev);
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

	// count non-zero pixels
	int count = 0;
	for (auto i = 0; i < M.size(); i++)
	{
		auto px = M[i];
		if (px != 0)
			++count;
	}

	//==== Fill the SZ-matrix

	Ng = (int)U.size();
	Ns = maxZoneArea;
	Nz = (int)Z.size();
	Np = count;

	// --Set to vector to be able to know each intensity's index
	I.assign(U.begin(), U.end());
	std::sort(I.begin(), I.end());

	// --allocate the matrix
	P.allocate(Ns, Ng);

	// --iterate zones and fill the matrix
	int i = 0;
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = (Environment::ibsi_compliance) ? z.first - 1 : int(iter - I.begin());
		// col
		int col = z.second - 1;	// 0-based => -1
		auto& k = P.xy(col, row);
		k++;
	}

	sum_p = 0;
	for (int i = 1; i <= Ng; ++i) {
		for (int j = 1; j <= Ns; ++j) {
			sum_p += P.matlab(i, j);
		}
	}

}

void GLSZMFeature::calculate(LR& r)
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

	// Copy the image matrix (SZ-matrix algorithm needs this copy)
	ImageMatrix M;
	M.allocate (r.aux_image_matrix.width, r.aux_image_matrix.height);
	pixData & D = M.WriteablePixels();

	// Squeeze the intensity range
	auto greyInfo = theEnvironment.get_coarse_gray_depth();
	if (Nyxus::theEnvironment.ibsi_compliance)
		greyInfo = 0;
	auto& imR = r.aux_image_matrix.ReadablePixels();
	bin_intensities (D, imR, r.aux_min, r.aux_max, greyInfo);

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

{ //STOPWATCH("sz01/sz01/T/#raw", "\t=");

	// Number of zones
	const int VISITED = -1;
	for (int row = 0; row < M.height; row++)
	{
		for (int col = 0; col < M.width; col++)
		{
			// Find a non-blank pixel
			auto pi = D.yx(row, col);
			if (pi == 0 || int(pi) == VISITED)
				continue;

			// Found a gray pixel. Find same-intensity neighbourhood of it.
			std::vector<std::tuple<int, int>> history;
			int x = col, y = row;
			int zoneArea = 1;
			D.yx(y, x) = VISITED;

			for (;;)
			{
				if (D.safe(y, x + 1) && D.yx(y, x + 1) != VISITED && D.yx(y, x + 1) == pi)
				{
					D.yx(y, x + 1) = VISITED;
					zoneArea++;

					// Remember this pixel
					history.push_back({ x,y });
					// Advance
					x = x + 1;
					// Proceed
					continue;
				}
				if (D.safe(y + 1, x + 1) && D.yx(y + 1, x + 1) != VISITED && D.yx(y + 1, x + 1) == pi)
				{
					D.yx(y + 1, x + 1) = VISITED;
					zoneArea++;

					history.push_back({ x,y });
					x = x + 1;
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x) && D.yx(y + 1, x) != VISITED && D.yx(y + 1, x) == pi)
				{
					D.yx(y + 1, x) = VISITED;
					zoneArea++;

					history.push_back({ x,y });
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x - 1) && D.yx(y + 1, x - 1) != VISITED && D.yx(y + 1, x - 1) == pi)
				{
					D.yx(y + 1, x - 1) = VISITED;
					zoneArea++;

					history.push_back({ x,y });
					x = x - 1;
					y = y + 1;
					continue;
				}

				// Return from the branch
				if (history.size() > 0)
				{
					// Recollect the coordinate where we diverted from
					std::tuple<int, int> prev = history[history.size() - 1];
					x = std::get<0>(prev);
					y = std::get<1>(prev);
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

}//t

	// count non-zero pixels
	int count = 0;
	for (const auto& px: M.ReadablePixels()) 
	{
		if(px != 0) 
			++count;
	}

	//==== Fill the SZ-matrix

	auto height = M.height;
	auto width = M.width;

	Ng = Environment::ibsi_compliance ? *std::max_element(I.begin(), I.end()) : I.size();
	Ns = height * width;
	Nz = (int)Z.size();
	Np = count;

	// --allocate the matrix
	P.allocate (Ns, Ng);
	
{ //STOPWATCH("sz02/sz02/T/#raw", "\t=");

	// --iterate zones and fill the matrix
	int i = 0;
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = (Environment::ibsi_compliance) ? z.first-1 : int (iter - I.begin());
		// col
		int col = z.second - 1;	// 0-based => -1
		auto & k = P.xy(col, row);
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
		if (!NyxusGpu::GLSZMfeature_calc (
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
			std::cerr << "ERROR: GLSZMfeature_calc_sums_of_P failed \n";
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
		// features are calculated in GLSZMfeature_calc_sums_of_P
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

void GLSZMFeature::calc_sums_of_P()
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
	si.resize (Ng+1);
	std::fill (si.begin(), si.end(), 0.0);

	// Aggregate by grayscale level
	for (int i = 1; i <= Ng; ++i)
	{
		double inten = (double) I[i-1];
		double sum = 0;
		for (int j = 1; j <= Ns; ++j)
		{
			double p = P.matlab (i,j);
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
		si [i] = sum;
	}

	// Reset by-position counters
	sj.clear();
	sj.resize (Ns+1);
	std::fill (sj.begin(), sj.end(), 0.0);
	for (int j = 1; j <= Ns; ++j)
	{
		double sum = 0;
		for (int i = 1; i <= Ng; ++i)
			sum += P.matlab (i,j);
		sj[j] = sum;
	}
}

bool GLSZMFeature::need (Nyxus::Feature2D f)
{
	return theFeatureSet.isEnabled (f);
}

void GLSZMFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::GLSZM_SAE][0] = fv_SAE;
	fvals[(int)Feature2D::GLSZM_LAE][0] = fv_LAE;
	fvals[(int)Feature2D::GLSZM_GLN][0] = fv_GLN;
	fvals[(int)Feature2D::GLSZM_GLNN][0] = fv_GLNN;
	fvals[(int)Feature2D::GLSZM_SZN][0] = fv_SZN;
	fvals[(int)Feature2D::GLSZM_SZNN][0] = fv_SZNN;
	fvals[(int)Feature2D::GLSZM_ZP][0] = fv_ZP;
	fvals[(int)Feature2D::GLSZM_GLV][0] = fv_GLV;
	fvals[(int)Feature2D::GLSZM_ZV][0] = fv_ZV;
	fvals[(int)Feature2D::GLSZM_ZE][0] = fv_ZE;
	fvals[(int)Feature2D::GLSZM_LGLZE][0] = fv_LGLZE;
	fvals[(int)Feature2D::GLSZM_HGLZE][0] = fv_HGLZE;
	fvals[(int)Feature2D::GLSZM_SALGLE][0] = fv_SALGLE;
	fvals[(int)Feature2D::GLSZM_SAHGLE][0] = fv_SAHGLE;
	fvals[(int)Feature2D::GLSZM_LALGLE][0] = fv_LALGLE;
	fvals[(int)Feature2D::GLSZM_LAHGLE][0] = fv_LAHGLE;
}

// 1. Small Area Emphasis
double GLSZMFeature::calc_SAE()
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
double GLSZMFeature::calc_LAE()
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
double GLSZMFeature::calc_GLN()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double x = si[i];
		f += x*x;
	}

	double retval = f / sum_p;
	return retval;
}

// 4. Gray Level Non - Uniformity Normalized
double GLSZMFeature::calc_GLNN()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;

	for (int i = 1; i <= Ng; i++)
	{
		double x = si[i];
		f += x*x;
	}

	double retval = f / double(sum_p * sum_p);
	return retval;
}

// 5. Size - Zone Non - Uniformity
double GLSZMFeature::calc_SZN()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		double x = sj[j];
		f += x*x;
	}

	double retval = f / sum_p;
	return retval;
}

// 6. Size - Zone Non - Uniformity Normalized
double GLSZMFeature::calc_SZNN()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		double x = sj[j];
		f += x*x;
	}

	double retval = f / double(sum_p * sum_p);
	return retval;
}

// 7. Zone Percentage
double GLSZMFeature::calc_ZP()
{
	double retval = sum_p / double(Np);
	return retval;
}

// 8. Gray Level Variance
double GLSZMFeature::calc_GLV()
{
	// Calculate feature. 'mu_GLV' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		for (int j = 1; j <= Ns; j++)
		{
			double d2 = (inten - mu_GLV) * (inten - mu_GLV);
			f += P.matlab(i, j) / sum_p * d2;
		}
	}
	return f;
}

// 9. Zone Variance
double GLSZMFeature::calc_ZV()
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
double GLSZMFeature::calc_ZE()
{
	// Calculate feature. 'f_ZE' is expected to have been initialized in calc_sums_of_P()
	double retval = -f_ZE;
	return retval;
}

// 11. Low Gray Level Zone Emphasis
double GLSZMFeature::calc_LGLZE()
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
double GLSZMFeature::calc_HGLZE()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		f += si[i] * (inten * inten);
	}

	double retval = f / sum_p;
	return retval;
}

// 13. Small Area Low Gray Level Emphasis
double GLSZMFeature::calc_SALGLE()
{
	// Calculate feature. 'f_SALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SALGLE / sum_p;
	return retval;
}

// 14. Small Area High Gray Level Emphasis
double GLSZMFeature::calc_SAHGLE()
{
	// Calculate feature. 'f_SAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SAHGLE / sum_p;
	return retval;
}

// 15. Large Area Low Gray Level Emphasis
double GLSZMFeature::calc_LALGLE()
{
	// Calculate feature. 'f_LALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LALGLE / sum_p;
	return retval;
}

// 16. Large Area High Gray Level Emphasis
double GLSZMFeature::calc_LAHGLE()
{
	// Calculate feature. 'f_LAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LAHGLE / sum_p;
	return retval;
}

void GLSZMFeature::extract (LR& r)
{
	GLSZMFeature f;
	f.calculate(r);
	f.save_value(r.fvals);
}

void GLSZMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		extract (r);
	}
}

