#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "glszm.h"
#include "../environment.h"

GLSZMFeature::GLSZMFeature() : FeatureMethod("GLSZMFeature")
{
	provide_features({ 
		GLSZM_SAE,
		GLSZM_LAE,
		GLSZM_GLN,
		GLSZM_GLNN,
		GLSZM_SZN,
		GLSZM_SZNN,
		GLSZM_ZP,
		GLSZM_GLV,
		GLSZM_ZV,
		GLSZM_ZE,
		GLSZM_LGLZE,
		GLSZM_HGLZE,
		GLSZM_SALGLE,
		GLSZM_SAHGLE,
		GLSZM_LALGLE,
		GLSZM_LAHGLE });
}

void GLSZMFeature::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {} // Not suporting

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
					D.set_at (y + 1, x + 1, VISITED);
					zoneArea++;

					history.push_back({ x,y });
					x = x + 1;
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x) && D.yx(y + 1, x) != VISITED && D.yx(y + 1, x) == pi)
				{
					D.set_at (y + 1, x, VISITED);
					zoneArea++;

					history.push_back({ x,y });
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x - 1) && D.yx(y + 1, x - 1) != VISITED && D.yx(y + 1, x - 1) == pi)
				{
					D.set_at (y + 1, x - 1, VISITED);
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
	for (auto i=0; i<M.size(); i++)
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
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

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

	// Copy the image matrix
	auto M = r.aux_image_matrix;
	pixData& D = M.WriteablePixels();

	// Squeeze the intensity range
	PixIntens piRange = r.aux_max - r.aux_min;		// Prepare ROI's intensity range
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();

	for (size_t i = 0; i < D.size(); i++)
		D[i] = Nyxus::to_grayscale (D[i], r.aux_min, piRange, nGrays, Environment::ibsi_compliance);

	// Number of zones
	const int VISITED = -1;
	for (int row=0; row<M.height; row++)
		for (int col = 0; col < M.width; col++)
		{
			// Find a non-blank pixel
			auto pi = D.yx(row, col);
			if (pi == 0 || int(pi)==VISITED)
				continue;

			// Found a gray pixel. Find same-intensity neighbourhood of it.
			std::vector<std::tuple<int, int>> history;
			int x = col, y = row;
			int zoneArea = 1;
			D.yx(y,x) = VISITED;

			for(;;)
			{
				if (D.safe(y,x+1) && D.yx(y,x+1) != VISITED && D.yx(y,x+1) == pi)
				{
					D.yx(y,x+1) = VISITED;
					zoneArea++;

					// Remember this pixel
					history.push_back({x,y});
					// Advance
					x = x + 1;
					// Proceed
					continue;
				}
				if (D.safe(y + 1, x + 1) && D.yx(y+1,x+1) != VISITED && D.yx(y + 1, x+1) == pi)
				{
					D.yx(y + 1, x+1) = VISITED;
					zoneArea++;

					history.push_back({ x,y });
					x = x + 1;
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x) && D.yx(y+1,x) != VISITED && D.yx(y + 1, x) == pi)
				{
					D.yx(y + 1, x) = VISITED;
					zoneArea++;

					history.push_back({ x,y });
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x - 1) && D.yx(y+1,x-1) != VISITED && D.yx(y + 1, x-1) == pi)
				{
					D.yx(y + 1, x-1) = VISITED;
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
					std::tuple<int, int> prev  = history[history.size() - 1];
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
			ACluster clu = {pi, zoneArea};
			Z.push_back (clu);
		}

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

	Ng = Environment::ibsi_compliance ? *std::max_element(std::begin(r.aux_image_matrix.ReadablePixels()), std::end(r.aux_image_matrix.ReadablePixels())) : (int)U.size();
	Ns = height * width;
	Nz = (int) Z.size();
	Np = count;	

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I (U.begin(), U.end());
	std::sort (I.begin(), I.end());	// Optional

	// --allocate the matrix
	P.allocate (Ns, Ng);
	
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

	sum_p = 0;
	for (int i = 1; i <= Ng; ++i) {
		for (int j = 1; j <= Ns; ++j) {
			sum_p += P.matlab(i, j);
		}
	}
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
		double sum = 0;
		for (int j = 1; j <= Ns; ++j)
		{
			double p = P.matlab (i,j);
			sum += p;

			// Once we're iterating matrix P, let's compute specialized sums
			double i2 = double(i) * double(i),
				j2 = double(j) * double(j);

			f_LAHGLE += p * i2 * j2;
			f_LALGLE += p * j2 / i2;
			f_SAHGLE += p * i2 / j2;
			f_SALGLE += p / (i2 * j2);

			double entrTerm = fast_log10(p / sum_p + EPS) / LOG10_2;
			f_ZE += p / sum_p * entrTerm;

			mu_ZV += p / sum_p * double(j);
			mu_GLV += p / sum_p * double(i);
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

bool GLSZMFeature::need (Nyxus::AvailableFeatures f)
{
	return theFeatureSet.isEnabled (f);
}

void GLSZMFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	// Clear result buffers
	double val = 0;
	fvals[GLSZM_SAE][0] = val;
	fvals[GLSZM_LAE][0] = val;
	fvals[GLSZM_GLN][0] = val;
	fvals[GLSZM_GLNN][0] = val;
	fvals[GLSZM_SZN][0] = val;
	fvals[GLSZM_SZNN][0] = val;
	fvals[GLSZM_ZP][0] = val;
	fvals[GLSZM_GLV][0] = val;
	fvals[GLSZM_ZV][0] = val;
	fvals[GLSZM_ZE][0] = val;
	fvals[GLSZM_LGLZE][0] = val;
	fvals[GLSZM_HGLZE][0] = val;
	fvals[GLSZM_SALGLE][0] = val;
	fvals[GLSZM_SAHGLE][0] = val;
	fvals[GLSZM_LALGLE][0] = val;
	fvals[GLSZM_LAHGLE][0] = val;

	// Non-informative matrix?
	if (sum_p == 0)
		return;

	// Precalculate sums of P
	calc_sums_of_P();

	// Calculate features
	if (need(GLSZM_SAE))
		fvals[GLSZM_SAE][0] = calc_SAE();

	if (need(GLSZM_LAE))
		fvals[GLSZM_LAE][0] = calc_LAE();
	
	if (need(GLSZM_GLN))
		fvals[GLSZM_GLN][0] = calc_GLN();
	
	if (need(GLSZM_GLNN))
		fvals[GLSZM_GLNN][0] = calc_GLNN();
	
	if (need(GLSZM_SZN))
		fvals[GLSZM_SZN][0] = calc_SZN();
	
	if (need(GLSZM_SZNN))
		fvals[GLSZM_SZNN][0] = calc_SZNN();
	
	if (need(GLSZM_ZP))
		fvals[GLSZM_ZP][0] = calc_ZP();
	
	if (need(GLSZM_GLV))
		fvals[GLSZM_GLV][0] = calc_GLV();
	
	if (need(GLSZM_ZV))
		fvals[GLSZM_ZV][0] = calc_ZV();
	
	if (need(GLSZM_ZE))
		fvals[GLSZM_ZE][0] = calc_ZE();
	
	if (need(GLSZM_LGLZE))
		fvals[GLSZM_LGLZE][0] = calc_LGLZE();
	
	if (need(GLSZM_HGLZE))
		fvals[GLSZM_HGLZE][0] = calc_HGLZE();
	
	if (need(GLSZM_SALGLE))
		fvals[GLSZM_SALGLE][0] = calc_SALGLE();
	
	if (need(GLSZM_SAHGLE))
		fvals[GLSZM_SAHGLE][0] = calc_SAHGLE();
	
	if (need(GLSZM_LALGLE))
		fvals[GLSZM_LALGLE][0] = calc_LALGLE();
	
	if (need(GLSZM_LAHGLE))
		fvals[GLSZM_LAHGLE][0] = calc_LAHGLE();
}

// 1. Small Area Emphasis
double GLSZMFeature::calc_SAE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double retval = sum_p / double(Np);
	return retval;
}

// 8. Gray Level Variance
double GLSZMFeature::calc_GLV()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'mu_GLV' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			double mu2 = (i - mu_GLV) * (i - mu_GLV);
			f += P.matlab(i,j) / sum_p * mu2;
		}
	}
	return f;
}

// 9. Zone Variance
double GLSZMFeature::calc_ZV()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'mu_ZV' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			double mu2 = (j - mu_ZV) * (j - mu_ZV);
			f += P.matlab(i, j) / sum_p * mu2;
		}
	}
	return f;
}

// 10. Zone Entropy
double GLSZMFeature::calc_ZE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'f_ZE' is expected to have been initialized in calc_sums_of_P()
	double retval = -f_ZE;
	return retval;
}

// 11. Low Gray Level Zone Emphasis
double GLSZMFeature::calc_LGLZE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		f += si[i] / (i * i);
	}

	double retval = f / sum_p;
	return retval;
}

// 12. High Gray Level Zone Emphasis
double GLSZMFeature::calc_HGLZE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		f += si[i] * (i * i);
	}

	double retval = f / sum_p;
	return retval;
}

// 13. Small Area Low Gray Level Emphasis
double GLSZMFeature::calc_SALGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'f_SALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SALGLE / sum_p;
	return retval;
}

// 14. Small Area High Gray Level Emphasis
double GLSZMFeature::calc_SAHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'f_SAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SAHGLE / sum_p;
	return retval;
}

// 15. Large Area Low Gray Level Emphasis
double GLSZMFeature::calc_LALGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'f_LALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LALGLE / sum_p;
	return retval;
}

// 16. Large Area High Gray Level Emphasis
double GLSZMFeature::calc_LAHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate feature. 'f_LAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LAHGLE / sum_p;
	return retval;
}

void GLSZMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		GLSZMFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

