#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "gldm.h"

GLDMFeature::GLDMFeature() : FeatureMethod("GLDMFeature")
{
	provide_features({ GLDM_SDE,
		GLDM_LDE,
		GLDM_GLN,
		GLDM_DN,
		GLDM_DNN,
		GLDM_GLV,
		GLDM_DV,
		GLDM_DE,
		GLDM_LGLE,
		GLDM_HGLE,
		GLDM_SDLGLE,
		GLDM_SDHGLE,
		GLDM_LDLGLE,
		GLDM_LDHGLE });
}

void GLDMFeature::calculate(LR& r)
{
	if (r.aux_min == r.aux_max)
		return;

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;	// Pairs of (intensity,number_of_neighbors)
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	const pixData& D = r.aux_image_matrix.ReadablePixels();

	// Gather zones
	for (int row = 1; row < D.height() - 1; row++)
		for (int col = 1; col < D.width() - 1; col++)
		{
			// Find a non-blank pixel
			PixIntens pi = D(row, col);
			if (pi == 0)
				continue;

			// Count dependencies
			int nd = 0;	// Number of dependencies
			if (D(row - 1, col) == pi)	// North
				nd++;
			if (D(row - 1, col + 1) == pi)	// North-East
				nd++;
			if (D(row, col + 1) == pi)	// East
				nd++;
			if (D(row + 1, col + 1) == pi)	// South-East
				nd++;
			if (D(row + 1, col) == pi)	// South
				nd++;
			if (D(row + 1, col - 1) == pi)	// South-West
				nd++;
			if (D(row, col - 1) == pi)	// West
				nd++;
			if (D(row - 1, col - 1) == pi)	// North-West
				nd++;

			// Save the intensity's dependency
			ACluster clu = { pi, nd };
			Z.push_back(clu);

			// Update unique intensities
			U.insert(pi);
		}

	//==== Fill the matrix

	Ng = (decltype(Ng))U.size();
	Nd = 8 + 1;	// N, NE, E, SE, S, SW, W, NW + zero
	Nz = (decltype(Nz))Z.size();

	// --allocate the matrix
	P.allocate(Nd, Ng);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --iterate zones and fill the matrix
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = int(iter - I.begin());
		// col
		int col = z.second;	// 1-based
		// increment
		auto& k = P(col, row);
		k++;
	}
}

void GLDMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void GLDMFeature::osized_calculate(LR& r, ImageLoader& imloader)
{}

void GLDMFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[GLDM_SDE][0] = calc_SDE();
	fvals[GLDM_LDE][0] = calc_LDE();
	fvals[GLDM_GLN][0] = calc_GLN();
	fvals[GLDM_DN][0] = calc_DN();
	fvals[GLDM_DNN][0] = calc_DNN();
	fvals[GLDM_GLV][0] = calc_GLV();
	fvals[GLDM_DV][0] = calc_DV();
	fvals[GLDM_DE][0] = calc_DE();
	fvals[GLDM_LGLE][0] = calc_LGLE();
	fvals[GLDM_HGLE][0] = calc_HGLE();
	fvals[GLDM_SDLGLE][0] = calc_SDLGLE();
	fvals[GLDM_SDHGLE][0] = calc_SDHGLE();
	fvals[GLDM_LDLGLE][0] = calc_LDLGLE();
	fvals[GLDM_LDHGLE][0] = calc_LDHGLE();
}

// 1. Small Dependence Emphasis(SDE)
double GLDMFeature::calc_SDE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) / double(i*i);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 2. Large Dependence Emphasis (LDE)
double GLDMFeature::calc_LDE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * double(j*j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 3. Gray Level Non-Uniformity (GLN)
double GLDMFeature::calc_GLN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double sum = 0.0;
		for (int j = 1; j <= Nd; j++)
		{
			sum += P.matlab(i, j);
		}
		f += sum * sum;
	}
	double retval = f / double(Nz);
	return retval;
}

// 4. Dependence Non-Uniformity (DN)
double GLDMFeature::calc_DN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Nd; i++)
	{
		double sum = 0.0;
		for (int j = 1; j <= Ng; j++)
		{
			sum += P.matlab(j, i);
		}
		f += sum * sum;
	}
	double retval = f / double(Nz);
	return retval;
}
// 5. Dependence Non-Uniformity Normalized (DNN)
double GLDMFeature::calc_DNN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Nd; i++)
	{
		double sum = 0.0;
		for (int j = 1; j <= Ng; j++)
		{
			sum += P.matlab(j, i); 
		}
		f += sum * sum;
	}
	double retval = f / double(Nz * Nz);
	return retval;
}

// 6. Gray Level Variance (GLV)
double GLDMFeature::calc_GLV()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			mu += P.matlab(i, j) * i;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			double mu2 = (i - mu) * (i - mu);
			f += P.matlab(i, j) * mu2;
		}
	}
	return f;
}

// 7. Dependence Variance (DV)
double GLDMFeature::calc_DV()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			mu += P.matlab(i, j) * j;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			double mu2 = (j - mu) * (j - mu);
			f += P.matlab(i, j) * mu2;
		}
	}
	return f;
}

// 8. Dependence Entropy (DE)
double GLDMFeature::calc_DE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			double entrTerm = log2(P.matlab(i, j) + EPS);
			f += P.matlab(i, j) * entrTerm;
		}
	}
	double retval = -f;
	return retval;
}

// 9. Low Gray Level Emphasis (LGLE)
double GLDMFeature::calc_LGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) / double(i*i);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 10. High Gray Level Emphasis (HGLE)
double GLDMFeature::calc_HGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * double(i*i);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 11. Small Dependence Low Gray Level Emphasis (SDLGLE)
double GLDMFeature::calc_SDLGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j < Nd; j++)
		{
			f += P.matlab(i, j) / double(i*i * j*j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 12. Small Dependence High Gray Level Emphasis (SDHGLE)
double GLDMFeature::calc_SDHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i < Ng; i++)
	{
		for (int j = 1; j < Nd; j++)
		{
			f += P.matlab(i, j) * double(i*i) / double(j*j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 13. Large Dependence Low Gray Level Emphasis (LDLGLE)
double GLDMFeature::calc_LDLGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i < Ng; i++)
	{
		for (int j = 1; j < Nd; j++)
		{
			f += P.matlab(i, j) * double(j * j) / double(i * i);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 14. Large Dependence High Gray Level Emphasis (LDHGLE)
double GLDMFeature::calc_LDHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i < Ng; i++)
	{
		for (int j = 1; j < Nd; j++)
		{
			f += P.matlab(i, j) * double(i * i * j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

void GLDMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		GLDMFeature gldm;
		gldm.calculate(r);
		gldm.save_value(r.fvals);
	}
}

