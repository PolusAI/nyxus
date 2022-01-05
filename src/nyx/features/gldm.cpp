#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "gldm.h"

GLDM_features::GLDM_features (int minI, int maxI, const ImageMatrix& im)
{
	//==== Check if the ROI is degenerate (equal intensity)
	if (minI == maxI)
	{
		bad_roi_data = true;
		return;
	}

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;	// Pairs of (intensity,number_of_neighbors)
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	const pixData& D = im.ReadablePixels();

	// Gather zones
	for (int row = 1; row < D.height()-1; row++)
		for (int col = 1; col < D.width()-1; col++)
		{
			// Find a non-blank pixel
			PixIntens pi = D(row, col);
			if (pi == 0)
				continue;

			// Count dependencies
			int nd = 0;	// Number of dependencies
			if (D (row-1, col) == pi)	// North
				nd++;
			if (D (row-1, col+1) == pi)	// North-East
				nd++;
			if (D (row, col+1) == pi)	// East
				nd++;
			if (D (row+1, col+1) == pi)	// South-East
				nd++;
			if (D (row+1, col) == pi)	// South
				nd++;
			if (D (row+1, col-1) == pi)	// South-West
				nd++;
			if (D (row, col-1) == pi)	// West
				nd++;
			if (D (row-1, col-1) == pi)	// North-West
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
	//xxx	P_matrix P;
	P.allocate (Nd, Ng);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --iterate zones and fill the matrix
	for (auto& z : Z)
	{
		// row
		auto iter = std::find (I.begin(), I.end(), z.first);
		int row = int(iter - I.begin());
		// col
		int col = z.second;	// 1-based
		// increment
		auto& k = P(col, row);
		k++;
	}
}

// 1. Small Dependence Emphasis(SDE)
double GLDM_features::calc_SDE()
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
double GLDM_features::calc_LDE()
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
double GLDM_features::calc_GLN()
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
double GLDM_features::calc_DN()
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
double GLDM_features::calc_DNN()
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
double GLDM_features::calc_GLV()
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
double GLDM_features::calc_DV()
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
double GLDM_features::calc_DE()
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
double GLDM_features::calc_LGLE()
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
double GLDM_features::calc_HGLE()
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
double GLDM_features::calc_SDLGLE()
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
double GLDM_features::calc_SDHGLE()
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
double GLDM_features::calc_LDLGLE()
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
double GLDM_features::calc_LDHGLE()
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

void GLDM_features::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		GLDM_features gldm ((int)r.fvals[MIN][0], (int)r.fvals[MAX][0], r.aux_image_matrix);
		r.fvals[GLDM_SDE][0] = gldm.calc_SDE();
		r.fvals[GLDM_LDE][0] = gldm.calc_LDE();
		r.fvals[GLDM_GLN][0] = gldm.calc_GLN();
		r.fvals[GLDM_DN][0] = gldm.calc_DN();
		r.fvals[GLDM_DNN][0] = gldm.calc_DNN();
		r.fvals[GLDM_GLV][0] = gldm.calc_GLV();
		r.fvals[GLDM_DV][0] = gldm.calc_DV();
		r.fvals[GLDM_DE][0] = gldm.calc_DE();
		r.fvals[GLDM_LGLE][0] = gldm.calc_LGLE();
		r.fvals[GLDM_HGLE][0] = gldm.calc_HGLE();
		r.fvals[GLDM_SDLGLE][0] = gldm.calc_SDLGLE();
		r.fvals[GLDM_SDHGLE][0] = gldm.calc_SDHGLE();
		r.fvals[GLDM_LDLGLE][0] = gldm.calc_LDLGLE();
		r.fvals[GLDM_LDHGLE][0] = gldm.calc_LDHGLE();
	}
}

