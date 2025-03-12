#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "gldm.h"
#include "../environment.h"

using namespace Nyxus;

GLDMFeature::GLDMFeature() : FeatureMethod("GLDMFeature")
{
	provide_features (GLDMFeature::featureset);
}

void GLDMFeature::calculate(LR& r)
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
		fv_LDHGLE = theEnvironment.nan_substitute;

		return;
	}

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;	// Pairs of (intensity,number_of_neighbors)
	std::vector<ACluster> Z;

	ImageMatrix M;
	M.allocate (r.aux_image_matrix.width, r.aux_image_matrix.height);
	pixData& D = M.WriteablePixels();

	// bin intensities
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

	// Gather zones
	for (int row = 0; row < D.get_height(); row++)
		for (int col = 0; col < D.get_width(); col++)
		{
			PixIntens pi = D.yx(row, col);

			if (pi == 0)
				continue;

			// Count dependencies
			int nd = 1;	// Number of dependencies
			PixIntens piQ; // Pixel intensity of questionn
			if (D.safe(row - 1, col)) 
			{

				piQ = D.yx(row - 1, col);	 // North

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row - 1, col + 1)) 
			{

				piQ = D.yx(row - 1, col + 1);	// North-East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row, col + 1)) 
			{

				piQ = D.yx(row, col + 1);	// East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col + 1)) 
			{

				piQ = D.yx(row + 1, col + 1);	// South-East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col)) 
			{

				piQ = D.yx(row + 1, col);		// South

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col - 1)) 
			{

				piQ = D.yx(row + 1, col - 1);	// South-West

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row, col - 1)) 
			{

				piQ = D.yx(row, col - 1);		// West

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row - 1, col - 1)) 
			{

				piQ = D.yx(row - 1, col - 1);	// North-West

				if (piQ == pi)
					nd++;
			}

			// Save the intensity's dependency
			ACluster clu = { pi, nd };
			Z.push_back(clu);
		}


	//==== Fill the matrix
	Ng = greyInfo==0 ? *std::max_element(I.begin(), I.end()) : (int) I.size();
	Nd = 8 + 1;	// N, NE, E, SE, S, SW, W, NW + zero

	// --allocate the matrix
	P.allocate (Nd + 1, Ng + 1);

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
		max_Nd = std::max (max_Nd, col+1);
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
		fv_LDHGLE = theEnvironment.nan_substitute;
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

void GLDMFeature::clear_buffers()
{
	bad_roi_data = false;
	int Ng = 0;
	int Nd = 0;
	int Nz = 0;

	double sum_p = 0;

	P.clear();
}

void GLDMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void GLDMFeature::osized_calculate (LR& r, ImageLoader&)
{
	clear_buffers();

	if (r.aux_min == r.aux_max)
		return;

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;	// Pairs of (intensity,number_of_neighbors)
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	WriteImageMatrix_nontriv D("GLDMFeature-osized_calculate-D", r.label);
	D.allocate_from_cloud(r.raw_pixels_NT, r.aabb, false);

	// Prepare ROI's intensity range for normalize_I()
	PixIntens piRange = r.aux_max - r.aux_min;

	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();

	size_t height = D.get_height(),
		width = D.get_width();

	// Gather zones
	for (size_t row = 0; row < height; row++)
		for (size_t col = 0; col < width; col++)
		{
			// Find a non-blank pixel
			PixIntens pi = Nyxus::to_grayscale((unsigned int) D.yx(row, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);

			if (pi == 0)
				continue;

			// Count dependencies
			int nd = 1;	// Number of dependencies
			PixIntens piQ; // Pixel intensity of questionn
			if (D.safe(row - 1, col)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row - 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// North

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row - 1, col + 1)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row - 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// North-East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row, col + 1)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col + 1)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row + 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// South-East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row + 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);		// South

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col - 1)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row + 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// South-West

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row, col - 1)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);		// West

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row - 1, col - 1)) {

				piQ = Nyxus::to_grayscale((unsigned int) D.yx(row - 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// North-West

				if (piQ == pi)
					nd++;
			}

			// Save the intensity's dependency
			ACluster clu = { pi, nd };
			Z.push_back(clu);

			// Update unique intensities
			U.insert(pi);
		}

	//==== Fill the matrix
	Ng = (int) U.size();
	Nd = 8 + 1;	// N, NE, E, SE, S, SW, W, NW + zero
	Nz = (decltype(Nz))Z.size();

	// --allocate the matrix
	P.allocate(Nd + 1, Ng + 1);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --iterate zones and fill the matrix
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = (Environment::ibsi_compliance) ? z.first - 1 : int(iter - I.begin());
		// col
		int col = z.second - 1;	// 1-based
		// increment
		auto& k = P.xy(col, row);
		k++;
	}

	Nz = 0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			Nz += P.matlab(i, j);
		}
	}
}

void GLDMFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::GLDM_SDE][0] = fv_SDE;
	fvals[(int)Feature2D::GLDM_LDE][0] = fv_LDE;
	fvals[(int)Feature2D::GLDM_GLN][0] = fv_GLN;
	fvals[(int)Feature2D::GLDM_DN][0] = fv_DN;
	fvals[(int)Feature2D::GLDM_DNN][0] = fv_DNN;
	fvals[(int)Feature2D::GLDM_GLV][0] = fv_GLV;
	fvals[(int)Feature2D::GLDM_DV][0] = fv_DV;
	fvals[(int)Feature2D::GLDM_DE][0] = fv_DE;
	fvals[(int)Feature2D::GLDM_LGLE][0] = fv_LGLE;
	fvals[(int)Feature2D::GLDM_HGLE][0] = fv_HGLE;
	fvals[(int)Feature2D::GLDM_SDLGLE][0] = fv_SDLGLE;
	fvals[(int)Feature2D::GLDM_SDHGLE][0] = fv_SDHGLE;
	fvals[(int)Feature2D::GLDM_LDLGLE][0] = fv_LDLGLE;
	fvals[(int)Feature2D::GLDM_LDHGLE][0] = fv_LDHGLE;
}

// 1. Small Dependence Emphasis(SDE)
double GLDMFeature::calc_SDE()
{
	double sum = 0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
			sum += P.matlab (i,j) / (double(j) * double(j));
	}
	double retval = sum / double(Nz);
	return retval;
}

// 2. Large Dependence Emphasis (LDE)
double GLDMFeature::calc_LDE()
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
double GLDMFeature::calc_GLN()
{
	std::vector<double> si(Ng+1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			si[i-1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		f += si[i-1] * si[i-1];
	}

	double retval = f / double(Nz);
	return retval;
}

// 4. Dependence Non-Uniformity (DN)
double GLDMFeature::calc_DN()
{
	std::vector<double> sj(Nd+1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			sj[j-1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int j = 1; j <= Nd; j++)
	{
		f += sj[j-1] * sj[j-1];
	}

	double retval = f / double(Nz);
	return retval;
}
// 5. Dependence Non-Uniformity Normalized (DNN)
double GLDMFeature::calc_DNN()
{
	std::vector<double> sj(Nd+1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			sj[j-1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int j = 1; j <= Nd; j++)
	{
		f += sj[j-1] * sj[j-1];
	}
	double retval = f / (double(Nz) * double(Nz));
	return retval;
}

// 6. Gray Level Variance (GLV)
double GLDMFeature::calc_GLV()
{
	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		for (int j = 1; j <= Nd; j++)
		{
			mu += P.matlab(i, j) / double(Nz) * inten;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		for (int j = 1; j <= Nd; j++)
		{
			double mu2 = (inten - mu) * (inten - mu);
			f += P.matlab(i, j) / double(Nz) * mu2;
		}
	}
	return f;
}

// 7. Dependence Variance (DV)
double GLDMFeature::calc_DV()
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
				mu2 = d*d;
			f += P.matlab(i, j) / double(Nz) * mu2;
		}
	}
	return f;
}

// 8. Dependence Entropy (DE)
double GLDMFeature::calc_DE()
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
double GLDMFeature::calc_LGLE()
{
	double sum_i = 0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten2 = (double) I[i-1];
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
double GLDMFeature::calc_HGLE()
{
	std::vector<double> si(Ng+1, 0.);
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			si[i-1] += P.matlab(i, j);
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		f +=  si[i-1] * inten * inten;
	}

	double retval = f / double(Nz);
	return retval;
}

// 11. Small Dependence Low Gray Level Emphasis (SDLGLE)
double GLDMFeature::calc_SDLGLE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) / double(inten * inten * j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 12. Small Dependence High Gray Level Emphasis (SDHGLE)
double GLDMFeature::calc_SDHGLE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * (inten * inten) / double(j*j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 13. Large Dependence Low Gray Level Emphasis (LDLGLE)
double GLDMFeature::calc_LDLGLE()
{
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double) I[i-1];
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * double(j * j) / (inten * inten);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 14. Large Dependence High Gray Level Emphasis (LDHGLE)
double GLDMFeature::calc_LDHGLE()
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

void GLDMFeature::extract (LR& r)
{		
	GLDMFeature gldm;
	gldm.calculate(r);
	gldm.save_value(r.fvals);
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

