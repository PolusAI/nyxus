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

	if (r.aux_min == r.aux_max)
		return;

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;	// Pairs of (intensity,number_of_neighbors)
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	const pixData& D = r.aux_image_matrix.ReadablePixels();

	// Prepare ROI's intensity range for normalize_I()
	PixIntens piRange = r.aux_max - r.aux_min;

	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
	bool disableGrayBinning = Environment::ibsi_compliance || nGrays >= piRange;

	// Gather zones
	for (int row = 0; row < D.get_height(); row++)
		for (int col = 0; col < D.get_width(); col++)
		{
			// Find a non-blank pixel
			PixIntens pi = Nyxus::to_grayscale (D.yx(row, col), r.aux_min, piRange, nGrays, disableGrayBinning);

			if (pi == 0)
				continue;

			// Count dependencies
			int nd = 1;	// Number of dependencies
			PixIntens piQ; // Pixel intensity of questionn
			if (D.safe(row - 1, col)) {

				piQ = Nyxus::to_grayscale (D.yx(row - 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// North

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row - 1, col + 1)) {

				piQ = Nyxus::to_grayscale (D.yx(row - 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// North-East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row, col + 1)) {

				piQ = Nyxus::to_grayscale (D.yx(row, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col + 1)) {

				piQ = Nyxus::to_grayscale (D.yx(row + 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// South-East

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col)) {

				piQ = Nyxus::to_grayscale(D.yx(row + 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);		// South

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row + 1, col - 1)) {
				
				piQ = Nyxus::to_grayscale (D.yx(row + 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// South-West

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row, col - 1)) {

				piQ = Nyxus::to_grayscale (D.yx(row, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);		// West

				if (piQ == pi)
					nd++;
			}

			if (D.safe(row - 1, col - 1)) {

				piQ = Nyxus::to_grayscale (D.yx(row - 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);	// North-West

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
	Ng = disableGrayBinning ? *std::max_element(std::begin(r.aux_image_matrix.ReadablePixels()), std::end(r.aux_image_matrix.ReadablePixels())) : (int)U.size();
	Nd = 8 + 1;	// N, NE, E, SE, S, SW, W, NW + zero
	Nz = (decltype(Nz))Z.size();

	// --allocate the matrix
	P.allocate(Nd+1, Ng+1);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --iterate zones and fill the matrix
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = disableGrayBinning ? z.first - 1 : int(iter - I.begin());
		// col
		int col = z.second - 1;	// 1-based
		// increment
		auto& k = P.xy(col, row);
		k++;
	}

	sum_p = 0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			sum_p += P.matlab(i,j);
		}
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

	sum_p = 0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			sum_p += P.matlab(i, j);
		}
	}
}

void GLDMFeature::save_value(std::vector<std::vector<double>>& fvals)
{

	if (sum_p == 0) {
		double val = 0.0;

		fvals[(int)Feature2D::GLDM_SDE][0] = val;
		fvals[(int)Feature2D::GLDM_LDE][0] = val;
		fvals[(int)Feature2D::GLDM_GLN][0] = val;
		fvals[(int)Feature2D::GLDM_DN][0] = val;
		fvals[(int)Feature2D::GLDM_DNN][0] = val;
		fvals[(int)Feature2D::GLDM_GLV][0] = val;
		fvals[(int)Feature2D::GLDM_DV][0] = val;
		fvals[(int)Feature2D::GLDM_DE][0] = val;
		fvals[(int)Feature2D::GLDM_LGLE][0] = val;
		fvals[(int)Feature2D::GLDM_HGLE][0] = val;
		fvals[(int)Feature2D::GLDM_SDLGLE][0] = val;
		fvals[(int)Feature2D::GLDM_SDHGLE][0] = val;
		fvals[(int)Feature2D::GLDM_LDLGLE][0] = val;
		fvals[(int)Feature2D::GLDM_LDHGLE][0] = val;
	}

	fvals[(int)Feature2D::GLDM_SDE][0] = calc_SDE();
	fvals[(int)Feature2D::GLDM_LDE][0] = calc_LDE();
	fvals[(int)Feature2D::GLDM_GLN][0] = calc_GLN();
	fvals[(int)Feature2D::GLDM_DN][0] = calc_DN();
	fvals[(int)Feature2D::GLDM_DNN][0] = calc_DNN();
	fvals[(int)Feature2D::GLDM_GLV][0] = calc_GLV();
	fvals[(int)Feature2D::GLDM_DV][0] = calc_DV();
	fvals[(int)Feature2D::GLDM_DE][0] = calc_DE();
	fvals[(int)Feature2D::GLDM_LGLE][0] = calc_LGLE();
	fvals[(int)Feature2D::GLDM_HGLE][0] = calc_HGLE();
	fvals[(int)Feature2D::GLDM_SDLGLE][0] = calc_SDLGLE();
	fvals[(int)Feature2D::GLDM_SDHGLE][0] = calc_SDHGLE();
	fvals[(int)Feature2D::GLDM_LDLGLE][0] = calc_LDLGLE();
	fvals[(int)Feature2D::GLDM_LDHGLE][0] = calc_LDHGLE();
}

// 1. Small Dependence Emphasis(SDE)
double GLDMFeature::calc_SDE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
		f +=  sj[j-1] / double(j*j);
	}

	double retval = f / sum_p;
	return retval;
}

// 2. Large Dependence Emphasis (LDE)
double GLDMFeature::calc_LDE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
		f +=  sj[j-1] * double(j*j);
	}

	double retval = f / sum_p;
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

	double retval = f / sum_p;
	return retval;
}

// 4. Dependence Non-Uniformity (DN)
double GLDMFeature::calc_DN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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

	double retval = f / sum_p;
	return retval;
}
// 5. Dependence Non-Uniformity Normalized (DNN)
double GLDMFeature::calc_DNN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
	double retval = f / double(sum_p * sum_p);
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
			mu += P.matlab(i, j)/sum_p * i;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			double mu2 = (i - mu) * (i - mu);
			f += P.matlab(i, j)/sum_p * mu2;
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
			mu += P.matlab(i, j)/sum_p * j;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			double mu2 = (j - mu) * (j - mu);
			f += P.matlab(i, j)/sum_p * mu2;
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
			double entrTerm = fast_log10(P.matlab(i, j)/sum_p + EPS) / LOG10_2;
			f += P.matlab(i, j)/sum_p * entrTerm;
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
		f +=  si[i-1] / double(i*i);
	}

	double retval = f / sum_p;
	return retval;
}

// 10. High Gray Level Emphasis (HGLE)
double GLDMFeature::calc_HGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

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
		f +=  si[i-1] * double(i*i);
	}

	double retval = f / sum_p;
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
	double retval = f / sum_p;
	return retval;
}

// 12. Small Dependence High Gray Level Emphasis (SDHGLE)
double GLDMFeature::calc_SDHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * double(i*i) / double(j*j);
		}
	}
	double retval = f / sum_p;
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
	double retval = f / sum_p;
	return retval;
}

// 14. Large Dependence High Gray Level Emphasis (LDHGLE)
double GLDMFeature::calc_LDHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nd; j++)
		{
			f += P.matlab(i, j) * double(i * i * j * j);
		}
	}
	double retval = f / sum_p;
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

