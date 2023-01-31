#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "ngtdm.h"
#include "image_matrix_nontriv.h"
#include "../environment.h"

NGTDMFeature::NGTDMFeature(): FeatureMethod("NGTDMFeature")
{
	provide_features({
		NGTDM_COARSENESS,
		NGTDM_CONTRAST,
		NGTDM_BUSYNESS,
		NGTDM_COMPLEXITY,
		NGTDM_STRENGTH });
}

void NGTDMFeature::clear_buffers()
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

void NGTDMFeature::calculate (LR& r)
{
	// Clear variables
	clear_buffers();

	// ROI image
	const ImageMatrix& im = r.aux_image_matrix;
	const pixData& D = im.ReadablePixels();

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

	// Gather zones
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
	for (int row = 0; row < D.height(); row++)
		for (int col = 0; col < D.width(); col++)
		{
			// Find a non-blank pixel 
			
			PixIntens pi = Nyxus::to_grayscale (D.yx(row, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance); 
			if (pi == 0)
				continue;

			// Update unique intensities
			U.insert(pi);

			// Evaluate the neighborhood
			double neigsI = 0;

			int nd = 0;	// Number of dependencies

			if (D.safe(row - 1, col) && D.yx(row-1, col) != 0)	// North
			{
				neigsI += Nyxus::to_grayscale (D.yx(row-1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			if (D.safe(row - 1, col + 1) && D.yx(row-1, col+1) != 0)	// North-East
			{
				neigsI += Nyxus::to_grayscale (D.yx(row-1, col+1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			if (D.safe(row, col + 1) && D.yx(row, col+1) != 0)	// East
			{
				neigsI += Nyxus::to_grayscale (D.yx(row, col+1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col + 1) && D.yx(row+1, col+1) != 0)	// South-East
			{
				neigsI += Nyxus::to_grayscale (D.yx(row+1, col+1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col) && D.yx(row+1, col) != 0)	// South
			{
				neigsI += Nyxus::to_grayscale (D.yx(row+1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col - 1) && D.yx(row+1, col-1) != 0)	// South-West
			{
				neigsI += Nyxus::to_grayscale (D.yx(row+1, col-1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row, col - 1) && D.yx(row, col-1) !=0)	// West
			{
				neigsI += Nyxus::to_grayscale (D.yx(row, col-1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row - 1, col - 1) && D.yx(row-1, col-1) != 0)	// North-West
			{
				neigsI += Nyxus::to_grayscale (D.yx(row-1, col-1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			// Save the intensity's average neigborhood intensity
			if (nd > 0) {
				//if(pi == 1) {
				//	std::cerr << "neigsI: " << neigsI << ", nd: " << nd << std::endl;
				//}
				neigsI /= nd;
				AveNeighborhoodInte z = { pi, neigsI };
				Z.push_back(z);
			}
		}

	// Fill the matrix
	// --dimensions
	Ng = Environment::ibsi_compliance ? *std::max_element(std::begin(im.ReadablePixels()), std::end(im.ReadablePixels())) : (int) U.size();
	Ngp = (int) U.size();

	// --allocate the matrix
	P.resize(Ng + 1, 0);
	S.resize(Ng + 1, 0);
	N.resize(Ng + 1, 0);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --Calculate N and S
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = (Environment::ibsi_compliance) ?
			z.first : int(iter - I.begin());
		// col
		int col = (int) z.second;	// 1-based
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

void NGTDMFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[NGTDM_COARSENESS][0] = _coarseness;
	fvals[NGTDM_CONTRAST][0] = _contrast;
	fvals[NGTDM_BUSYNESS][0] = _busyness;
	fvals[NGTDM_COMPLEXITY][0] = _complexity;
	fvals[NGTDM_STRENGTH][0] = _strength;
}

void NGTDMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting

void NGTDMFeature::osized_calculate (LR& r, ImageLoader&)
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
	WriteImageMatrix_nontriv D ("NGTDMFeature_osized_calculate_D", r.label);
	D.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);

	// Gather zones
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
	for (int row = 0; row < D.get_height(); row++)
		for (int col = 0; col < D.get_width(); col++)
		{
			// Find a non-blank pixel 
			PixIntens pi = Nyxus::to_grayscale(D.yx(row, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
			if (pi == 0)
				continue;

			// Update unique intensities
			U.insert(pi);

			// Evaluate the neighborhood
			double neigsI = 0;

			int nd = 0;	// Number of dependencies

			if (D.safe(row - 1, col) && D.yx(row - 1, col) != 0)	// North
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			if (D.safe(row - 1, col + 1) && D.yx(row - 1, col + 1) != 0)	// North-East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			if (D.safe(row, col + 1) && D.yx(row, col + 1) != 0)	// East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col + 1) && D.yx(row + 1, col + 1) != 0)	// South-East
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col + 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col) && D.yx(row + 1, col) != 0)	// South
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row + 1, col - 1) && D.yx(row + 1, col - 1) != 0)	// South-West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row + 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row, col - 1) && D.yx(row, col - 1) != 0)	// West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}
			if (D.safe(row - 1, col - 1) && D.yx(row - 1, col - 1) != 0)	// North-West
			{
				neigsI += Nyxus::to_grayscale(D.yx(row - 1, col - 1), r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
				nd++;
			}

			// Save the intensity's average neigborhood intensity
			if (nd > 0) 
			{
				neigsI /= nd;
				AveNeighborhoodInte z = { pi, neigsI };
				Z.push_back(z);
			}
		}

	// Fill the matrix

	Ng = (int)U.size();
	Ngp = (int)U.size();

	// --allocate the matrix
	P.resize(Ng + 1, 0);
	S.resize(Ng + 1, 0);
	N.resize(Ng + 1, 0);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --Calculate N and S
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = (Environment::ibsi_compliance) ?
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

// Coarseness
double NGTDMFeature::calc_Coarseness()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate the feature
	double sum = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum += P[i] * S[i];
	double retval = 1.0 / sum;
	return retval;
}

// Contrast
double NGTDMFeature::calc_Contrast()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate the feature
	// --term 1
	double sum = 0.0;
	for (int i=1; i<=Ng; i++)
		for (int j = 1; j <= Ng; j++)
		{
			double tmp = P[i] * P[j] * (i - j) * (i - j);
			sum += tmp;
		}
	int Ngp_p2 = Ngp > 1 ? Ngp * (Ngp - 1) : Ngp;
	double term1 = sum / double(Ngp_p2);

	// --term 2
	sum = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum += S[i];
	double term2 = sum / Nvc;

	double retval = term1 * term2;
	return retval;
}

// Busyness
double NGTDMFeature::calc_Busyness()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Trivial case?
	if (Ngp == 1)
		return 0.0;

	// Calculate the feature
	double sum1 = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum1 += P[i] * S[i];

	double sum2 = 0.0;
	for (int i = 1; i <= Ng; i++)
		for (int j = 1; j <= Ng; j++)
		{
			if (P[i] != 0 && P[j] != 0) {
				double tmp = P[i] * double(i) - P[j] * double(j);
				sum2 += std::abs (tmp);
			}
		}
	
	if (sum2 == 0) return 0;

	double retval = sum1 / sum2;
	return retval;
}

// Complexity
double NGTDMFeature::calc_Complexity()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate the feature
	double sum = 0.0;
	for (int i = 1; i <= Ng; i++)
		for (int j = 1; j <= Ng; j++)
		{
			if (P[i] != 0 && P[j] != 0) {
				sum += std::abs(i-j) * (P[i]*S[i] + P[j]*S[j]) / (P[i]+P[j]) ;
			}
		}

	double retval = sum / double(Nvp);
	return retval;
}

// Strength
double NGTDMFeature::calc_Strength()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	// Calculate the feature
	double sum1 = 0.0;
	for (int i = 1; i <= Ng; i++)
		for (int j = 1; j <= Ng; j++)
		{
			if (P[i] != 0 && P[j] != 0) {
				sum1 += (P[i] + P[j]) * (i - j) * (i - j);
			}
		}

	double sum2 = 0.0;
	for (int i = 1; i <= Ng; i++)
		sum2 += S[i];

	double retval = sum1 / sum2;
	return retval;
}

void NGTDMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		NGTDMFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

