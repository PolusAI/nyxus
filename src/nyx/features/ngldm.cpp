#include <limits.h>
#include "ngldm.h"
#include "../environment.h"

using namespace Nyxus;

NGLDMfeature::NGLDMfeature() : FeatureMethod("NGLDMfeature")
{
	provide_features (NGLDMfeature::featureset);
}

void NGLDMfeature::clear_buffers()
{
	f_LDE =
		f_HDE =
		f_LGLCE =
		f_HGLCE =
		f_LDLGLE =
		f_LDHGLE =
		f_HDLGLE =
		f_HDHGLE =
		f_GLNU =
		f_GLNUN =
		f_DCNU =
		f_DCNUN =
		f_GLCM =
		f_GLV =
		f_DCM =
		f_DCP = 
		f_DCV =
		f_DCENT =
		f_DCENE = 0;
}

template <class PixelCloud> void NGLDMfeature::gather_unique_intensities (std::vector<PixIntens> & V, PixelCloud & C, PixIntens max_inten, int nGrays, bool ibsi)
{
	// Find unique intensities
	std::unordered_set<PixIntens> U;
	PixIntens range = max_inten - 0;
	for (Pixel2 p : C)
	{
		PixIntens inten_ = Nyxus::to_grayscale (p.inten, 0, range, nGrays, ibsi);
		U.insert(inten_);
	}

	// Cast the set to vector to be able to access intensities by indices
	V.insert (V.end(), U.begin(), U.end()); 
	std::sort (V.begin(), V.end());
}

void NGLDMfeature::gather_unique_intensities2 (
	std::vector<PixIntens>& V, 
	const pixData& C, 
	PixIntens max_inten, 
	int n_greys,
	bool ibsi)
{
	std::unordered_set<PixIntens> U;
	PixIntens range = max_inten - 0;
	for (auto p : C)
	{
		PixIntens inten_ = Nyxus::to_grayscale (p, 0, range, n_greys, ibsi);
		U.insert(inten_);
	}

	// -- Set to vector to be able to know each intensity's index
	V.insert(V.end(), U.begin(), U.end()); // alt.: std::vector<PixIntens> V (U.begin(), U.end());
	std::sort(V.begin(), V.end());
}

/**
 * Calculates an NGLD-matrix.
 *
 * @param NGLDM		(output) the NGLDM
 * @param Nr		(output) Nr - max column index of non-zero element of NGLDM plus 1 for zero dependency
 * @param I			Masked ROI image matrix. (Non-ROI elements are equal to zero.) 
 * @param U			Grey levels LUT
 * @param max_inten	Maximum intensity
 */

template <class Imgmatrix, class Maskmatrix> void NGLDMfeature::calc_ngld_matrix (
	SimpleMatrix<unsigned int> & NGLDM,
	int & Nr,
	/*not const*/ Imgmatrix& I,
	Maskmatrix& roi_mask,
	const std::vector<PixIntens>& U,
	PixIntens max_inten,
	int nGrays,
	bool ibsi)
{
	// Define the neighborhood at max Chebyshev distance \sqrt{2}
	struct ShiftToNeighbor
	{
		int dx, dy;
	}; 
	const static ShiftToNeighbor shifts[] =
	{
		{-1, 0},	// West
		{-1, -1},	// North-West
		{0, -1},	// North
		{1, -1},	// North-East
		{1, 0},		// East
		{1, 1},		// South-East
		{0, 1},		// South
		{-1, 1}		// South-West
	};

	// Temps
	PixIntens range = max_inten - 0;

	// Reset the max dependency
	int max_dep = 0;

	// Iterate every ROI pixel. Border pixels are valid NGLDM centers; neighbors
	// outside the image or outside the ROI simply do not contribute matches.
	for (int y = 0; y < I.get_height(); y++)
		for (int x = 0; x < I.get_width(); x++)
		{
			if (roi_mask.yx(y, x) == 0)
				continue;

			// Raw intensity of the central pixel whose
			PixIntens cpi = I.yx (y,x);

			// Binned intensity
			PixIntens cpi_ = Nyxus::to_grayscale(cpi, 0, range, nGrays, ibsi);	// binned 'cpi'

			// Get a dense index value for sparse binned intensity cpi_
			auto iter = std::find (U.begin(), U.end(), cpi_);
			if (iter == U.end())
				continue;
			int row = static_cast<int>(iter - U.begin());

			// Having pixel (x,y) as the center, iterate pixels of the neighborhood and update its histogram
			int n_matches = 0;	// (y,x)'s dependency -- the number of matches of center pixel (y,x)'s intensity in its neighborhood
			for (int i = 0; i < 8; i++)
			{
				int ny = y + shifts[i].dy;
				int nx = x + shifts[i].dx;
				if (!roi_mask.safe(ny, nx) || roi_mask.yx(ny, nx) == 0)
					continue;

				PixIntens npi = I.yx(ny, nx);	// neighboring pixel intensity
				PixIntens npi_ = Nyxus::to_grayscale(npi, 0, range, nGrays, ibsi);	// binned 'npi'

				if (cpi_ == npi_)
					n_matches++;
			}
			unsigned int& binCount = NGLDM.yx (row, n_matches);
			binCount++;

			// Update the max dependency
			max_dep = std::max(max_dep, n_matches);
		}

	// The result matrix NGLDM is one column wider due to eistence of the leftmost zero-dependency column
	Nr = max_dep + 1;
}

void NGLDMfeature::calculate (LR& r, const Fsettings& s)
{
	clear_buffers();

	// intercept blank ROIs
	if (r.aux_min == r.aux_max)
	{
		f_LDE =
		f_HDE =
		f_LGLCE =
		f_HGLCE =
		f_LDLGLE =
		f_LDHGLE =
		f_HDLGLE =
		f_HDHGLE =
		f_GLNU =
		f_GLNUN =
		f_DCNU =
		f_DCNUN =
		f_GLCM =
		f_GLV =
		f_DCM =
		f_DCP =
		f_DCV =
		f_DCENT =
		f_DCENE = STNGS_NAN(s); //former theEnvironment.resultOptions.noval()

		return;
	}

	//==== Prepare the NGLD-matrix kit: matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and NGLDM's dimensions
	std::vector<PixIntens> greyLevelsLUT;
	SimpleMatrix<unsigned int> NGLDM;	
	int Ng,	// number of grey levels
		Nr;	// maximum number of non-zero dependencies
	prepare_NGLDM_matrix_kit (NGLDM, greyLevelsLUT, Ng, Nr, r, STNGS_NGREYS(s), STNGS_IBSI(s));

	//==== Calculate vectors of totals by intensity and by dependence
	std::vector<double> Sg, Sr;
	calc_rowwise_and_columnwise_totals (Sg, Sr, NGLDM, Ng, Nr);

	//==== Calculate features
	calc_features (Sg, Sr, NGLDM, Nr, greyLevelsLUT, r.aux_area);
}

void NGLDMfeature::prepare_NGLDM_matrix_kit (
	// OUT
	SimpleMatrix<unsigned int> & NGLDM, 
	std::vector<PixIntens> & grey_levels_LUT, 
	int & Ng, 
	int & Nr, 
	// IN
	LR& r,
	int n_greys,
	bool ibsi)
{
	//==== Temps
	const pixData& I = r.aux_image_matrix.ReadablePixels();

	//==== Unique binned intensities gathered from ROI pixels only
	gather_unique_intensities (grey_levels_LUT, r.raw_pixels, r.aux_max, n_greys, ibsi);
	Ng = grey_levels_LUT.size();

	int maxNr = 9;	// max number of columns in the NGLDM = max dependence 8 (due to 8 neighbors) + zero
	std::vector<Pixel2> mask_pixels;
	mask_pixels.reserve(r.raw_pixels.size());
	for (const auto& p : r.raw_pixels)
		mask_pixels.emplace_back(p.x, p.y, 1);
	ImageMatrix roi_mask;
	roi_mask.allocate(r.aabb.get_width(), r.aabb.get_height());
	roi_mask.calculate_from_pixelcloud(mask_pixels, r.aabb);
	pixData& M = roi_mask.WriteablePixels();

	//==== NGLD-matrix
	NGLDM.allocate (maxNr, Ng);	// Ng rows, maxNr columns, but we may end up having fewer informative columns after the NGLD-matrix calculation
	NGLDM.fill(0);
	calc_ngld_matrix (NGLDM, Nr, I, M, grey_levels_LUT, r.aux_max, n_greys, ibsi);	// sets the actual max dependency 'Nr'
}

void NGLDMfeature::calc_rowwise_and_columnwise_totals (
	std::vector<double>& Sg, 
	std::vector<double>& Sr, 
	const SimpleMatrix<unsigned int>& NGLDM, 
	const int Ng, 
	const int Nr)
{
	// Sum dependencies of each grey level
	Sg.resize (Ng);
	for (int gray_i = 0; gray_i < Ng; gray_i++)
	{
		double sumD = 0;
		for (int r = 0; r < Nr; r++)
			sumD += NGLDM.yx (gray_i, r);
		Sg[gray_i] = sumD;
	}

	// Sum grey levels of each dependence
	Sr.resize (Nr);
	for (int r = 0; r < Nr; r++)
	{
		double sumG = 0;
		for (int gray_i = 0; gray_i < Ng; gray_i++)
			sumG += NGLDM.yx (gray_i, r);
		Sr[r] = sumG;
	}
}

void NGLDMfeature::calc_features (const std::vector<double>& Sg, const std::vector<double>& Sr, SimpleMatrix<unsigned int>& NGLDM, int Nr, const std::vector<PixIntens>& U, unsigned int roi_area)
{
	// While Nr is passed as a calculated parameter, Ng is simply the number of NGLDM's rows
	auto Ng = NGLDM.height();

	// Total of all the NGLDM elements
	double Ns = 0;
	for (double grey_level_sum : Sg)
		Ns += grey_level_sum;

	// Calculate features. NGLDM column 0 means no matching neighbors, which
	// corresponds to IBSI dependence count 1 (the center pixel itself).
	for (int i = 0; i < Ng; ++i)
	{
		double grey_level = static_cast<double>(U[i]);
		for (int j = 0; j < Nr; ++j)
		{
			double dependence_count = static_cast<double>(j + 1);
			double sij = NGLDM.yx(i, j);
			double pij = sij / Ns;

			f_LDE += sij / dependence_count / dependence_count;
			f_HDE += sij * dependence_count * dependence_count;
			if (grey_level != 0.0)
			{
				f_LGLCE += sij / grey_level / grey_level;
				f_LDLGLE += sij / dependence_count / dependence_count / grey_level / grey_level;
				f_HDLGLE += sij * dependence_count * dependence_count / grey_level / grey_level;
			}
			f_HGLCE += sij * grey_level * grey_level;
			f_LDHGLE += sij * grey_level * grey_level / dependence_count / dependence_count;
			f_HDHGLE += sij * dependence_count * dependence_count * grey_level * grey_level;

			f_GLCM += grey_level * pij;	// Mean Grey Level Count
			f_DCM += dependence_count * pij;	// Mean Dependence Count
			if (pij > 0.0)
				f_DCENT -= pij * std::log(pij) / std::log(2);	// Dependence Count Entropy
			f_DCENE += pij * pij;	// Dependence Count Energy
		}
	}

	for (double grey_level_sum : Sg)
	{
		f_GLNU += grey_level_sum * grey_level_sum;	// Grey Level Non Uniformity
		f_GLNUN += grey_level_sum * grey_level_sum;	// Grey Level Non Uniformity Normalised
	}

	for (double dependence_count_sum : Sr)
	{
		f_DCNU += dependence_count_sum * dependence_count_sum;	// Dependence Count Non Uniformity
		f_DCNUN += dependence_count_sum * dependence_count_sum;	// Dependence Count Non Uniformity Normalised
	}

	for (int i = 0; i < Ng; ++i)
	{
		double grey_level = static_cast<double>(U[i]);
		for (int j = 0; j < Nr; ++j)
		{
			double dependence_count = static_cast<double>(j + 1);
			double sij = NGLDM.yx(i, j);
			double pij = sij / Ns;

			// Grey Level Variance
			//	F_{\mathit{ngl.gl.var}}=  \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} (i-\mu)^2 p_{ij}
			//		where
			//	\mu = \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} i\,p_{ij}
			f_GLV += (grey_level - f_GLCM) * (grey_level - f_GLCM) * pij;

			// Dependence Count Variance
			//	F_{\mathit{ngl.dc.var}}= \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} (j-\mu)^2 p_{ij}
			//		where
			//	\mu = \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} j\,p_{ij}
			f_DCV += (dependence_count - f_DCM) * (dependence_count - f_DCM) * pij;
		}
	}
	f_LDE /= Ns;	// Low Dependence Emphasis	F_{\mathit{ngl.LDE}} = \frac{1}{N_s} \sum_{j=1}^{N_n} \frac{s_{.j}}{j^2}
	f_HDE /= Ns;	// High Dependence Emphasis	F_{\mathit{ngl.HDE}} = \frac{1}{N_s} \sum_{j=1}^{N_n} j^2 s_{.j}
	f_LGLCE /= Ns;	// Low Grey Level CountEmphasis	F_{\mathit{ngl.LGLCE}}=\frac{1}{N_s} \sum_{i=1}^{N_g} \frac{s_{i.}}{i^2}
	f_HGLCE /= Ns;	// High Grey Level CountEmphasis	F_{\mathit{ngl.HGLCE}}=\frac{1}{N_s} \sum_{i=1}^{N_g} i^2 s_{i.}
	f_LDLGLE /= Ns;	// Low Dependence Low Grey Level Emphasis	F_{\mathit{ngl.LDLGE}}=\frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} \frac{s_{ij}}{i^2 j^2}
	f_LDHGLE /= Ns;	// Low Dependence High Grey Level Emphasis	F_{\mathit{ngl.LDHGE}}=\frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} \frac{i^2 s_{ij}}{j^2}
	f_HDLGLE /= Ns;	// High Dependence Low Grey Level Emphasis	F_{\mathit{ngl.HDLGE}}=\frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} \frac{j^2 s_{ij}}{i^2}
	f_HDHGLE /= Ns;	// High Dependence High Grey Level Emphasis	F_{\mathit{ngl.HDHGE}}=\frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} i^2 j^2 s_{ij}
	f_GLNU /= Ns;	// Grey Level Non Uniformity	F_{\mathit{ngl.GLNU}}= \frac{1}{N_s} \sum_{i=1}^{N_g} s_{i.}^2
	f_GLNUN /= (Ns * Ns);	// Grey Level Non Uniformity Normalised	F_{\mathit{ngl.GLNU.NORM}}= \frac{1}{N_s^2} \sum_{i=1}^{N_g} s_{i.}^2
	f_DCNU /= Ns;	// Dependence Count Non Uniformity	F_{\mathit{ngl.DCNU}}= \frac{1}{N_s} \sum_{j=1}^{N_n} s_{.j}^2
	f_DCNUN /= (Ns * Ns);	// Dependence Count Non Uniformity Normalised	F_{\mathit{ngl.DCNU.NORM}}= \frac{1}{N_s^2} \sum_{i=1}^{N_n} s_{.j}^2
	f_DCP = 1;		// Dependence count percentage, =1 (per IBSI Release 0.0.1 dev, p. 126)
}

void NGLDMfeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::NGLDM_LDE][0] = f_LDE;
	fvals[(int)Feature2D::NGLDM_HDE][0] = f_HDE;
	fvals[(int)Feature2D::NGLDM_LGLCE][0] = f_LGLCE;
	fvals[(int)Feature2D::NGLDM_HGLCE][0] = f_HGLCE;
	fvals[(int)Feature2D::NGLDM_LDLGLE][0] = f_LDLGLE;
	fvals[(int)Feature2D::NGLDM_LDHGLE][0] = f_LDHGLE;
	fvals[(int)Feature2D::NGLDM_HDLGLE][0] = f_HDLGLE;
	fvals[(int)Feature2D::NGLDM_HDHGLE][0] = f_HDHGLE;
	fvals[(int)Feature2D::NGLDM_GLNU][0] = f_GLNU;
	fvals[(int)Feature2D::NGLDM_GLNUN][0] = f_GLNUN;
	fvals[(int)Feature2D::NGLDM_DCNU][0] = f_DCNU;
	fvals[(int)Feature2D::NGLDM_DCNUN][0] = f_DCNUN;
	fvals[(int)Feature2D::NGLDM_GLM][0] = f_GLCM;
	fvals[(int)Feature2D::NGLDM_GLV][0] = f_GLV;
	fvals[(int)Feature2D::NGLDM_DCM][0] = f_DCM;
	fvals[(int)Feature2D::NGLDM_DCP][0] = f_DCP;
	fvals[(int)Feature2D::NGLDM_DCV][0] = f_DCV;
	fvals[(int)Feature2D::NGLDM_DCENT][0] = f_DCENT;
	fvals[(int)Feature2D::NGLDM_DCENE][0] = f_DCENE;
}

void NGLDMfeature::extract (LR& r, const Fsettings& s)
{
	NGLDMfeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void NGLDMfeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		extract (r, s);
	}
}

void NGLDMfeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void NGLDMfeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	clear_buffers();

	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;

	//==== Unique binned intensities
	std::vector<PixIntens> V;
	gather_unique_intensities (V, r.raw_pixels_NT, r.aux_max, STNGS_NGREYS(s), STNGS_IBSI(s));

	//==== Image matrix
	WriteImageMatrix_nontriv I ("NGLDMfeature-osized_calculate-I", r.label);
	I.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);
	WriteImageMatrix_nontriv M("NGLDMfeature-osized_calculate-M", r.label);
	M.allocate_from_cloud(r.raw_pixels_NT, r.aabb, true);

	//==== NGLD-matrix
	int Ng = V.size(),
		maxNr = 9;	// 8 neighbors + zero
	SimpleMatrix<unsigned int> NGLDM;
	NGLDM.allocate (maxNr, Ng);	// Ng rows, Nd columns
	NGLDM.fill (0);
	int Nr = 0;
	calc_ngld_matrix (NGLDM, Nr, I, M, V, r.aux_max, STNGS_NGREYS(s), STNGS_IBSI(s));
	
	//==== Calculate vectors of totals by intensity (Sg) and by distance (Sr)
	std::vector<double> Sg, Sr;
	calc_rowwise_and_columnwise_totals (Sg, Sr, NGLDM, Ng, Nr);

	//==== Calculate features
	calc_features (Sg, Sr, NGLDM, Nr, V, r.aux_area);
}
