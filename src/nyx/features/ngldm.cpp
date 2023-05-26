#include <limits.h>
#include "ngldm.h"
#include "../environment.h"

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
		f_DCV =
		f_DCE =
		f_DCENE = 0;
}

template <class PixelCloud> void NGLDMfeature::gather_unique_intensities (std::vector<PixIntens> & V, PixelCloud & C, PixIntens max_inten)
{
	std::unordered_set<PixIntens> U;
	PixIntens range = max_inten - 0;
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
	for (Pixel2 p : C)
	{
		PixIntens inten_ = Nyxus::to_grayscale(p.inten, 0, range, nGrays, Environment::ibsi_compliance);
		U.insert(inten_);
	}

	// -- Set to vector to be able to know each intensity's index
	V.insert (V.end(), U.begin(), U.end()); //std::vector<PixIntens> V (U.begin(), U.end());
	std::sort (V.begin(), V.end());
}

template <class Imgmatrix> void NGLDMfeature::calc_ngldm (SimpleMatrix<unsigned int> & NGLDM, Imgmatrix & I, std::vector<PixIntens> & V, PixIntens max_inten)
{
	// Define the neighborhood
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
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();

	// Iterate pixels having all the 8 neighbors
	for (int y = 1; y < I.get_height() - 1; y++)
		for (int x = 1; x < I.get_width() - 1; x++)
		{
			// Raw intensity of the central pixel whose 
			PixIntens cpi = I.yx (y,x);

			// Skip off-ROI pixels
			if (cpi == 0)
				continue;

			// Binned intensity
			PixIntens cpi_ = Nyxus::to_grayscale(cpi, 0, range, nGrays, Environment::ibsi_compliance);	// binned 'cpi'

			// Get a dense index value for sparse binned intensity cpi_
			auto iter = std::find(V.begin(), V.end(), cpi_);
			int row = (int)(iter - V.begin());

			// Analyze the neighborhood histogram of pixel (y,x)
			for (int i = 0; i < 8; i++)
			{
				PixIntens npi = I.yx(y + shifts[i].dy, x + shifts[i].dx);	// neighboring pixel intensity
				PixIntens npi_ = Nyxus::to_grayscale(npi, 0, range, nGrays, Environment::ibsi_compliance);	// binned 'npi'

				bool match = cpi_ == npi_;

				if (!match)
					continue;

				unsigned int& binCount = NGLDM.yx(row, i);
				binCount += (int)match;
			}
		}
}

void NGLDMfeature::calculate (LR& r)
{
	clear_buffers();

	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;

	//==== Unique binned intensities
	std::vector<PixIntens> V; 
	gather_unique_intensities (V, r.raw_pixels, r.aux_max);

	//==== Temps
	const pixData & I = r.aux_image_matrix.ReadablePixels();
	int Ng = V.size(), 
		Nr = 8;	// 8 neighbors
	PixIntens range = r.aux_max - 0;
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();

	//==== NGLD-matrix
	SimpleMatrix<unsigned int> NGLDM;
	NGLDM.allocate (Nr, Ng);	// Ng rows, Nd columns
	NGLDM.fill (0);
	calc_ngldm (NGLDM, I, V, r.aux_max);

	//==== Calculate vectors of totals by intensity (Mx) and by distance (Md)
	std::vector<double> Sg, Sr;
	calc_rowwise_and_columnwise_totals (Sg, Sr, NGLDM, Ng, Nr);

	//==== Calculate features
	calc_features (Sg, Sr, NGLDM, r.aux_area);
}

void NGLDMfeature::calc_rowwise_and_columnwise_totals (
	std::vector<double>& Sg, 
	std::vector<double>& Sr, 
	const SimpleMatrix<unsigned int>& NGLDM, 
	const int Ng, 
	const int Nr)
{
	// Sum distances of each grey levels
	Sg.resize (Ng);
	for (int gray_i = 0; gray_i < Ng; gray_i++)
	{
		double sumD = 0;
		for (int r = 0; r < Nr; r++)
			sumD += NGLDM.yx (gray_i, r);
		Sg[gray_i] = sumD;
	}

	// Sum grey levels of each distance
	Sr.resize (Nr);
	for (int r = 0; r < Nr; r++)
	{
		double sumG = 0;
		for (int gray_i = 0; gray_i < Ng; gray_i++)
			sumG += NGLDM.yx (gray_i, r);
		Sr[r] = sumG;
	}
}

void NGLDMfeature::calc_features (const std::vector<double>& Sg, const std::vector<double>& Sr, SimpleMatrix<unsigned int>& M, unsigned int roi_area)
{
	auto Ng = M.height(), 
		Nr = M.width();

	// Total of all the NGLDM elements
	double Ns = 0;
	for (auto sr : Sr)
		Ns += sr;

	// Calculate features
	for (int i = 0; i < Ng; ++i)
	{
		double sj = 0;
		for (int j = 0; j < Nr; ++j)
		{
			double iInt = i + 1;
			double sij = M.yx(i,j);
			double k = j + 1;
			double pij = sij / Ns;

			f_LDE += sij / k / k;
			f_HDE += sij * k * k;
			if (iInt != 0)
				f_LGLCE += sij / iInt / iInt; // Low Grey Level Count Emphasis 
			f_HGLCE += sij * iInt * iInt;	// High Grey Level Count Emphasis 
			if (iInt != 0)
				f_LDLGLE += sij / k / k / iInt / iInt;	// Low Dependence Low Grey Level Emphasis
			f_LDHGLE += sij * iInt * iInt / k / k;	// Low Dependence High Grey Level Emphasis
			if (iInt != 0)
				f_HDLGLE += sij * k * k / iInt / iInt;	// High Dependence Low Grey Level Emphasis
			f_HDHGLE += sij * k * k * iInt * iInt;	// High Dependence High Grey Level Emphasis

			f_GLCM += iInt * pij;	// Mean Grey Level Count
			f_DCM += k * pij;	// Mean Dependence Count
			if (pij > 0)
				f_DCE -= pij * std::log(pij) / std::log(2);	// Dependence Count Entropy	F_{\mathit{ngl.dc.entr}} = - \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} p_{ij} \log_2 p_{ij}
			f_DCENE += pij * pij;	// Dependence Count Energy	F_{\mathit{ngl.dc.energy}} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} p_{ij}^2
			sj += sij;
		}
		f_GLNU += sj * sj;	// Grey Level Non Uniformity
		f_GLNUN += sj * sj;	// Grey Level Non Uniformity Normalised
	}

	for (int j = 0; j < Nr; ++j)
	{
		double si = 0;
		for (int i = 0; i < Ng; ++i)
		{
			double sij = M.yx (i, j);
			si += sij;
		}
		f_DCNU += si * si;	// Dependence Count Non Uniformity
		f_DCNUN += si * si;	// Dependence Count Non Uniformity Normalised
	}
	for (int i = 0; i < Ng; ++i)
	{
		for (int j = 0; j < Nr; ++j)
		{
			double i_1base = i + 1;
			double sij = M.yx (i, j);
			double k = j + 1;
			double pij = sij / Ns;

			// Grey Level Variance	
			//	F_{\mathit{ngl.gl.var}}=  \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} (i-\mu)^2 p_{ij} 
			//		where 
			//	\mu = \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} i\,p_{ij}
			f_GLV += (i_1base - f_GLCM) * (i_1base - f_GLCM) * pij;

			// Dependence Count Variance
			//	F_{\mathit{ngl.dc.var}}= \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} (j-\mu)^2 p_{ij}
			//		where
			//	\mu = \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} j\,p_{ij}
			f_DCV += (k - f_DCM) * (k - f_DCM) * pij;	
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
}

void NGLDMfeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[NGLDM_LDE][0] = f_LDE;
	fvals[NGLDM_HDE][0] = f_HDE;
	fvals[NGLDM_LGLCE][0] = f_LGLCE;
	fvals[NGLDM_HGLCE][0] = f_HGLCE;
	fvals[NGLDM_LDLGLE][0] = f_LDLGLE;
	fvals[NGLDM_LDHGLE][0] = f_LDHGLE;
	fvals[NGLDM_HDLGLE][0] = f_HDLGLE;
	fvals[NGLDM_HDHGLE][0] = f_HDHGLE;
	fvals[NGLDM_GLNU][0] = f_GLNU;
	fvals[NGLDM_GLNUN][0] = f_GLNUN;
	fvals[NGLDM_DCNU][0] = f_DCNU;
	fvals[NGLDM_DCNUN][0] = f_DCNUN;
	fvals[NGLDM_GLM][0] = f_GLCM;
	fvals[NGLDM_GLV][0] = f_GLV;
	fvals[NGLDM_DCM][0] = f_DCM;
	fvals[NGLDM_DCV][0] = f_DCV;
	fvals[NGLDM_DCE][0] = f_DCE;
	fvals[NGLDM_DCENE][0] = f_DCENE;
}

void NGLDMfeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	// Iterate ROIs of this batch
	for (auto i = start; i < end; i++)
	{
		// Get ahold of ROI's cached data
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Calculate feature of this ROI
		NGLDMfeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void NGLDMfeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void NGLDMfeature::osized_calculate(LR& r, ImageLoader&)
{
	clear_buffers();

	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;

	//==== Unique binned intensities
	std::vector<PixIntens> V;
	gather_unique_intensities (V, r.raw_pixels_NT, r.aux_max);

	//==== Image matrix
	WriteImageMatrix_nontriv I ("NGLDMfeature-osized_calculate-I", r.label);
	I.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);

	//==== NGLD-matrix
	int Ng = V.size(),
		Nr = 8;	// 8 neighbors
	SimpleMatrix<unsigned int> NGLDM;
	NGLDM.allocate (Nr, Ng);	// Ng rows, Nd columns
	NGLDM.fill (0);
	calc_ngldm (NGLDM, I, V, r.aux_max);
	
	//==== Calculate vectors of totals by intensity (Sg) and by distance (Sr)
	std::vector<double> Sg, Sr;
	calc_rowwise_and_columnwise_totals (Sg, Sr, NGLDM, Ng, Nr);

	//==== Calculate features
	calc_features (Sg, Sr, NGLDM, r.aux_area);
}
