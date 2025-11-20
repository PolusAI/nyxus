#include <limits.h>
#include "3d_ngldm.h"
#include "../environment.h"

using namespace Nyxus;

// Define the neighborhood at max Chebyshev distance \sqrt{2}
struct ShiftToNeighbor
{
	int dx, dy, dz;
};
const static ShiftToNeighbor shifts[] =
{
	{-1,		0,			0},		// West
	{-1,		-1,		0},		// North-West
	{0,		-1,		0},		// North
	{+1,		-1,		0},		// North-East
	{+1,		0,			0},		// East
	{+1,		+1,		0},		// South-East
	{0,		+1,		0},		// South
	{-1,		+1,		0}	,		// South-West

	{-1,		0,			+1},		// West
	{-1,		-1,		+1},		// North-West
	{0,		-1,		+1},		// North
	{+1,		-1,		+1},		// North-East
	{+1,		0,			+1},		// East
	{+1,		+1,		+1},		// South-East
	{0,		+1,		+1},		// South
	{-1,		+1,		+1},		// South-West	

	{-1,		0,			-1},		// West
	{-1,		-1,		-1},		// North-West
	{0,		-1,		-1},		// North
	{+1,		-1,		-1},		// North-East
	{+1,		0,			-1},		// East
	{+1,		+1,		-1},		// South-East
	{0,		+1,		-1},		// South
	{-1,		+1,		-1}		// South-West	
};

const static int nsh = sizeof(shifts) / sizeof(ShiftToNeighbor);

D3_NGLDM_feature::D3_NGLDM_feature() : FeatureMethod("D3_NGLDM_feature")
{
	provide_features(D3_NGLDM_feature::featureset);
}

void D3_NGLDM_feature::clear_buffers()
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

template <class PixelCloud> void D3_NGLDM_feature::gather_unique_intensities (std::vector<PixIntens>& V, PixelCloud& C, PixIntens max_inten, int nGrays)
{
	// Find unique intensities
	std::unordered_set<PixIntens> U;
	PixIntens range = max_inten - 0;
	for (Pixel2 p : C)
	{
		PixIntens inten_ = Nyxus::to_grayscale(p.inten, 0, range, nGrays, Environment::ibsi_compliance);
		U.insert(inten_);
	}

	// Cast the set to vector to be able to access intensities by indices
	V.insert(V.end(), U.begin(), U.end());
	std::sort(V.begin(), V.end());
}

void D3_NGLDM_feature::gather_unique_intensities2 (std::vector<PixIntens>& V, const SimpleCube<PixIntens> & C, PixIntens max_inten, int nGrays)
{
	std::unordered_set<PixIntens> U;
	PixIntens range = max_inten - 0;
	for (auto p : C)
	{
		PixIntens inten_ = Nyxus::to_grayscale(p, 0, range, nGrays, Environment::ibsi_compliance);
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

void D3_NGLDM_feature::calc_ngld_matrix (SimpleMatrix<unsigned int>& NGLDM, int& Nr, /*not const*/ SimpleCube<PixIntens>& I, const std::vector<PixIntens>& U, PixIntens max_inten, int nGrays)
{
	// Temps
	PixIntens range = max_inten - 0;

	// Reset the max dependency
	int max_dep = 0;

	// Iterate pixels of the image skipping margin pixels 
	// in order for a pixel to have all the 8 neighbors
	for (int z = 1; z < I.depth() - 1; z++)
	{
		for (int y = 1; y < I.height() - 1; y++)
		{
			for (int x = 1; x < I.width() - 1; x++)
			{
				// Raw intensity of the central voxel
				PixIntens cpi = I.zyx (z, y, x);

				// Do not skip off-ROI pixels
				//	if (cpi == 0)
				//		continue;

				// Binned intensity
				PixIntens cpi_ = Nyxus::to_grayscale (cpi, 0, range, nGrays, Environment::ibsi_compliance);	// binned 'cpi'

				// Get a dense index value for sparse binned intensity cpi_
				auto iter = std::find(U.begin(), U.end(), cpi_);
				int row = (int)(iter - U.begin());

				// Having pixel (x,y) as the center, iterate pixels of the neighborhood and update its histogram 
				int n_matches = 0;	// (y,x)'s dependency -- the number of matches of center pixel (y,x)'s intensity in its neighborhood
				for (int i = 0; i < nsh; i++)
				{
					if (I.safe(z+shifts[i].dz, y+shifts[i].dy, x+shifts[i].dx))
					{
						PixIntens npi = I.zyx (z+shifts[i].dz, y+shifts[i].dy, x+shifts[i].dx);	// neighboring voxel intensity
						PixIntens npi_ = Nyxus::to_grayscale (npi, 0, range, nGrays, Environment::ibsi_compliance);	// binned 'npi'
						if (cpi_ == npi_)
							n_matches++;
					}
				}
				unsigned int& binCount = NGLDM.yx(row, n_matches);
				binCount++;

				// Update the max dependency
				max_dep = std::max(max_dep, n_matches);
			}
		}
	}

	// The result matrix NGLDM is one column wider due to eistence of the leftmost zero-dependency column
	Nr = max_dep + 1;
}

void D3_NGLDM_feature::calculate (LR& r, const Fsettings& s)
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
		f_DCENE = STNGS_NAN(s);

		return;
	}

	//==== Prepare the NGLD-matrix kit: matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and NGLDM's dimensions
	std::vector<PixIntens> greyLevelsLUT;
	SimpleMatrix<unsigned int> NGLDM;
	int Ng,	// number of grey levels
		Nr;	// maximum number of non-zero dependencies
	prepare_NGLDM_matrix_kit (NGLDM, greyLevelsLUT, Ng, Nr, r, STNGS_NGREYS(s));

	//==== Calculate vectors of totals by intensity and by dependence
	std::vector<double> Sg, Sr;
	calc_rowwise_and_columnwise_totals(Sg, Sr, NGLDM, Ng, Nr);

	//==== Calculate features
	calc_features(Sg, Sr, NGLDM, Nr, greyLevelsLUT, r.aux_area);
}

void D3_NGLDM_feature::prepare_NGLDM_matrix_kit (SimpleMatrix<unsigned int>& NGLDM, std::vector<PixIntens>& grey_levels_LUT, int& Ng, int& Nr, LR& r, int n_greys)
{
	//==== Temps
	/*const*/ SimpleCube<PixIntens> & I = r.aux_image_cube;

	//==== Unique binned intensities gathered from the image matrix, not from raw pixels
	gather_unique_intensities2 (grey_levels_LUT, I, r.aux_max, n_greys);
	Ng = grey_levels_LUT.size();

	int maxNr = nsh + 1;	// max number of columns in the NGLDM = max dependence 8 (due to 8 neighbors) + zero
	PixIntens range = r.aux_max - 0;

	//==== NGLD-matrix
	NGLDM.allocate(maxNr, Ng);	// Ng rows, maxNr columns, but we may end up having fewer informative columns after the NGLD-matrix calculation
	NGLDM.fill(0);
	calc_ngld_matrix (NGLDM, Nr, I, grey_levels_LUT, r.aux_max, n_greys);	// sets the actual max dependency 'Nr'
}

void D3_NGLDM_feature::calc_rowwise_and_columnwise_totals(
	std::vector<double>& Sg,
	std::vector<double>& Sr,
	const SimpleMatrix<unsigned int>& NGLDM,
	const int Ng,
	const int Nr)
{
	// Sum dependencies of each grey level
	Sg.resize(Ng);
	for (int gray_i = 0; gray_i < Ng; gray_i++)
	{
		double sumD = 0;
		for (int r = 0; r < Nr; r++)
			sumD += NGLDM.yx(gray_i, r);
		Sg[gray_i] = sumD;
	}

	// Sum grey levels of each dependence
	Sr.resize(Nr);
	for (int r = 0; r < Nr; r++)
	{
		double sumG = 0;
		for (int gray_i = 0; gray_i < Ng; gray_i++)
			sumG += NGLDM.yx(gray_i, r);
		Sr[r] = sumG;
	}
}

void D3_NGLDM_feature::calc_features(const std::vector<double>& Sg, const std::vector<double>& Sr, SimpleMatrix<unsigned int>& NGLDM, int Nr, const std::vector<PixIntens> U, unsigned int roi_area)
{
	// While Nr is passed as a calculated parameter, Ng is simply the number of NGLDM's rows
	auto Ng = NGLDM.height();

	// Total of all the NGLDM elements
	double Ns = 0;
	for (int i = 0; i < Ng; ++i)
		for (int j = 0; j < Nr; ++j)
		{
			auto sij = NGLDM.yx(i, j);
			Ns += sij;
		}

	// Calculate features
	for (int i = 0; i < Ng; ++i)
	{
		double sj = 0;
		for (int j = 1; j < Nr; ++j)
		{
			double iInt = U[i];	// get intensity by its index
			double sij = NGLDM.yx(i, j);
			double k = j + 1;
			double pij = sij / Ns;

			f_LDE += sij / j / j;

			f_HDE += sij * j * j;
			if (iInt != 0)
				f_LGLCE += sij / iInt / iInt; // Low Grey Level Count Emphasis 
			f_HGLCE += sij * iInt * iInt;	// High Grey Level Count Emphasis 
			if (iInt != 0 && j != 0)
				f_LDLGLE += sij / j / j / iInt / iInt; // Low Dependence Low Grey Level Emphasis
			f_LDHGLE += sij * iInt * iInt / k / k;	// Low Dependence High Grey Level Emphasis
			if (iInt != 0)
				f_HDLGLE += sij * k * k / iInt / iInt;	// High Dependence Low Grey Level Emphasis
			f_HDHGLE += sij * k * k * iInt * iInt;	// High Dependence High Grey Level Emphasis

			f_GLCM += iInt * pij;	// Mean Grey Level Count
			f_DCM += (double(j + 1) * pij);		// Mean Dependence Count
			if (pij > 0)
				f_DCENT -= pij * std::log(pij) / std::log(2);	// Dependence Count Entropy	F_{\mathit{ngl.dc.entr}} = - \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} p_{ij} \log_2 p_{ij}
			f_DCENE += pij * pij;	// Dependence Count Energy	F_{\mathit{ngl.dc.energy}} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_n} p_{ij}^2
			sj += sij;
		}
		f_GLNU += sj * sj;	// Grey Level Non Uniformity
		f_GLNUN += sj * sj;	// Grey Level Non Uniformity Normalised
	}

	for (int i = 0; i < Ng; ++i)
	{
		// Aggregate nonzero dependencies at each grey level
		double si = 0;
		for (int j = 1; j < Nr; ++j)	// note: j \in [1,Nr) due to considering only nonzero dependencies
		{
			double sij = NGLDM.yx(i, j);
			si += sij;
		}
		f_DCNU += si * si;	// Dependence Count Non Uniformity
		f_DCNUN += si * si;	// Dependence Count Non Uniformity Normalised 
	}

	for (int i = 0; i < Ng; ++i)
	{
		for (int j = 1; j < Nr; ++j)
		{
			double i_1base = i + 1;
			double sij = NGLDM.yx(i, j);
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
	f_DCP = 1;		// Dependence count percentage, =1 (per IBSI Release 0.0.1 dev, p. 126)
}

void D3_NGLDM_feature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature3D::NGLDM_LDE][0] = f_LDE;
	fvals[(int)Feature3D::NGLDM_HDE][0] = f_HDE;
	fvals[(int)Feature3D::NGLDM_LGLCE][0] = f_LGLCE;
	fvals[(int)Feature3D::NGLDM_HGLCE][0] = f_HGLCE;
	fvals[(int)Feature3D::NGLDM_LDLGLE][0] = f_LDLGLE;
	fvals[(int)Feature3D::NGLDM_LDHGLE][0] = f_LDHGLE;
	fvals[(int)Feature3D::NGLDM_HDLGLE][0] = f_HDLGLE;
	fvals[(int)Feature3D::NGLDM_HDHGLE][0] = f_HDHGLE;
	fvals[(int)Feature3D::NGLDM_GLNU][0] = f_GLNU;
	fvals[(int)Feature3D::NGLDM_GLNUN][0] = f_GLNUN;
	fvals[(int)Feature3D::NGLDM_DCNU][0] = f_DCNU;
	fvals[(int)Feature3D::NGLDM_DCNUN][0] = f_DCNUN;
	fvals[(int)Feature3D::NGLDM_GLM][0] = f_GLCM;
	fvals[(int)Feature3D::NGLDM_GLV][0] = f_GLV;
	fvals[(int)Feature3D::NGLDM_DCM][0] = f_DCM;
	fvals[(int)Feature3D::NGLDM_DCP][0] = f_DCP;
	fvals[(int)Feature3D::NGLDM_DCV][0] = f_DCV;
	fvals[(int)Feature3D::NGLDM_DCENT][0] = f_DCENT;
	fvals[(int)Feature3D::NGLDM_DCENE][0] = f_DCENE;
}

void D3_NGLDM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	// Iterate ROIs of this batch
	for (auto i = start; i < end; i++)
	{
		// Get ahold of ROI's cached data
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Calculate feature of this ROI
		D3_NGLDM_feature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}

/*static*/ void D3_NGLDM_feature::extract (LR& r, const Fsettings& s)
{
	D3_NGLDM_feature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void D3_NGLDM_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void D3_NGLDM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	calculate (r, s);
}
