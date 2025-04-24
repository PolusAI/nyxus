#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include "3d_glrlm.h"
#include "../environment.h"

using namespace Nyxus;

int D3_GLRLM_feature::n_levels = 0;


D3_GLRLM_feature::D3_GLRLM_feature() : FeatureMethod("D3_GLRLM_feature")
{
	provide_features(D3_GLRLM_feature::featureset);
}

void D3_GLRLM_feature::calculate(LR& r)
{
	//==== Clear the feature values buffers
	clear_buffers();

	auto minI = r.aux_min,
		maxI = r.aux_max;

	// intercept blank ROIs
	if (minI == maxI)
	{
		// insert a non-NAN value for all 4 angles to make the output expecting 4-angled values happy
		auto w = theEnvironment.resultOptions.noval();	// safe NAN
		angled_SRE.resize(4, w);
		angled_LRE.resize(4, w);
		angled_GLN.resize(4, w);
		angled_GLNN.resize(4, w);
		angled_RLN.resize(4, w);
		angled_RLNN.resize(4, w);
		angled_RP.resize(4, w);
		angled_GLV.resize(4, w);
		angled_RV.resize(4, w);
		angled_RE.resize(4, w);
		angled_LGLRE.resize(4, w);
		angled_HGLRE.resize(4, w);
		angled_SRLGLE.resize(4, w);
		angled_SRHGLE.resize(4, w);
		angled_LRLGLE.resize(4, w);
		angled_LRHGLE.resize(4, w);

		bad_roi_data = true;
		return;
	}

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	using AngleZones = std::vector<ACluster>;

	//==== While scanning clusters, learn unique intensities 
	using AngleUniqInte = std::unordered_set<PixIntens>;

	//==== Iterate angles 
	for (int angleIdx = 0; angleIdx < 8; angleIdx++)
	{
		// Clusters at angle 'angleIdx'
		AngleZones Z;

		// Unique intensities at angle 'angleIdx'
		AngleUniqInte U;

		// We need it to estimate the x-dimension of matrix P
		int maxZoneArea = 0;

		// Copy the image matrix. We'll use it to maintain state of cluster scanning 
		int w = r.aux_image_cube.width(),
			h = r.aux_image_cube.height(),
			d = r.aux_image_cube.depth();
		SimpleCube<PixIntens> D;
		D.allocate (w, h, d);

		// Squeeze the intensity range
		auto greyInfo = theEnvironment.get_coarse_gray_depth();
		auto greyInfo_localFeature = D3_GLRLM_feature::n_levels;
		if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
			greyInfo = greyInfo_localFeature;
		if (Nyxus::theEnvironment.ibsi_compliance)
			greyInfo = 0;

		bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

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

		// Find zones
		const int VISITED = -1;

		// --- Scan the image and check non-blank pixels' clusters
		for (int zslice = 0; zslice < d; zslice++)
		{
			for (int row = 0; row < h; row++)
			{
				for (int col = 0; col < w; col++)
				{
					// Find a non-blank pixel
					auto pi = D.zyx (zslice, row, col);
					if (pi == 0 || int(pi) == VISITED)
						continue;

					// Found a non-blank pixel. Find same-intensity neighbourhood of it.
					std::vector<std::tuple<int, int, int>> history;
					int x = col, y = row, z = zslice;
					int zoneArea = 1;
					D.zyx (z, y, x) = VISITED;

					// State machine scanning the rest of the cluster
					for (;;)
					{
						int dx, dy, dz;

						//********** ang XY,Z = 0
						// ang X,Y = 0
						if (angleIdx == 0 && D.safe(z, y, x+1) && D.zyx(z, y, x+1) == pi)
						{
							D.zyx(z, y, x + 1) = VISITED;
							zoneArea++;

							// Remember this pixel
							history.push_back({ x,y,z });
							// Advance 
							x = x + 1;
							// Proceed
							continue;
						}

						// ang X,Y = 45
						if (angleIdx == 1 && D.safe(z, y+1, x+1) && D.zyx(z, y+1, x+1) == pi)
						{
							D.zyx (z, y+1, x+1) = VISITED;
							zoneArea++;

							history.push_back({ x,y,z });
							x = x + 1;
							y = y + 1;
							continue;
						}

						// ang X,Y = 90
						if (angleIdx == 2 && D.safe(z, y+1, x) && D.zyx(z, y+1, x) == pi)
						{
							D.zyx(z, y+1, x) = VISITED;
							zoneArea++;

							history.push_back({ x,y,z });
							y = y + 1;
							continue;
						}

						// ang X,Y = 135
						if (angleIdx == 3 && D.safe(z, y+1, x-1) && D.zyx(z, y+1, x-1) == pi)
						{
							D.zyx(z, y+1, x-1) = VISITED;
							zoneArea++;

							history.push_back({ x,y,z });
							x = x - 1;
							y = y + 1;
							continue;
						}

						//********** ang XY,Z != 0
						// ang X,Y = 0
						if (angleIdx == 4 && D.safe(z+1, y, x) && D.zyx(z+1, y, x) == pi)
						{
							D.zyx(z+1, y, x) = VISITED;
							zoneArea++;

							// remember this pixel
							history.push_back({ x,y,z });
							// advance 
							z = z + 1;
							// proceed
							continue;
						}

						// ang X,Y = 45
						if (angleIdx == 5 && D.safe(z+1, y+1, x) && D.zyx(z+1, y+1, x) == pi)
						{
							D.zyx(z+1, y+1, x) = VISITED;
							zoneArea++;

							history.push_back({ x,y,z });
							z = z + 1;
							y = y + 1;
							continue;
						}

						// ang X,Y = 90
						if (angleIdx == 6 && D.safe(z+1, y, x+1) && D.zyx(z+1, y, x+1) == pi)
						{
							D.zyx(z+1, y, x+1) = VISITED;
							zoneArea++;

							history.push_back({ x,y,z });
							z = z + 1;
							x = x + 1;
							continue;
						}

						// ang X,Y = 135
						if (angleIdx == 7 && D.safe(z+1, y+1, x-1) && D.zyx(z+1, y+1, x-1) == pi)
						{
							D.zyx(z+1, y+1, x-1) = VISITED;
							zoneArea++;

							history.push_back({ x,y,z });
							z = z + 1;
							y = y + 1;
							x = x - 1;
							continue;
						}

						// Return from the branch
						if (history.size() > 0)
						{
							// Recollect the coordinate where we diverted from
							std::tuple<int, int, int> prev = history [history.size() - 1];
							history.pop_back();
						}

						// We are done exploring this cluster
						break;
					}

					// --2
					maxZoneArea = std::max(maxZoneArea, zoneArea);

					// --3
					ACluster clu = { pi, zoneArea };
					Z.push_back(clu);
				}
			}
		}

		// count non-zero pixels
		int count = 0;

		for (const auto& px : r.aux_image_cube)
		{
			if (px != 0)
				count++;
		}

		//==== Create a zone matrix

		int Ng = Environment::ibsi_compliance ? *std::max_element(I.begin(), I.end()) : I.size();
		int Nr = maxZoneArea;
		int Nz = (int)Z.size();
		int Np = count;

		// --allocate the matrix
		P_matrix P;
		P.allocate(Nr, Ng);	// w = Nr, h = card(I) aka Ng

		// --iterate zones and fill the matrix
		for (auto& z : Z)
		{
			auto inten = z.first;
			// row (grey level)
			int row = -1;
			if (Environment::ibsi_compliance)
				row = inten - 1;
			else
			{
				auto lower = std::lower_bound(I.begin(), I.end(), inten);	// enjoy sorted vector 'I'
				row = int(lower - I.begin());	// intensity index in array of unique intensities 'I'
			}
			// col
			int col = z.second - 1;	// 0-based => -1
			// update the matrix
			auto& k = P.xy(col, row);
			k++;
		}

		// --save this angle's results
		angles_P.push_back(P);
		angles_Ng.push_back(Ng);
		angles_Nr.push_back(Nr);
		angles_Np.push_back(Np);

		double sum = 0;
		for (int i = 1; i <= P.height(); ++i)
			for (int j = 1; j <= P.width(); ++j)
				sum += P.matlab(i, j);
		sum = 0;
		for (auto p : P)
			sum += p;

		sum_p.push_back(sum);
	} //- angles

	calc_SRE(angled_SRE);
	calc_LRE(angled_LRE);
	calc_GLN(angled_GLN);
	calc_GLNN(angled_GLNN);
	calc_RLN(angled_RLN);
	calc_RLNN(angled_RLNN);
	calc_RP(angled_RP);
	calc_GLV(angled_GLV);
	calc_RV(angled_RV);
	calc_RE(angled_RE);
	calc_LGLRE(angled_LGLRE);
	calc_HGLRE(angled_HGLRE);
	calc_SRLGLE(angled_SRLGLE);
	calc_SRHGLE(angled_SRHGLE);
	calc_LRLGLE(angled_LRLGLE);
	calc_LRHGLE(angled_LRHGLE);
}

void D3_GLRLM_feature::clear_buffers()
{
	bad_roi_data = false;
	angles_Ng.clear();
	angles_Nr.clear();
	angles_Np.clear();
	angles_P.clear();
	sum_p.clear();
	I.clear();

	angled_SRE.clear();
	angled_LRE.clear();
	angled_GLN.clear();
	angled_GLNN.clear();
	angled_RLN.clear();
	angled_RLNN.clear();
	angled_RP.clear();
	angled_GLV.clear();
	angled_RV.clear();
	angled_RE.clear();
	angled_LGLRE.clear();
	angled_HGLRE.clear();
	angled_SRLGLE.clear();
	angled_SRHGLE.clear();
	angled_LRLGLE.clear();
	angled_LRHGLE.clear();
}

// Not supporting the online mode
void D3_GLRLM_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting

void D3_GLRLM_feature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature3D::GLRLM_SRE] = angled_SRE;
	fvals[(int)Feature3D::GLRLM_LRE] = angled_LRE;
	fvals[(int)Feature3D::GLRLM_GLN] = angled_GLN;
	fvals[(int)Feature3D::GLRLM_GLNN] = angled_GLNN;
	fvals[(int)Feature3D::GLRLM_RLN] = angled_RLN;
	fvals[(int)Feature3D::GLRLM_RLNN] = angled_RLNN;
	fvals[(int)Feature3D::GLRLM_RP] = angled_RP;
	fvals[(int)Feature3D::GLRLM_GLV] = angled_GLV;
	fvals[(int)Feature3D::GLRLM_RV] = angled_RV;
	fvals[(int)Feature3D::GLRLM_RE] = angled_RE;
	fvals[(int)Feature3D::GLRLM_LGLRE] = angled_LGLRE;
	fvals[(int)Feature3D::GLRLM_HGLRE] = angled_HGLRE;
	fvals[(int)Feature3D::GLRLM_SRLGLE] = angled_SRLGLE;
	fvals[(int)Feature3D::GLRLM_SRHGLE] = angled_SRHGLE;
	fvals[(int)Feature3D::GLRLM_LRLGLE] = angled_LRLGLE;
	fvals[(int)Feature3D::GLRLM_LRHGLE] = angled_LRHGLE;

	// -- averages --
	fvals[(int)Feature3D::GLRLM_SRE_AVE][0] = calc_ave(angled_SRE);
	fvals[(int)Feature3D::GLRLM_LRE_AVE][0] = calc_ave(angled_LRE);
	fvals[(int)Feature3D::GLRLM_GLN_AVE][0] = calc_ave(angled_GLN);
	fvals[(int)Feature3D::GLRLM_GLNN_AVE][0] = calc_ave(angled_GLNN);
	fvals[(int)Feature3D::GLRLM_RLN_AVE][0] = calc_ave(angled_RLN);
	fvals[(int)Feature3D::GLRLM_RLNN_AVE][0] = calc_ave(angled_RLNN);
	fvals[(int)Feature3D::GLRLM_RP_AVE][0] = calc_ave(angled_RP);
	fvals[(int)Feature3D::GLRLM_GLV_AVE][0] = calc_ave(angled_GLV);
	fvals[(int)Feature3D::GLRLM_RV_AVE][0] = calc_ave(angled_RV);
	fvals[(int)Feature3D::GLRLM_RE_AVE][0] = calc_ave(angled_RE);
	fvals[(int)Feature3D::GLRLM_LGLRE_AVE][0] = calc_ave(angled_LGLRE);
	fvals[(int)Feature3D::GLRLM_HGLRE_AVE][0] = calc_ave(angled_HGLRE);
	fvals[(int)Feature3D::GLRLM_SRLGLE_AVE][0] = calc_ave(angled_SRLGLE);
	fvals[(int)Feature3D::GLRLM_SRHGLE_AVE][0] = calc_ave(angled_SRHGLE);
	fvals[(int)Feature3D::GLRLM_LRLGLE_AVE][0] = calc_ave(angled_LRLGLE);
	fvals[(int)Feature3D::GLRLM_LRHGLE_AVE][0] = calc_ave(angled_LRHGLE);
}


// 1. Short Run Emphasis 
// ai - angle index
void D3_GLRLM_feature::calc_SRE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		double f = 0.;
		std::vector<double> rj(Nr + 1, 0.);
		for (int i = 1; i <= Ng; ++i) {
			for (int j = 1; j <= Nr; ++j) {
				rj[j] += P.matlab(i, j);
			}
		}

		for (int j = 1; j <= Nr; ++j) {
			f += rj[j] / (j * j);
		}

		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 2. Long Run Emphasis 
void D3_GLRLM_feature::calc_LRE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * j * j;
			}
		}

		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 3. Gray Level Non-Uniformity 
void D3_GLRLM_feature::calc_GLN(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			double sum = 0.0;
			for (int j = 1; j <= Nr; j++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 4. Gray Level Non-Uniformity Normalized 
void D3_GLRLM_feature::calc_GLNN(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			double sum = 0.0;
			for (int j = 1; j <= Nr; j++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(sum_p[ai] * sum_p[ai]);
		af.push_back(retval);
	}
}

// 5. Run Length Non-Uniformity
void D3_GLRLM_feature::calc_RLN(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int j = 1; j <= Nr; j++)
		{
			double sum = 0.0;
			for (int i = 1; i <= Ng; i++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 6. Run Length Non-Uniformity Normalized 
void D3_GLRLM_feature::calc_RLNN(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int j = 1; j <= Nr; j++)
		{
			double sum = 0.0;
			for (int i = 1; i <= Ng; i++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(sum_p[ai] * sum_p[ai]);
		af.push_back(retval);
	}
}

// 7. Run Percentage
void D3_GLRLM_feature::calc_RP(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Np = angles_Np[ai];

		double retval = double(sum_p[ai] / Np);
		af.push_back(retval);
	}
}

// 8. Gray Level Variance 
void D3_GLRLM_feature::calc_GLV(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0)
		{
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double mu = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				mu += P.matlab(i, j) / sum_p[ai] * inten;
			}
		}

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				double mu2 = (inten - mu) * (inten - mu);
				f += P.matlab(i, j) / sum_p[ai] * mu2;
			}
		}
		af.push_back(f);
	}
}

// 9. Run Variance 
void D3_GLRLM_feature::calc_RV(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double mu = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				mu += P.matlab(i, j) / sum_p[ai] * j;
			}
		}

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				double mu2 = (j - mu) * (j - mu);
				f += P.matlab(i, j) / sum_p[ai] * mu2;
			}
		}
		af.push_back(f);
	}
}

// 10. Run Entropy 
void D3_GLRLM_feature::calc_RE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				double entrTerm = fast_log10(P.matlab(i, j) / sum_p[ai] + EPS) / LOG10_2;
				f += P.matlab(i, j) / sum_p[ai] * entrTerm;
			}
		}
		double retval = -f;
		af.push_back(retval);
	}
}

// 11. Low Gray Level Run Emphasis 
void D3_GLRLM_feature::calc_LGLRE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) / double(inten * inten);
			}
		}
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 12. High Gray Level Run Emphasis 
void D3_GLRLM_feature::calc_HGLRE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(inten * inten);
			}
		}
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 13. Short Run Low Gray Level Emphasis 
void D3_GLRLM_feature::calc_SRLGLE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) / double(inten * inten * j * j);
			}
		}
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 14. Short Run High Gray Level Emphasis 
void D3_GLRLM_feature::calc_SRHGLE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(inten * inten) / double(j * j);
			}
		}
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 15. Long Run Low Gray Level Emphasis 
void D3_GLRLM_feature::calc_LRLGLE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0)
		{
			af.push_back(0.0);
			continue;
		}

		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(j * j) / double(inten * inten);
			}
		}
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 16. Long Run High Gray Level Emphasis 
void D3_GLRLM_feature::calc_LRHGLE(AngledFtrs& af)
{
	af.clear();

	for (int ai = 0; ai < 4; ai++)
	{
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I[i - 1];
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(inten * inten * j * j);
			}
		}
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

void D3_GLRLM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		D3_GLRLM_feature glrlm;
		glrlm.calculate(r);
		glrlm.save_value(r.fvals);
	}
}

// 'afv' is angled feature values
double D3_GLRLM_feature::calc_ave(const std::vector<double>& afv)
{
	if (afv.empty())
		return 0;

	double n = static_cast<double> (afv.size()),
		ave = std::reduce(afv.begin(), afv.end()) / n;

	return ave;
}