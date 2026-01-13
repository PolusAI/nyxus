#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include "3d_glrlm.h"
#include "../environment.h"

using namespace Nyxus;

D3_GLRLM_feature::D3_GLRLM_feature() : FeatureMethod("D3_GLRLM_feature")
{
	provide_features(D3_GLRLM_feature::featureset);
}

const static AngleShift shifts13 [] =
{
	{1,  1,  1},
	{1,  1,  0},
	{1,  1, -1},
	{1,  0,  1},
	{1,  0,  0},
	{1,  0, -1},
	{1, -1,  1},
	{1, -1,  0},
	{1, -1, -1},
	{0,  1,  1},
	{0,  1,  0},
	{0,  1, -1},
	{0,  0,  1}
};

/*static*/ void D3_GLRLM_feature::gather_rl_zones (
	// out
	std::vector<std::pair<PixIntens, int>> &Zones, 
	// in
	const AngleShift &sh,
	SimpleCube <PixIntens> &D, 
	PixIntens zeroI)
{
	size_t w = D.width(),
		h = D.height(),
		d = D.depth();

	// Number of zones
	const int VISITED = -1;

	for (int zslice = 0; zslice < d; zslice++)
	{
		for (int row = 0; row < h; row++)
		{
			for (int col = 0; col < w; col++)
			{
				// Find a non-blank pixel
				auto pi = D.zyx(zslice, row, col);
				if (pi == 0 || int(pi) == VISITED)
					continue;

				// Found a non-blank pixel. Find same-intensity neighbourhood of it.
				std::vector<std::tuple<int, int, int>> history;
				int x = col, y = row, z = zslice;
				int zoneArea = 1;
				D.zyx(z, y, x) = VISITED;

				// State machine scanning the rest of the cluster
				for (;;)
				{
					if (D.safe(z+sh.dz, y+sh.dy, x+sh.dx) && D.zyx(z+sh.dz, y+sh.dy, x+sh.dx) != VISITED && D.zyx(z+sh.dz, y+sh.dy, x+sh.dx) == pi)
					{
						D.zyx(z+sh.dz, y+sh.dy, x+sh.dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance 
						z += sh.dz;
						y += sh.dy;
						x += sh.dx;
						// Proceed
						continue;
					}

					// Return from the branch
					if (history.size() > 0)
					{
						// Recollect the coordinate where we diverted from
						std::tuple<int, int, int> prev = history[history.size() - 1];
						history.pop_back();
					}

					// We are done exploring this cluster
					break;
				}

				// --3
				std::pair <PixIntens, int> zo = { pi, zoneArea };
				Zones.push_back(zo);
			}
		}
	}
}

void D3_GLRLM_feature::calculate (LR& r, const Fsettings& s)
{
	n_angles_ = sizeof(shifts13) / sizeof(AngleShift);

	//==== Clear the feature values buffers
	clear_buffers();

	auto minI = r.aux_min,
		maxI = r.aux_max;

	// intercept blank ROIs
	if (minI == maxI)
	{
		// insert a non-NAN value for all 4 angles to make the output expecting 4-angled values happy
		double w = STNGS_NAN(s);
		angled_SRE.resize (n_angles_, w);
		angled_LRE.resize (n_angles_, w);
		angled_GLN.resize (n_angles_, w);
		angled_GLNN.resize (n_angles_, w);
		angled_RLN.resize (n_angles_, w);
		angled_RLNN.resize (n_angles_, w);
		angled_RP.resize (n_angles_, w);
		angled_GLV.resize (n_angles_, w);
		angled_RV.resize (n_angles_, w);
		angled_RE.resize (n_angles_, w);
		angled_LGLRE.resize (n_angles_, w);
		angled_HGLRE.resize (n_angles_, w);
		angled_SRLGLE.resize (n_angles_, w);
		angled_SRHGLE.resize (n_angles_, w);
		angled_LRLGLE.resize (n_angles_, w);
		angled_LRHGLE.resize (n_angles_, w);
		return;
	}

	// grey-bin

	int w = r.aux_image_cube.width(),
		h = r.aux_image_cube.height(),
		d = r.aux_image_cube.depth();

	SimpleCube <PixIntens> G;
	G.allocate (w,h,d);

	auto greyInfo = STNGS_GLRLM_GREYDEPTH(s);
	if (STNGS_IBSI(s))
		greyInfo = 0;

	bin_intensities_3d (G, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

	// sorted intensities

	std::vector <PixIntens> I;
	if (ibsi_grey_binning(greyInfo))
	{
		auto n_ibsi_levels = *std::max_element (G.begin(), G.end());
		I.resize (n_ibsi_levels);
		for (int i=0; i<n_ibsi_levels; i++)
			I[i] = i+1;
	}
	else // radiomics and matlab
	{
		std::unordered_set<PixIntens> U (G.begin(), G.end());
		U.erase (0);	// discard intensity '0'
		I.assign (U.begin(), U.end());
		std::sort (I.begin(), I.end());
	}

	//==== Iterate angles 
	for (const AngleShift & ash : shifts13)
	{
		// a scratch copy

		SimpleCube <PixIntens> D(G);

		// find zones

		std::vector <std::pair<PixIntens, int>> Zones;
		PixIntens zeroI = 0;
		D3_GLRLM_feature::gather_rl_zones (Zones, ash, D, zeroI);

		// zone stats
		int maxZoneArea = 0;
		for (const std::pair <PixIntens, int> &zo : Zones)
			maxZoneArea = (std::max) (maxZoneArea, zo.second);

		//==== create the matrix

		int Ng = STNGS_IBSI(s) ? *std::max_element(I.begin(), I.end()) : I.size();
		int Nr = maxZoneArea;
		int Nz = (int) Zones.size();
		size_t Np = r.raw_pixels_3D.size();

		// --allocate the matrix
		P_matrix P;
		P.allocate (Nr /*cols*/, Ng /*rows*/);

		// --iterate zones and fill the matrix
		for (const auto &z : Zones)
		{
			auto inten = z.first;

			// row (grey level)
			int row = -1;
			if (STNGS_IBSI(s))
				row = inten - 1;
			else
			{
				auto lower = std::lower_bound(I.begin(), I.end(), inten);	// enjoying sorted vector I
				row = int(lower - I.begin());
			}
			
			// col
			int col = z.second - 1;	// 0-based => -1
			
			// update the matrix
			auto &k = P.xy (col, row);
			k++;
		}

		// GLRL-matrix stats
		double sum = 0;
		for (auto p : P)
			sum += p;

		double sre = calc_SRE (P, sum),
			lre = calc_LRE (P, sum),
			gln = calc_GLN (P, sum),
			glnn = calc_GLNN (P, sum),
			rln = calc_RLN (P, sum),
			rlnn = calc_RLNN (P, sum),
			rp = Np > 0 ? sum/double(Np) : STNGS_NAN(s),
			glv = calc_GLV (P, I, sum),
			rv = calc_RV (P, sum),
			re = calc_RE (P, sum),
			lglre = calc_LGLRE (P, I, sum),
			hglre = calc_HGLRE (P, I, sum),
			srlgle = calc_SRLGLE (P, I, sum),
			srhgle = calc_SRHGLE (P, I, sum),
			lrlgle = calc_LRLGLE (P, I, sum),
			lrhgle = calc_LRHGLE (P, I, sum);
		angled_SRE.push_back (sre);
		angled_LRE.push_back (lre);
		angled_GLN.push_back (gln);
		angled_GLNN.push_back (glnn);
		angled_RLN.push_back (rln);
		angled_RLNN.push_back(rlnn);
		angled_RP.push_back (rp);
		angled_GLV.push_back (glv);
		angled_RV.push_back (rv);
		angled_RE.push_back (re);
		angled_LGLRE.push_back (lglre);
		angled_HGLRE.push_back (hglre);
		angled_SRLGLE.push_back (srlgle);
		angled_SRHGLE.push_back (srhgle);
		angled_LRLGLE.push_back (lrlgle);
		angled_LRHGLE.push_back (lrhgle);
	} // angles

}

void D3_GLRLM_feature::clear_buffers()
{
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

double D3_GLRLM_feature::calc_SRE (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.;
	std::vector<double> rj(Nr + 1, 0.);
	for (int i = 1; i <= Ng; ++i) 
	{
		for (int j = 1; j <= Nr; ++j) 
		{
			rj[j] += P.matlab(i, j);
		}
	}

	for (int j = 1; j <= Nr; ++j) 
	{
		f += rj[j] / double(j * j);
	}

	double retval = f / sum_p;
	return retval;
}

// 2. Long Run Emphasis 

double D3_GLRLM_feature::calc_LRE (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nr; j++)
		{
			f += P.matlab(i, j) * double(j * j);
		}
	}

	double retval = f / sum_p;
	return retval;
}

// 3. Gray Level Non-Uniformity 

double D3_GLRLM_feature::calc_GLN (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

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

	double retval = f / sum_p;
	return retval;
}

// 4. Gray Level Non-Uniformity Normalized 

double D3_GLRLM_feature::calc_GLNN (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

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

	double retval = f / (sum_p * sum_p);
	return retval;
}

// 5. Run Length Non-Uniformity

double D3_GLRLM_feature::calc_RLN (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.0;
	for (int x=1; x<= Nr; x++)
	{
		double sumI = 0.0;	// total of intensities at given run-length

		for (int y=1; y<=Ng; y++)
			sumI += P.matlab (y,x);

		f += sumI * sumI;
	}

	double retval = f / sum_p;
	return retval;
}


// 6. Run Length Non-Uniformity Normalized 

double D3_GLRLM_feature::calc_RLNN (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

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

	double retval = f / (sum_p * sum_p);
	return retval;
}

// 7. Run Percentage (trivial math)

// 8. Gray Level Variance 

double D3_GLRLM_feature::calc_GLV (const SimpleMatrix<int> &P, const std::vector<PixIntens> &I, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		auto inten = I[i - 1];
		for (int j = 1; j <= Nr; j++)
		{
			mu += P.matlab(i, j) / sum_p * inten;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		auto inten = I[i - 1];
		for (int j = 1; j <= Nr; j++)
		{
			double mu2 = (inten - mu) * (inten - mu);
			f += P.matlab(i, j) / sum_p * mu2;
		}
	}
	return f;
}

// 9. Run Variance 

double D3_GLRLM_feature::calc_RV (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nr; j++)
		{
			mu += P.matlab(i, j) / sum_p * j;
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nr; j++)
		{
			double mu2 = (j - mu) * (j - mu);
			f += P.matlab(i, j) / sum_p * mu2;
		}
	}

	return f;
}

// 10. Run Entropy 

double D3_GLRLM_feature::calc_RE (const SimpleMatrix<int> &P, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Nr; j++)
		{
			double entrTerm = fast_log10(P.matlab(i, j) / sum_p + EPS) / LOG10_2;
			f += P.matlab(i, j) / sum_p * entrTerm;
		}
	}
	double retval = -f;
	return retval;
}

// 11. Low Gray Level Run Emphasis 

double D3_GLRLM_feature::calc_LGLRE (const SimpleMatrix<int> &P, const std::vector<PixIntens> &I, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		auto inten = I[i - 1];
		for (int j = 1; j <= Nr; j++)
		{
			f += P.matlab(i, j) / double(inten * inten);
		}
	}
	double retval = f / double(sum_p);
	return retval;
}

// 12. High Gray Level Run Emphasis 

double D3_GLRLM_feature::calc_HGLRE (const SimpleMatrix<int> &P, const std::vector<PixIntens> &I, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		auto inten = I[i - 1];
		for (int j = 1; j <= Nr; j++)
		{
			f += P.matlab(i, j) * double(inten) * double(inten);
		}
	}
	double retval = f / double(sum_p);
	return retval;
}

// 13. Short Run Low Gray Level Emphasis 

double D3_GLRLM_feature::calc_SRLGLE (const SimpleMatrix<int> &P, const std::vector<PixIntens> &I, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			auto inten = I [i-1];
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) / double(inten * inten * j * j);
			}
		}
		double retval = f / sum_p;
		return retval;
}

// 14. Short Run High Gray Level Emphasis 

double D3_GLRLM_feature::calc_SRHGLE (const SimpleMatrix<int> &P, const std::vector<PixIntens> &I, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();
	
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		auto inten = I[i - 1];
		for (int j = 1; j <= Nr; j++)
		{
			f += P.matlab(i, j) * double(inten * inten) / double(j * j);
		}
	}
	double retval = f / sum_p;
	return retval;
}

// 15. Long Run Low Gray Level Emphasis 

double D3_GLRLM_feature::calc_LRLGLE (const SimpleMatrix<int> &P, const std::vector<PixIntens> &I, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		auto inten = I[i - 1];
		for (int j = 1; j <= Nr; j++)
		{
			f += P.matlab(i, j) * double(j * j) / double(inten * inten);
		}
	}
	double retval = f / sum_p;
	return retval; 
}

// 16. Long Run High Gray Level Emphasis 

double D3_GLRLM_feature::calc_LRHGLE (const SimpleMatrix<int> &P, const std::vector<PixIntens> &I, const double sum_p)
{
	if (sum_p == 0)
		return 0.0;

	int Ng = P.height(),
		Nr = P.width();

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		auto inten = I[i - 1];
		for (int j = 1; j <= Nr; j++)
		{
			f += P.matlab(i, j) * double(inten * inten * j * j);
		}
	}
	double retval = f / sum_p;
	return retval;
}

/*static*/ void D3_GLRLM_feature::reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings& s, const Dataset& _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		D3_GLRLM_feature glrlm;
		glrlm.calculate (r, s);
		glrlm.save_value (r.fvals);
	}
}

/*static*/ void D3_GLRLM_feature::extract (LR& r, const Fsettings& s)
{
	D3_GLRLM_feature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

