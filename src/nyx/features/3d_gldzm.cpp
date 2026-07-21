#include <algorithm>
#include <climits>
#include <set>
#include <stack>
#include <unordered_set>
#include <vector>
#include "../environment.h"
#include "3d_gldzm.h"
#include "image_cube.h"
#include "streaming_ccl.h"


int D3_GLDZM_feature::n_levels = 0;

D3_GLDZM_feature::D3_GLDZM_feature() : FeatureMethod("D3_GLDZM_feature")
{
	provide_features (D3_GLDZM_feature::featureset);
}

void D3_GLDZM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings& s, const Dataset & ds)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_GLDZM_feature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}

/*static*/ void D3_GLDZM_feature::extract (LR& r, const Fsettings& s)
{
	D3_GLDZM_feature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void D3_GLDZM_feature::clear_buffers()
{
	f_SDE =
		f_LDE =
		f_LGLZE =
		f_HGLZE =
		f_SDLGLE =
		f_SDHGLE =
		f_LDLGLE =
		f_LDHGLE =
		f_GLNU =
		f_GLNUN =
		f_ZDNU =
		f_ZDNUN =
		f_ZP =
		f_GLM =
		f_GLV =
		f_ZDM =
		f_ZDV =
		f_ZDE =
		f_GLE = 0;
}

void D3_GLDZM_feature::prepare_GLDZM_matrix_kit (SimpleMatrix<unsigned int>& GLDZM, int& Ng, int& Nd, std::vector<PixIntens>& greysLUT, LR& r, const Fsettings& s)
{
	//==== Compose the distance matrix

	// -- Zones (intensity clusters)
	std::vector<IDZ_cluster_indo> Z;

	// -- binned intensities
	SimpleCube<PixIntens> D;
	D.allocate (r.aux_image_cube.width(), r.aux_image_cube.height(), r.aux_image_cube.depth());

	auto greyInfo = STNGS_NGREYS(s); // former Nyxus::theEnvironment.get_coarse_gray_depth()
	auto greyInfo_localFeature = D3_GLDZM_feature::n_levels;
	if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
		greyInfo = greyInfo_localFeature;
	if (STNGS_IBSI(s))	// former Nyxus::theEnvironment.ibsi_compliance
		greyInfo = 0;

	auto& imR = r.aux_image_cube;
	bin_intensities_3d (D, imR, r.aux_min, r.aux_max, greyInfo);

	// allocate intensities matrix
	std::vector<PixIntens> I;
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

	//==== Find zones
	constexpr int huge = INT_MAX;	// Value greater than any pixel's distance to ROI border
	const int VISITED = -1;
	for (int row = 0; row < D.height(); row++)
		for (int col = 0; col < D.width(); col++)
			for (int dep = 0; dep < D.depth(); dep++)
		{
			auto inten = D.zyx (dep, row, col);

			// Skip visited pixels
			if (int(inten) == VISITED)
				continue;

			// Skip 0-intensity pixels (usually out of mask pixels)
			if (ibsi_grey_binning(greyInfo))
				if (inten == 0)
					continue;

			// Once found a nonblank pixel, explore its same-intensity neighbourhood (aka "zone") pixel's distance 
			// to the image border and figure out the whole zone's metric - minimum member pixel's distance to the border.
			std::stack<std::tuple<int, int, int>> parentstack;	// x,y,z

			int x = col,
				y = row,
				z = dep;

			// Initial zone size
			int zoneSize = 1;	// once found a never-visited pixel, we already have a 1-pixel zone

			// Prepare an initial approximation of zone's distance to border
			int zoneMetric = dist2border <SimpleCube<PixIntens>>(D, x, y, z);

			// Scan the neighborhood of pixel (x,y)
			for (;;)
			{
				//==== Calculate the metric of this pixel. It may happen to be the only pixel of a zone

				// Prevent rescanning: mark eroded pixels with 'VISITED'. (The goal is to erode the whole zone.)
				D.zyx(z, y, x) = VISITED;

				//==== Check if zone continues to the East
				int _x = x + 1,
					_y = y,
					_z = z;
				if (D.safe(_z, _y, _x) && D.zyx(_z, _y, _x) != VISITED && D.zyx(_z, _y, _x) == inten)
				{

					// Store pixel (x+1,y)'s parent pixel pisition
					parentstack.push({ x,y,z });

					// Update zone's metric
					int dist2roi = dist2border <SimpleCube<PixIntens>>(D, _x, _y, _z);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					x = _x;
					continue;
				}

				//==== Check if zone continues to the South
				_x = x;
				_y = y + 1;
				_z = z;
				if (D.safe(_z, _y, _x) && D.zyx(_z, _y, _x) != VISITED && D.zyx(_z, _y, _x) == inten)
				{

					// Store pixel (x,y+1)'s parent pixel pisition
					parentstack.push({ x,y,z });

					// Update zone's metric
					int dist2roi = dist2border <SimpleCube<PixIntens>>(D, _x, _y, _z);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					y = _y;
					continue;
				}

				//==== Check if zone continues to the West
				_x = x - 1;
				_y = y;
				_z = z;
				if (D.safe(_z, _y, _x) && D.zyx(_z, _y, _x) != VISITED && D.zyx(_z, _y, _x) == inten)
				{
					// Store pixel (x-1,y)'s parent pixel pisition
					parentstack.push({ x,y,z });

					// Update zone's metric
					int dist2roi = dist2border <SimpleCube<PixIntens>>(D, _x, _y, _z);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					x = _x;
					continue;
				}

				//==== Check if zone continues to the North
				_x = x;
				_y = y - 1;
				_z = z;
				if (D.safe(_z, _y, _x) && D.zyx(_z, _y, _x) != VISITED && D.zyx(_z, _y, _x) == inten)
				{

					// Store pixel (x,y-1)'s parent pixel pisition
					parentstack.push({ x,y,z });

					// Update zone's metric
					int dist2roi = dist2border <SimpleCube<PixIntens>>(D, _x, _y, _z);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					y = _y;
					continue;
				}

				//==== Check if zone continues upwards
				_x = x;
				_y = y;
				_z = z+1;
				if (D.safe(_z, _y, _x) && D.zyx(_z, _y, _x) != VISITED && D.zyx(_z, _y, _x) == inten)
				{

					// Store pixel (x+1,y)'s parent pixel pisition
					parentstack.push({ x,y,z });

					// Update zone's metric
					int dist2roi = dist2border <SimpleCube<PixIntens>>(D, _x, _y, _z);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					z = _z;
					continue;
				}

				//==== Check if zone continues downwards
				_x = x;
				_y = y;
				_z = z-1;
				if (D.safe(_z, _y, _x) && D.zyx(_z, _y, _x) != VISITED && D.zyx(_z, _y, _x) == inten)
				{

					// Store pixel (x,y+1)'s parent pixel pisition
					parentstack.push({ x,y,z });

					// Update zone's metric
					int dist2roi = dist2border <SimpleCube<PixIntens>>(D, _x, _y, _z);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					z = _z;
					continue;
				}

				// We are done exploring pixel's potential neighborhood. There might happen a zone or not (just this pixel)
				if (parentstack.empty() == false)
				{
					// Not a trivial (single-pixel) zone

					// Restore the last saved parent position as current
					// in order to get ahold the terminal pixel's parent who hopefully has other children
					auto parent_xy = parentstack.top();
					parentstack.pop();
					x = std::get<0>(parent_xy);
					y = std::get<1>(parent_xy);
					z = std::get<2>(parent_xy);
				}
				else
					// Empty 'parentstack' indicates that no neighbors ("children") of pixel (x,y) are found or have all been ingested. 
					// We are good to register this zone and proceed with searching another one.
					break;
			}

			// At this point, 'parentstack' is expected to be empty
			{
				// Done scanning the whole zone. Register it
				IDZ_cluster_indo clu = { inten, zoneMetric, zoneSize };
				Z.push_back(clu);
			}
		}

	//==== Fill the zonal metric matrix

	// -- number of discrete intensity values in the image
	Ng = (int)I.size();

	// -- max zone distance to ROI or image border
	Nd = 0;
	for (auto& z : Z)
		Nd = std::max(Nd, std::get<1>(z));

	// -- Set to vector to be able to know each intensity's index
	greysLUT.clear();
	for (auto grey : I)
		greysLUT.push_back(grey);
	std::sort(greysLUT.begin(), greysLUT.end());

	// -- Zone intensity -to- zone distance matrix
	GLDZM.allocate(Nd, Ng);	// Ng rows, Nd columns
	GLDZM.fill(0);
	calc_gldzm_matrix(GLDZM, Z, greysLUT);
}

void D3_GLDZM_feature::calc_gldzm_matrix (SimpleMatrix<unsigned int>& GLDZM, const std::vector<IDZ_cluster_indo>& Z, const std::vector<PixIntens>& I)
{
	int i = 0;
	for (auto& z : Z)
	{
		// row. Gray tones are sparse so we need to find indices of tones in 'Z' and use them as rows of P-matrix
		auto iter = std::find(I.begin(), I.end(), std::get<0>(z));
		int row = (int)(iter - I.begin());
		// column (a distance). Distances are dense \in [1,Nd]
		int col = std::get<1>(z) - 1;	// 0-based => -1
		auto& k = GLDZM.yx(row, col);
		k++;
	}
}

template <class Cube> int D3_GLDZM_feature::dist2border (Cube & I, const int x, const int y, const int z)
{
	// scan left
	int dist2l = 0;
	for (int x0 = x - 1; x0 >= 0; x0--)
		if (I.zyx(z, y, x0) == 0 || x0 == 0)	// we're at the ROI border or left margin (x0==0) 
		{
			dist2l = x - x0;
			break;
		}
	// scan right
	int dist2r = 0, w = I.width();
	for (int x0 = x + 1; x0 < w; x0++)
		if (I.zyx(z, y, x0) == 0 || x0 == w - 1)	// we're at the ROI border or right margin (x0==w-1)
		{
			dist2r = x0 - x;
			break;
		}
	// scan up
	int dist2t = 0;
	for (int y0 = y - 1; y0 >= 0; y0--)
		if (I.zyx(z, y0, x) == 0 || y0 == 0)	// we're at the ROI border or top margin (y0==0)
		{
			dist2t = y - y0;
			break;
		}
	// scan down
	int dist2b = 0, h = I.height();
	for (int y0 = y + 1; y0 < h; y0++)
		if (I.zyx(z, y0, x) == 0 || y0 == h - 1)	// we're at the ROI border or bottom margin (y0==h-1)
		{
			dist2b = y0 - y;
			break;
		}
	// make distances 1-based
	dist2l++;
	dist2r++;
	dist2t++;
	dist2b++;

	// result
	int retval = std::min(std::min(std::min(dist2l, dist2r), dist2t), dist2b);
	if (retval == 0)
		retval = 1;	// Requirement of GLDZM: pixel on the border is within distance 1 from ROI border
	return retval;
}

template <class Imgmatrx> void D3_GLDZM_feature::calc_row_and_column_sum_vectors (std::vector<double>& Mx, std::vector<double>& Md, Imgmatrx& P, const int Ng, const int Nd, const std::vector<PixIntens>& greysLUT)
{
	// Sum distances of each grey levels
	Mx.resize(Ng);
	for (int g = 0; g < Ng; g++)
	{
		double sumD = 0;
		for (int d = 0; d < Nd; d++)
			sumD += P.yx(g, d);
		Mx[g] = sumD;
	}

	// Sum grey levels of each distance
	Md.resize(Nd);
	for (int d = 0; d < Nd; d++)
	{
		double sumG = 0;
		for (int g = 0; g < Ng; g++)
		{
			// skip zero intensities
			auto inten = greysLUT[g];
			if (inten == 0)
				continue;

			sumG += P.yx(g, d);
		}
		Md[d] = sumG;
	}
}

template <class Imgmatrx> void D3_GLDZM_feature::calc_features (const std::vector<double>& Mx, const std::vector<double>& Md, Imgmatrx& P, const std::vector<PixIntens>& greysLUT, unsigned int roi_area)
{
	int Ng = Mx.size(),
		Nd = Md.size();

	// Ns is the number of realised zones
	double Ns = 0;
	for (int g = 0; g < Ng; g++)
		for (int d = 0; d < Nd; d++)
			if (greysLUT[g])	// skip zero grey level zones
				Ns += P.yx(g, d);

	// Nv is the number of potential zones
	double Nv = roi_area;

	for (int d_ = 0; d_ < Nd; d_++)
	{
		double d = (double)(d_ + 1);
		double m = Md[d_];
		f_SDE += m / d / d;			// Small Distance Emphasis = \frac{1}{N_s} \sum_d \frac{m_d}{d^2}
		f_LDE += d * d * m;			// Large Distance Emphasis = \frac{1}{N_s} \sum_d d^2 m_d 
		f_ZDNU += m * m;			// Zone Distance Non-Uniformity = \frac{1}{N_s} \sum_d m_d^2
									// Zone Distance Non-Uniformity Normalized = \frac{1}{N_s^2} \sum_d m_d^2
	}

	f_SDE /= Ns;
	f_LDE /= Ns;
	f_ZDNU /= Ns;
	f_ZDNUN = f_ZDNU / Ns;

	for (int g = 0; g < Ng; g++)
	{
		// skip zero intensities in general and to prevent arithmetic overflow
		if (greysLUT[g] == 0)
			continue;

		double g_ = (double)greysLUT[g];
		double x = Mx[g];
		f_LGLZE += x / (g_ * g_);	// Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \frac{m_x}{x^2}
		f_HGLZE += (g_ * g_) * x;	// High Grey Level Emphasis = \frac{1}{N_s} \sum_x x^2 m_x
		f_GLNU += x * x;			// Grey Level Non-Uniformity = \frac{1}{N_s} \sum_x m_x^2
	}
	f_LGLZE /= Ns;
	f_HGLZE /= Ns;
	f_GLNU /= Ns;
	f_GLNUN = f_GLNU / Ns;	// Grey Level Non-Uniformity Normalized = \frac{1}{N_s^2} \sum_x m_x^2

	for (int g = 0; g < Ng; g++)
		for (int d = 0; d < Nd; d++)
		{
			// skip zero intensities in general and to prevent arithmetic overflow
			if (greysLUT[g] == 0)
				continue;

			double g_ = (double)greysLUT[g],
				d_ = double(d + 1);
			double p = P.yx(g, d);
			f_SDLGLE += p / g_ / g_ / d_ / d_;	// Small Distance Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{ m_{x,d}}{x^2 d^2}
			f_SDHGLE += g_ * g_ * p / d_ / d_;	// Small Distance High Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{x^2  m_{x,d}}{d^2}
			f_LDLGLE += d_ * d_ * p / g_ / g_;	// Large Distance Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{d^2 m_{x,d}}{x^2}
			f_LDHGLE += g_ * g_ * d_ * d_ * p;	// Large Distance High Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \x^2 d^2 m_{x,d}
			f_GLM += g_ * p;					// Grey Level Mean = \mu_x = \sum_x \sum_d x p_{x,d}
			f_ZDM += d_ * p;					// Zone Distance Mean = \mu_d = \sum_x \sum_d d p_{x,d} 
			f_ZDE += p / Ns * log2(p / Ns + EPS);	// Zone Distance Entropy = - \sum_x \sum_d p_{x,d} \textup{log}_2 ( p_{x,d} )
		}
	f_SDLGLE /= Ns;
	f_SDHGLE /= Ns;
	f_LDLGLE /= Ns;
	f_LDHGLE /= Ns;
	f_GLM /= Ns;
	f_ZDM /= Ns;
	f_ZDE = -f_ZDE;
	f_ZP = Ns / Nv; // Zone Percentage = \frac{N_s}{N_v} 
					// (ZP measures the fraction of the number of realised zones and the maximum 
					// number of potential zones.)
	f_GLE = f_ZDE;

	for (int g = 0; g < Ng; g++)
		for (int d = 0; d < Nd; d++)
		{
			// skip zero intensities
			if (greysLUT[g] == 0)
				continue;

			// Grey Level Variance = \sum_x \sum_d \left(x - \mu_x \right)^2 p_{x,d}
			double p = P.yx(g, d) / Ns,
				x = (double)greysLUT[g],
				dif = x - f_GLM;
			f_GLV += dif * dif * p;

			// Zone Distance Variance} = \sum_x \sum_d \left(d - \mu_d \right)^2 p_{x,d} 
			double d_ = (double)(d + 1);
			dif = d_ - f_ZDM;
			f_ZDV += dif * dif * p;
		}
}


void D3_GLDZM_feature::calculate (LR& r, const Fsettings& s)
{
	clear_buffers();

	// intercept blank ROIs
	if (r.aux_min == r.aux_max)
	{
		f_SDE =
		f_LDE =
		f_LGLZE =
		f_HGLZE =
		f_SDLGLE =
		f_SDHGLE =
		f_LDLGLE =
		f_LDHGLE =
		f_GLNU =
		f_GLNUN =
		f_ZDNU =
		f_ZDNUN =
		f_ZP =
		f_GLM =
		f_GLV =
		f_ZDM =
		f_ZDV =
		f_ZDE =
		f_GLE = STNGS_NAN(s);

		return;
	}

	// prepare the GLDZM-matrix kit: matrix itself and its dimensions
	std::vector<PixIntens> greyLevelsLUT;
	SimpleMatrix<unsigned int> GLDZM;
	int Ng,	// number of grey levels
		Nd;	// maximum number of non-zero dependencies
	prepare_GLDZM_matrix_kit (GLDZM, Ng, Nd, greyLevelsLUT, r, s);

	// calculate vectors of totals by intensity (Mx) and by distance (Md)
	std::vector<double> Mx, Md;
	calc_row_and_column_sum_vectors (Mx, Md, GLDZM, Ng, Nd, greyLevelsLUT);

	// calculate features, set variables f_GLE, f_GML, f_GLV, etc
	calc_features (Mx, Md, GLDZM, greyLevelsLUT, r.aux_area);
}

void D3_GLDZM_feature::save_value (std::vector<std::vector<double>> & fvals)
{
	fvals[(int)Nyxus::Feature3D::GLDZM_SDE][0] = f_SDE;
	fvals[(int)Nyxus::Feature3D::GLDZM_LDE][0] = f_LDE;
	fvals[(int)Nyxus::Feature3D::GLDZM_LGLZE][0] = f_LGLZE;
	fvals[(int)Nyxus::Feature3D::GLDZM_HGLZE][0] = f_HGLZE;
	fvals[(int)Nyxus::Feature3D::GLDZM_SDLGLE][0] = f_SDLGLE;
	fvals[(int)Nyxus::Feature3D::GLDZM_SDHGLE][0] = f_SDHGLE;
	fvals[(int)Nyxus::Feature3D::GLDZM_LDLGLE][0] = f_LDLGLE;
	fvals[(int)Nyxus::Feature3D::GLDZM_LDHGLE][0] = f_LDHGLE;
	fvals[(int)Nyxus::Feature3D::GLDZM_GLNU][0] = f_GLNU;
	fvals[(int)Nyxus::Feature3D::GLDZM_GLNUN][0] = f_GLNUN;
	fvals[(int)Nyxus::Feature3D::GLDZM_ZDNU][0] = f_ZDNU;
	fvals[(int)Nyxus::Feature3D::GLDZM_ZDNUN][0] = f_ZDNUN;
	fvals[(int)Nyxus::Feature3D::GLDZM_ZP][0] = f_ZP;
	fvals[(int)Nyxus::Feature3D::GLDZM_GLM][0] = f_GLM;
	fvals[(int)Nyxus::Feature3D::GLDZM_GLV][0] = f_GLV;
	fvals[(int)Nyxus::Feature3D::GLDZM_ZDM][0] = f_ZDM;
	fvals[(int)Nyxus::Feature3D::GLDZM_ZDV][0] = f_ZDV;
	fvals[(int)Nyxus::Feature3D::GLDZM_ZDE][0] = f_ZDE;
}

void D3_GLDZM_feature::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {}

// Out-of-core 3D GLDZM. prepare_GLDZM_matrix_kit()'s flood fill is 6-connectivity (face-adjacent
// only -- East/West/North/South/Up/Down, no diagonals) tracking each zone's MINIMUM per-voxel
// distance to the ROI border (zone SIZE is also tracked in-core but never actually consumed by
// calc_gldzm_matrix, so this streaming version doesn't bother). Same streaming two-pass CCL
// approach as GLSZM (docs/13-3d-ooc-glszm-gldzm-design.md), but with only 3 causal neighbours
// (West, North, straight-down-Z) since 6-connectivity halves to 3, and a MIN-combining
// LabelUnionFind instead of a size-summing one.
//
// dist2border() is 2D-only (scans left/right/up/down within the CURRENT Z-plane only to the
// nearest literal-0 voxel or the plane edge) -- a limitation/convention of this implementation,
// replicated exactly, not "fixed". It needs only the current plane's dense buffer, already
// resident in the 2-plane window, so no extra memory cost versus GLSZM.
//
// Background-skip convention is GLDZM's OWN (a THIRD distinct rule, different from both GLRLM's
// unconditional zeroI=0 and GLSZM's matlab-aware zeroI): skip a voxel ONLY when
// ibsi_grey_binning(greyInfo) is true AND its binned value is literal 0 -- for radiomics/matlab
// binning, background is never skipped and forms its own zone(s), exactly mirroring
// prepare_GLDZM_matrix_kit's `if (ibsi_grey_binning(greyInfo)) if (inten==0) continue;`.
void D3_GLDZM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	clear_buffers();

	if (r.aux_min == r.aux_max)
	{
		f_SDE = f_LDE = f_LGLZE = f_HGLZE = f_SDLGLE = f_SDHGLE = f_LDLGLE = f_LDHGLE =
		f_GLNU = f_GLNUN = f_ZDNU = f_ZDNUN = f_ZP = f_GLM = f_GLV = f_ZDM = f_ZDV = f_ZDE = f_GLE = STNGS_NAN(s);
		return;
	}

	// grey-binning selection mirrors prepare_GLDZM_matrix_kit exactly: no GLDZM-specific greydepth
	// setting exists, so it starts from the general STNGS_NGREYS(s), can be overridden by the static
	// D3_GLDZM_feature::n_levels, then forced to 0 (IBSI/no-rescale) when STNGS_IBSI(s) is set.
	auto greyInfo = STNGS_NGREYS(s);
	auto greyInfo_localFeature = D3_GLDZM_feature::n_levels;
	if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
		greyInfo = greyInfo_localFeature;
	if (STNGS_IBSI(s))
		greyInfo = 0;

	PixIntens mn = r.aux_min, mx = r.aux_max;
	const PixIntens bg = TextureFeature::bin_pixel (0, mn, mx, greyInfo);
	const bool skipZero = ibsi_grey_binning (greyInfo);	// GLDZM's own rule: only skip in ibsi mode

	const int W = (int) r.aabb.get_width(),
		H = (int) r.aabb.get_height(),
		Dz = (int) r.aabb.get_z_depth();
	const StatsInt xmin = r.aabb.get_xmin(), ymin = r.aabb.get_ymin(), zmin = r.aabb.get_zmin();
	const size_t nvox = r.raw_voxels_NT.size();
	const bool hasBackground = nvox < (size_t) W * H * Dz;

	// --- grey levels (mirrors prepare_GLDZM_matrix_kit: unique values of the WHOLE binned cube,
	// incl. background, minus literal 0 -- or a dense 1..max linspace for ibsi_grey_binning)
	std::vector<PixIntens> I;
	if (ibsi_grey_binning (greyInfo))
	{
		PixIntens maxbin = 0;
		for (size_t i = 0; i < nvox; i++)
		{
			PixIntens b = TextureFeature::bin_pixel (r.raw_voxels_NT[i].inten, mn, mx, greyInfo);
			if (b > maxbin) maxbin = b;
		}
		I.resize (maxbin);
		for (int i = 0; i < (int) maxbin; i++) I[i] = i + 1;
	}
	else
	{
		std::set<PixIntens> uniq;
		if (hasBackground && bg != 0)
			uniq.insert (bg);
		for (size_t i = 0; i < nvox; i++)
		{
			PixIntens b = TextureFeature::bin_pixel (r.raw_voxels_NT[i].inten, mn, mx, greyInfo);
			if (b != 0) uniq.insert (b);
		}
		I.assign (uniq.begin(), uniq.end());
	}

	// --- streaming 6-connectivity CCL over a 2-plane window, tracking min distance-to-border
	LabelUnionFind uf;
	std::vector<PixIntens> prevPlane, curPlane;
	std::vector<int> prevLabel, curLabel;
	std::vector<Pixel3> slab;

	// 2D-only distance to the nearest literal-0 voxel or the plane edge, within 'plane' -- matches
	// dist2border() exactly (1-based, min of 4 directions, clamped to >=1).
	auto dist2border_plane = [&](const std::vector<PixIntens>& plane, int x, int y) -> long long
	{
		int dist2l = 0;
		for (int x0 = x - 1; x0 >= 0; x0--)
			if (plane[(size_t) y * W + x0] == 0 || x0 == 0) { dist2l = x - x0; break; }
		int dist2r = 0;
		for (int x0 = x + 1; x0 < W; x0++)
			if (plane[(size_t) y * W + x0] == 0 || x0 == W - 1) { dist2r = x0 - x; break; }
		int dist2t = 0;
		for (int y0 = y - 1; y0 >= 0; y0--)
			if (plane[(size_t) y0 * W + x] == 0 || y0 == 0) { dist2t = y - y0; break; }
		int dist2b = 0;
		for (int y0 = y + 1; y0 < H; y0++)
			if (plane[(size_t) y0 * W + x] == 0 || y0 == H - 1) { dist2b = y0 - y; break; }
		dist2l++; dist2r++; dist2t++; dist2b++;
		long long d = (std::min) ((std::min) (dist2l, dist2r), (std::min) (dist2t, dist2b));
		if (d == 0) d = 1;
		return d;
	};

	for (int z = 0; z < Dz; z++)
	{
		curPlane.assign ((size_t) W * H, bg);
		r.raw_voxels_NT.read_slab ((size_t)(zmin + z), slab);
		for (const auto& v : slab)
		{
			int lx = (int) v.x - (int) xmin, ly = (int) v.y - (int) ymin;
			if (lx >= 0 && lx < W && ly >= 0 && ly < H)
				curPlane[(size_t) ly * W + lx] = TextureFeature::bin_pixel (v.inten, mn, mx, greyInfo);
		}
		curLabel.assign ((size_t) W * H, -1);

		for (int y = 0; y < H; y++)
			for (int x = 0; x < W; x++)
			{
				PixIntens pi = curPlane[(size_t) y * W + x];
				if (skipZero && pi == 0)
					continue;

				long long d = dist2border_plane (curPlane, x, y);

				int found = -1;
				if (x > 0 && curPlane[(size_t) y * W + (x - 1)] == pi)
				{
					int lbl = curLabel[(size_t) y * W + (x - 1)];
					if (lbl >= 0) found = lbl;
				}
				if (y > 0 && curPlane[(size_t) (y - 1) * W + x] == pi)
				{
					int lbl = curLabel[(size_t) (y - 1) * W + x];
					if (lbl >= 0) found = (found < 0) ? lbl : uf.union_min (found, lbl);
				}
				if (z > 0 && prevPlane[(size_t) y * W + x] == pi)
				{
					int lbl = prevLabel[(size_t) y * W + x];
					if (lbl >= 0) found = (found < 0) ? lbl : uf.union_min (found, lbl);
				}

				if (found < 0)
					found = uf.make_label (pi, d);
				else
					uf.update_min (found, d);
				curLabel[(size_t) y * W + x] = found;
			}

		prevPlane.swap (curPlane);
		prevLabel.swap (curLabel);
	}

	// --- one zone per distinct root: (intensity, min distance)
	std::vector<std::pair<PixIntens, int>> Zones;
	std::vector<char> seenRoot (uf.num_labels(), 0);
	for (size_t i = 0; i < uf.num_labels(); i++)
	{
		int rt = uf.find ((int) i);
		if (seenRoot[rt]) continue;
		seenRoot[rt] = 1;
		Zones.push_back ({ uf.intensity_of (rt), (int) uf.metric (rt) });
	}

	int Ng = (int) I.size();
	int Nd = 0;
	for (auto& zo : Zones)
		Nd = (std::max) (Nd, zo.second);

	SimpleMatrix<unsigned int> GLDZM;
	GLDZM.allocate (Nd, Ng);	// Ng rows, Nd columns -- matches calc_gldzm_matrix's layout
	GLDZM.fill (0);
	for (auto& zo : Zones)
	{
		int row = (int)(std::lower_bound (I.begin(), I.end(), zo.first) - I.begin());
		int col = zo.second - 1;
		GLDZM.yx (row, col)++;
	}

	std::vector<double> Mx, Md;
	calc_row_and_column_sum_vectors (Mx, Md, GLDZM, Ng, Nd, I);
	calc_features (Mx, Md, GLDZM, I, r.aux_area);
}
