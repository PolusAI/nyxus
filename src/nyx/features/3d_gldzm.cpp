#include <array>
#include <climits>
#include "../environment.h"
#include "3d_gldzm.h"
#include "image_cube.h"


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

// City-block distance from every ROI voxel to the nearest voxel outside the ROI, which is the
// distance IBSI's GLDZM measures. A voxel touching the ROI's surface is at distance 1. Anything
// outside the bounding box is outside the ROI, and that is exact because the box is tight.
// Breadth-first from the surface inwards: the frontier holds the voxels settled at the current
// distance, so every voxel is settled once, at its shortest distance.
void D3_GLDZM_feature::calc_dist2border (SimpleCube<int>& dist, const SimpleCube<unsigned char>& roi_mask)
{
	const int w = roi_mask.width(),
		h = roi_mask.height(),
		d = roi_mask.depth();
	const int UNSETTLED = INT_MAX;

	dist.allocate (w, h, d);
	dist.fill (0);	// off-ROI voxels stay at 0: they are the border this measures against

	// the 6 city-block moves
	static const int mv[6][3] = { {-1,0,0}, {+1,0,0}, {0,-1,0}, {0,+1,0}, {0,0,-1}, {0,0,+1} };

	std::vector<std::array<int, 3>> frontier, next_frontier;	// x,y,z of the voxels settled last

	// distance 1 is every ROI voxel with a move that leaves the ROI or leaves the box
	for (int z = 0; z < d; z++)
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				if (roi_mask.zyx(z, y, x) == 0)
					continue;

				dist.zyx(z, y, x) = UNSETTLED;

				for (int i = 0; i < 6; i++)
				{
					int nx = x + mv[i][0],
						ny = y + mv[i][1],
						nz = z + mv[i][2];
					if (!roi_mask.safe(nz, ny, nx) || roi_mask.zyx(nz, ny, nx) == 0)
					{
						dist.zyx(z, y, x) = 1;
						frontier.push_back ({ x, y, z });
						break;
					}
				}
			}

	// each round settles the ROI voxels one move further in
	for (int step = 2; !frontier.empty(); step++)
	{
		next_frontier.clear();

		for (const auto& v : frontier)
			for (int i = 0; i < 6; i++)
			{
				int nx = v[0] + mv[i][0],
					ny = v[1] + mv[i][1],
					nz = v[2] + mv[i][2];
				if (!roi_mask.safe(nz, ny, nx) || roi_mask.zyx(nz, ny, nx) == 0)
					continue;
				if (dist.zyx(nz, ny, nx) != UNSETTLED)
					continue;

				dist.zyx(nz, ny, nx) = step;
				next_frontier.push_back ({ nx, ny, nz });
			}

		frontier.swap (next_frontier);
	}
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

	// -- ROI mask over the bounding box. The binned cube cannot stand in for it: matlab binning
	// sends intensity 0 to level 1, so after binning the background filling the rest of the box is
	// indistinguishable from a genuine level-1 ROI voxel. The 2D twin gldzm.cpp masks for the same
	// reason. A byte per bounding-box voxel is the cheapest element type that carries the answer.
	SimpleCube<unsigned char> roi_mask;
	roi_mask.allocate (D.width(), D.height(), D.depth());
	roi_mask.fill (0);
	auto xmin = r.aabb.get_xmin(),
		ymin = r.aabb.get_ymin(),
		zmin = r.aabb.get_zmin();
	for (const auto& p : r.raw_pixels_3D)
		roi_mask.zyx (int(p.z - zmin), int(p.y - ymin), int(p.x - xmin)) = 1;

	// allocate intensities matrix. The grey levels are the ROI's, not the bounding box's.
	std::vector<PixIntens> I;
	if (ibsi_grey_binning(greyInfo))
	{
		PixIntens n_ibsi_levels = 0;
		for (int z = 0; z < D.depth(); z++)
			for (int y = 0; y < D.height(); y++)
				for (int x = 0; x < D.width(); x++)
					if (roi_mask.zyx(z, y, x))
						n_ibsi_levels = std::max (n_ibsi_levels, D.zyx(z, y, x));
		I.resize (n_ibsi_levels);
		for (PixIntens i = 0; i < n_ibsi_levels; i++)
			I[i] = i + 1;
	}
	else // radiomics and matlab
	{
		std::unordered_set<PixIntens> U;
		for (int z = 0; z < D.depth(); z++)
			for (int y = 0; y < D.height(); y++)
				for (int x = 0; x < D.width(); x++)
					if (roi_mask.zyx(z, y, x))
						U.insert (D.zyx(z, y, x));
		U.erase(0);	// discard intensity '0'
		I.assign(U.begin(), U.end());
		std::sort(I.begin(), I.end());
	}

	//==== Distance of every ROI voxel to the ROI border. One pass over the box, not a lookup per
	// zone member: a zone's metric is a minimum over its voxels, so every ROI voxel's distance is
	// read at least once anyway.
	SimpleCube<int> dist;
	calc_dist2border (dist, roi_mask);

	//==== Find zones
	// A zone is a 26-connected component of one grey level within the ROI -- the connectivity IBSI
	// defines for GLDZM and the one GLSZM uses for the same notion of a zone. The zone's distance
	// metric is the smallest distance to the border any of its voxels has.
	std::vector<std::array<int, 3>> stack;	// x,y,z of the zone voxels whose neighbourhood is still to be scanned

	for (int dep = 0; dep < D.depth(); dep++)
		for (int row = 0; row < D.height(); row++)
			for (int col = 0; col < D.width(); col++)
			{
				// zones are grown over the ROI; the background filling the rest of the bounding box
				// is not part of any of them
				if (roi_mask.zyx(dep, row, col) == 0)
					continue;

				auto inten = D.zyx (dep, row, col);

				// Grey level 0 is the one level the LUT never carries -- the IBSI branch fills it
				// with 1..max and the other erases 0 -- so a zone of it would have no row in the
				// matrix. It reaches here only from the binning schemes that leave a zero at zero,
				// which the MATLAB one does not.
				if (inten == 0)
					continue;

				// Taking a voxel into a zone clears its mask bit. That both prevents rescanning and
				// keeps "is this a zone member" one lookup: a voxel already in a zone is no longer
				// available to another.
				roi_mask.zyx(dep, row, col) = 0;

				int zoneSize = 1;
				int zoneMetric = dist.zyx (dep, row, col);

				stack.clear();
				stack.push_back ({ col, row, dep });

				while (!stack.empty())
				{
					auto v = stack.back();
					stack.pop_back();

					for (int dz = -1; dz <= 1; dz++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dx = -1; dx <= 1; dx++)
							{
								if (dx == 0 && dy == 0 && dz == 0)
									continue;

								int _x = v[0] + dx,
									_y = v[1] + dy,
									_z = v[2] + dz;
								if (!roi_mask.safe(_z, _y, _x) || roi_mask.zyx(_z, _y, _x) == 0)
									continue;
								if (D.zyx(_z, _y, _x) != inten)
									continue;

								roi_mask.zyx(_z, _y, _x) = 0;
								zoneSize++;
								zoneMetric = std::min (zoneMetric, dist.zyx(_z, _y, _x));
								stack.push_back ({ _x, _y, _z });
							}
				}

				// Done scanning the whole zone. Register it
				IDZ_cluster_indo clu = { inten, zoneMetric, zoneSize };
				Z.push_back(clu);
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

void D3_GLDZM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	calculate (r, s);
}
