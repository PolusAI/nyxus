#include <limits.h>
#include <stack>
#include "gldzm.h"
#include "../environment.h"

using namespace Nyxus;

int GLDZMFeature::n_levels = 0;

GLDZMFeature::GLDZMFeature() : FeatureMethod("GLDZMFeature")
{
	provide_features (GLDZMFeature::featureset);
}

void GLDZMFeature::clear_buffers()
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

void GLDZMFeature::calc_gldzm_matrix (SimpleMatrix<unsigned int> & GLDZM, const std::vector<IDZ_cluster_indo> & Z, const std::vector<PixIntens> & I)
{
	int i = 0;
	for (auto& z : Z)
	{
		// row. Gray tones are sparse so we need to find indices of tones in 'Z' and use them as rows of P-matrix
		auto iter = std::find (I.begin(), I.end(), std::get<0>(z));
		int row = (int)(iter - I.begin());
		// column (a distance). Distances are dense \in [1,Nd]
		int col = std::get<1>(z) - 1;	// 0-based => -1
		auto& k = GLDZM.yx(row, col);
		k++;
	}
}

void GLDZMFeature::prepare_GLDZM_matrix_kit (SimpleMatrix<unsigned int>& GLDZM, int& Ng, int& Nd, std::vector<PixIntens>& greysLUT, LR& r, int n_greys, bool ibsi)
{
	//==== Compose the distance matrix

	// -- Zones (intensity clusters)
	std::vector<IDZ_cluster_indo> Z;

	ImageMatrix M;
	M.allocate(r.aux_image_matrix.width, r.aux_image_matrix.height);
	pixData& D = M.WriteablePixels();

	auto greyInfo = n_greys; // former theEnvironment.get_coarse_gray_depth()
	auto greyInfo_localFeature = GLDZMFeature::n_levels;
	if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
		greyInfo = greyInfo_localFeature;
	if (ibsi)	// former Nyxus::theEnvironment.ibsi_compliance
		greyInfo = 0;

	auto& imR = r.aux_image_matrix.ReadablePixels();
	bin_intensities (D, imR, r.aux_min, r.aux_max, greyInfo);

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
	for (int row = 0; row < M.height; row++)
		for (int col = 0; col < M.width; col++)
		{
			auto inten = D.yx(row, col);

			// Skip visited pixels
			if (int(inten) == VISITED)
				continue;

			// Skip 0-intensity pixels (usually out of mask pixels)
			if (ibsi_grey_binning(greyInfo))
				if (inten == 0)
					continue;

			// Once found a nonblank pixel, explore its same-intensity neighbourhood (aka "zone") pixel's distance 
			// to the image border and figure out the whole zone's metric - minimum member pixel's distance to the border.
			std::stack<std::tuple<int, int>> parentstack;

			int x = col,
				y = row;

			// Initial zone size
			int zoneSize = 1;	// once found a never-visited pixel, we already have a 1-pixel zone

			// Prepare an initial approximation of zone's distance to border
			int zoneMetric = dist2border <pixData> (D, x, y); 

			// Scan the neighborhood of pixel (x,y)
			for (;;)
			{
				//==== Calculate the metric of this pixel. It may happen to be the only pixel of a zone
				
				// Prevent rescanning: mark eroded pixels with 'VISITED'. (The goal is to erode the whole zone.)
				D.yx (y,x) = VISITED;

				//==== Check if zone continues to the East
				int _x = x+1, 
					_y = y;
				if (D.safe(_y,_x) && D.yx(_y,_x) != VISITED && D.yx(_y,_x) == inten)
				{

					// Store pixel (x+1,y)'s parent pixel pisition
					parentstack.push ({x,y});

					// Update zone's metric
					int dist2roi = dist2border <pixData>(D, _x, _y);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					x = _x;
					continue;
				}

				//==== Check if zone continues to the South
				_x = x;
				_y = y+1;
				if (D.safe(_y, _x) && D.yx(_y, _x) != VISITED && D.yx(_y, _x) == inten)
				{

					// Store pixel (x,y+1)'s parent pixel pisition
					parentstack.push ({x,y});	

					// Update zone's metric
					int dist2roi = dist2border <pixData>(D, _x, _y);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					y = _y;
					continue;
				}

				//==== Check if zone continues to the West
				_x = x-1;
				_y = y;
				if (D.safe(_y, _x) && D.yx(_y, _x) != VISITED && D.yx(_y, _x) == inten)
				{

					// Store pixel (x-1,y)'s parent pixel pisition
					parentstack.push ({x,y});	

					// Update zone's metric
					int dist2roi = dist2border <pixData>(D, _x, _y);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					x = _x;
					continue;
				}

				//==== Check if zone continues to the North
				_x = x;
				_y = y-1;
				if (D.safe(_y, _x) && D.yx(_y, _x) != VISITED && D.yx(_y, _x) == inten)
				{

					// Store pixel (x,y-1)'s parent pixel pisition
					parentstack.push ({x,y});	

					// Update zone's metric
					int dist2roi = dist2border <pixData> (D, _x, _y);
					zoneMetric = std::min (zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Make the new neighborhood pixel current parent
					y = _y;
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
					x = std::get<0> (parent_xy);
					y = std::get<1> (parent_xy);
				}
				else
					// Empty 'parentstack' indicates that no neighbors ("children") of pixel (x,y) are found or have all been ingested. 
					// We are good to register this zone and proceed with searching another one.
					break;
			}

			// At this point, 'parentstack' is expected to be empty
			{
				// Done scanning the whole zone. Register it
				IDZ_cluster_indo clu = {inten, zoneMetric, zoneSize};
				Z.push_back (clu);
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
	std::sort (greysLUT.begin(), greysLUT.end());

	// -- Zone intensity -to- zone distance matrix
	GLDZM.allocate (Nd, Ng);	// Ng rows, Nd columns
	GLDZM.fill (0);
	calc_gldzm_matrix (GLDZM, Z, greysLUT);
}

void GLDZMFeature::calculate (LR& r, const Fsettings& s)
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

	//==== Prepare the GLDZM-matrix kit: matrix itself and its dimensions
	std::vector<PixIntens> greyLevelsLUT;
	SimpleMatrix<unsigned int> GLDZM;
	int Ng,	// number of grey levels
		Nd;	// maximum number of non-zero dependencies
	prepare_GLDZM_matrix_kit (GLDZM, Ng, Nd, greyLevelsLUT, r, STNGS_NGREYS(s), STNGS_IBSI(s));

	//==== Calculate vectors of totals by intensity (Mx) and by distance (Md)
	std::vector<double> Mx, Md;
	calc_row_and_column_sum_vectors (Mx, Md, GLDZM, Ng, Nd, greyLevelsLUT);

	//==== Calculate features, set variables f_GLE, f_GML, f_GLV, etc
	calc_features (Mx, Md, GLDZM, greyLevelsLUT, r.aux_area);
}

template <class Imgmatrx> int GLDZMFeature::dist2border (Imgmatrx& I, const int x, const int y)
{
	// scan left
	int dist2l = 0;
	for (int x0 = x - 1; x0 >= 0; x0--)
		if (I.yx(y, x0) == 0 || x0 == 0)	// we're at the ROI border or left margin (x0==0)
		{
			dist2l = x - x0;
			break;
		}
	// scan right
	int dist2r = 0, w = I.get_width();
	for (int x0 = x + 1; x0 < w; x0++)
		if (I.yx(y, x0) == 0 || x0 == w - 1)	// we're at the ROI border or right margin (x0==w-1)
		{
			dist2r = x0 - x;
			break;
		}
	// scan up
	int dist2t = 0;
	for (int y0 = y - 1; y0 >= 0; y0--)
		if (I.yx(y0, x) == 0 || y0 == 0)	// we're at the ROI border or top margin (y0==0)
		{
			dist2t = y - y0;
			break;
		}
	// scan down
	int dist2b = 0, h = I.get_height();
	for (int y0 = y + 1; y0 < h; y0++)
		if (I.yx(y0, x) == 0 || y0 == h - 1)	// we're at the ROI border or bottom margin (y0==h-1)
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

template <class Imgmatrx> void GLDZMFeature::calc_row_and_column_sum_vectors (std::vector<double> & Mx, std::vector<double>& Md, Imgmatrx& P, const int Ng, const int Nd, const std::vector<PixIntens>& greysLUT)
{
	// Sum distances of each grey levels
	Mx.resize (Ng);
	for (int g = 0; g < Ng; g++)
	{
		double sumD = 0;
		for (int d = 0; d < Nd; d++)
			sumD += P.yx (g, d);
		Mx[g] = sumD;
	}

	// Sum grey levels of each distance
	Md.resize (Nd);
	for (int d = 0; d < Nd; d++)
	{
		double sumG = 0;
		for (int g = 0; g < Ng; g++)
		{	
			// skip zero intensities
			auto inten = greysLUT[g];
			if (inten == 0)
				continue;

			sumG += P.yx (g,d);
		}
		Md[d] = sumG;
	}
}

template <class Imgmatrx> void GLDZMFeature::calc_features (const std::vector<double>& Mx, const std::vector<double>& Md, Imgmatrx& P, const std::vector<PixIntens>& greysLUT, unsigned int roi_area)
{
	int Ng = Mx.size(),
		Nd = Md.size();

	// Ns is the number of realised zones
	double Ns = 0;
	for (int g = 0; g < Ng; g++)
		for (int d = 0; d < Nd; d++)
			if (greysLUT[g])	// skip zero grey level zones
				Ns += P.yx(g,d);

	// Nv is the number of potential zones
	double Nv = roi_area;

	for (int d_ = 0; d_ < Nd; d_++)
	{
		double d = (double) (d_+ 1);
		double m = Md [d_];
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

		double g_ = (double) greysLUT[g];
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

			double g_ = (double) greysLUT[g],
				d_ = double (d + 1);
			double p = P.yx (g,d);
			f_SDLGLE += p / g_ / g_ / d_ / d_;	// Small Distance Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{ m_{x,d}}{x^2 d^2}
			f_SDHGLE += g_ * g_ * p / d_ / d_;	// Small Distance High Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{x^2  m_{x,d}}{d^2}
			f_LDLGLE += d_ * d_ * p / g_ / g_;	// Large Distance Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{d^2 m_{x,d}}{x^2}
			f_LDHGLE += g_ * g_ * d_ * d_ * p;	// Large Distance High Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \x^2 d^2 m_{x,d}
			f_GLM += g_ * p;					// Grey Level Mean = \mu_x = \sum_x \sum_d x p_{x,d}
			f_ZDM += d_ * p;					// Zone Distance Mean = \mu_d = \sum_x \sum_d d p_{x,d} 
			f_ZDE += p/Ns * log2(p/Ns + EPS);	// Zone Distance Entropy = - \sum_x \sum_d p_{x,d} \textup{log}_2 ( p_{x,d} )
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
			double d_ = (double) (d + 1);
			dif = d_ - f_ZDM;
			f_ZDV += dif * dif * p;
		}
}

void GLDZMFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::GLDZM_SDE][0] = f_SDE;
	fvals[(int)Feature2D::GLDZM_LDE][0] = f_LDE;
	fvals[(int)Feature2D::GLDZM_LGLZE][0] = f_LGLZE;
	fvals[(int)Feature2D::GLDZM_HGLZE][0] = f_HGLZE;
	fvals[(int)Feature2D::GLDZM_SDLGLE][0] = f_SDLGLE;
	fvals[(int)Feature2D::GLDZM_SDHGLE][0] = f_SDHGLE;
	fvals[(int)Feature2D::GLDZM_LDLGLE][0] = f_LDLGLE;
	fvals[(int)Feature2D::GLDZM_LDHGLE][0] = f_LDHGLE;
	fvals[(int)Feature2D::GLDZM_GLNU][0] = f_GLNU;
	fvals[(int)Feature2D::GLDZM_GLNUN][0] = f_GLNUN;
	fvals[(int)Feature2D::GLDZM_ZDNU][0] = f_ZDNU;
	fvals[(int)Feature2D::GLDZM_ZDNUN][0] = f_ZDNUN;
	fvals[(int)Feature2D::GLDZM_ZP][0] = f_ZP;
	fvals[(int)Feature2D::GLDZM_GLM][0] = f_GLM;
	fvals[(int)Feature2D::GLDZM_GLV][0] = f_GLV;
	fvals[(int)Feature2D::GLDZM_ZDM][0] = f_ZDM;
	fvals[(int)Feature2D::GLDZM_ZDV][0] = f_ZDV;
	fvals[(int)Feature2D::GLDZM_ZDE][0] = f_ZDE;
}

void GLDZMFeature::extract (LR& r, const Fsettings& s)
{
	GLDZMFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void GLDZMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	// Iterate ROIs of this batch
	for (auto i = start; i < end; i++)
	{
		// Get ahold of ROI's cached data
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		extract (r, s);
	}
}

void GLDZMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void GLDZMFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	clear_buffers();

	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;

	//==== Compose the distance matrix

	// -- Zones (intensity clusters)
	OutOfRamPixelCloud Z_int, Z_dist;
	Z_int.init (r.label, "GLDZMFeature-osized_calculate-Z_int");
	Z_dist.init (r.label, "GLDZMFeature-osized_calculate-Z_dist");

	// -- Unique intensities 
	std::unordered_set<PixIntens> U;

	// -- Create an image matrix for this ROI for
	//		(1) making a binned (coarser) pixel intensity image and 
	//		(2) zone finding alsorithm's ability to leave marks in recognized zones
	WriteImageMatrix_nontriv D("GLDZMFeature-osized_calculate-D", r.label);
	D.allocate_from_cloud(r.raw_pixels_NT, r.aabb, false);

	// -- Squeeze pixels' intensity range for getting more prominent zones 
	PixIntens piRange = r.aux_max - 0;	// reflecting the fact that the original image's pixel intensity range is [0-r.aux_max] where 0 represents off-ROI pixels
	unsigned int nGrays = STNGS_NGREYS(s);	// former theEnvironment.get_coarse_gray_depth()
	for (size_t i = 0; i < D.size(); i++)
	{
		// raw intensity
		unsigned int Ir = D[i];
		// ignore blank pixels
		if (Ir == 0)
			continue;
		// binned intensity
		unsigned int Ib = Nyxus::to_grayscale (Ir, 0, piRange, nGrays, Environment::ibsi_compliance);
		D.set_at (i, Ib);
		// update the set of unique intensities
		U.insert(Ib);
	}

	// -- Find zones
	constexpr int huge = INT_MAX;	// Value greater than any pixel's distance to ROI border
	const int VISITED = -1;
	// Helpful temps
	auto height = D.get_height(),
		width = D.get_width();
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			// Find a non-blank pixel
			auto inten = D.yx(row, col);
			if (inten == 0 || int(inten) == VISITED)
				continue;

			// Found a gray pixel. Find same-intensity neighbourhood of it.
			std::vector<std::tuple<int, int>> history;

			int x = col,
				y = row;
			int zoneSize = 1;	// '1' because we already have a pixel. Hopefully it's a part of a zone
			D.set_at(y, x, VISITED);

			// Keep track of this pixel, hopefully 1st pixel of the cluster
			history.push_back({ x,y });
			int zoneMetric = dist2border <WriteImageMatrix_nontriv>(D, x, y);

			// Scan the neighborhood of pixel (x,y)
			for (;;)
			{
				// East
				int _x = x + 1,
					_y = y;
				if (D.safe(_y, _x) && D.yx(_y, _x) != VISITED && D.yx(_y, _x) == inten)
				{
					D.set_at(y, x + 1, VISITED);

					// Update zone's metric
					int dist2roi = dist2border <WriteImageMatrix_nontriv>(D, _x, _y);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Remember this pixel
					history.push_back({ _x,_y });
					// Advance
					x = x + 1;
					// Proceed
					continue;
				}

				// South
				_x = x;
				_y = y + 1;
				if (D.safe(_y, _x) && D.yx(_y, _x) != VISITED && D.yx(_y, _x) == inten)
				{
					D.set_at(_y, _x, VISITED);

					// Update zone's metric
					int dist2roi = dist2border <WriteImageMatrix_nontriv>(D, _x, _y);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					history.push_back({ _x,_y });
					y = y + 1;
					continue;
				}

				// We are done exploring this cluster
				break;
			}

			// Done scanning a cluster. Perform 3 actions:
			// --1
			U.insert(inten);

			// --2 Create a zone
			Z_int.add_pixel(Pixel2(-1, -1, (StatsInt)inten));	// using just intensity field of triplet Pixel2
			Z_dist.add_pixel(Pixel2(-1, -1, (StatsInt)zoneMetric));
		}

	//==== Fill the zonal metric matrix

	// -- number of discrete intensity values in the image
	int Ng = (int)U.size();

	// -- max zone distance to ROI or image border
	int Nd = 0;
	for (auto p : Z_dist)
		Nd = std::max (Nd, (int)p.inten);

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> greysLUT (U.begin(), U.end());
	std::sort (greysLUT.begin(), greysLUT.end());	// Optional

	// -- Zone intensity -to- zone distance matrix
	WriteImageMatrix_nontriv P ("GLDZMFeature-osized_calculate-P", r.label);
	P.allocate (Nd, Ng, 0);	// Ng rows, Nd columns

	// -- fill it
	for (int i=0; i<Z_int.size(); i++)
	{
		auto inten = Z_int[i].inten;
		auto dist = Z_dist[i].inten;
		// row. Gray tones are sparse so we need to find indices of tones in 'Z' and use them as rows of P-matrix
		auto iter = std::find (greysLUT.begin(), greysLUT.end(), inten);
		int row = (int) (iter - greysLUT.begin());
		// col (a distance). Distances are dense \in [1,Nd]
		int col = dist - 1;	// 0-based => -1
		auto k = P.yx (row, col);
		P.set_at (row, col, k+1);
	}

	//==== Calculate vectors of totals by intensity (Mx) and by distance (Md)
	std::vector<double> Mx, Md;
	calc_row_and_column_sum_vectors (Mx, Md, P, Ng, Nd, greysLUT);

	//==== Calculate features, set variables f_GLE, f_GML, f_GLV, etc
	calc_features (Mx, Md, P, greysLUT, r.aux_area);
}

