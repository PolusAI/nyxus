#include <limits.h>
#include "gldzm.h"
#include "../environment.h"

GLDZMFeature::GLDZMFeature() : FeatureMethod("GLDZMFeature")
{
	provide_features({
		GLDZM_SDE,
		GLDZM_LDE,
		GLDZM_LGLE,
		GLDZM_HGLE,
		GLDZM_SDLGLE,
		GLDZM_SDHGLE,
		GLDZM_LDLGLE,
		GLDZM_LDHGLE,
		GLDZM_GLNU,
		GLDZM_GLNUN,
		GLDZM_ZDNU,
		GLDZM_ZDNUN,
		GLDZM_ZP,
		GLDZM_GLM,
		GLDZM_GLV,
		GLDZM_ZDM,
		GLDZM_ZDV,
		GLDZM_ZDE });
}

void GLDZMFeature::clear_buffers()
{
	f_SDE =
		f_LDE =
		f_LGLE =
		f_HGLE =
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

void GLDZMFeature::calculate (LR& r)
{
	clear_buffers();

	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;

	//==== Compose the distance matrix

	// -- Zones (intensity clusters)
	using ACluster = std::pair<PixIntens, int>;	// <zone intensity, zone distance>
	std::vector<ACluster> Z;

	// -- Unique intensities 
	std::unordered_set<PixIntens> U;

	// -- We need a copy of ROI's image matrix for 
	//		(1) making a binned (coarser) pixel intensity image and 
	//		(2) zone finding alsorithm's ability to leave marks in recognized zones
	auto M = r.aux_image_matrix;
	pixData& D = M.WriteablePixels();

	// -- Squeeze the copy's intensity range for getting more prominent zones
	PixIntens piRange = r.aux_max - 0; // Not 'r.aux_max - r.aux_min' because unlike in 'LR::raw_pixels' the min intensity in an image matrix ==0
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
	for (size_t i = 0; i < D.size(); i++)
	{
		// raw intensity
		unsigned int Ir = D[i];
		// ignore blank pixels
		if (Ir == 0)
			continue;
		// binned intensity
		unsigned int Ib = Nyxus::to_grayscale (Ir, 0, piRange, nGrays, Environment::ibsi_compliance);
		D[i] = Ib;
		// update the set of unique intensities
		U.insert(Ib);
	}

	//==== Find zones
	constexpr int huge = INT_MAX;	// Value greater than any pixel's distance to ROI border
	const int VISITED = -1;
	for (int row = 0; row < M.height; row++)
		for (int col = 0; col < M.width; col++)
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
			D.yx(y, x) = VISITED;

			// Keep track of this pixel, hopefully 1st pixel of the cluster
			history.push_back ({x,y});
			int zoneMetric = dist2border <pixData> (D, x, y); //dist2closestRoiBorder (D, x, y);

			// Scan the neighborhood of pixel (x,y)
			for (;;)
			{
				// East
				int _x = x + 1,
					_y = y;
				if (D.safe(_y,_x) && D.yx(_y,_x) != VISITED && D.yx(_y,_x) == inten)
				{
					D.yx(y, x + 1) = VISITED;

					// Update zone's metric
					int dist2roi = dist2border <pixData> (D, _x, _y); //dist2closestRoiBorder (D, _x, _y);
					zoneMetric = std::min (zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					// Remember this pixel
					history.push_back ({_x,_y});
					// Advance
					x = x + 1;
					// Proceed
					continue;
				}

				// South
				_x = x;
				_y = y + 1;
				if (D.safe(_y,_x) && D.yx(_y,_x) != VISITED && D.yx(_y,_x) == inten)
				{
					D.yx(_y,_x) = VISITED;

					// Update zone's metric
					int dist2roi = dist2border <pixData> (D, _x, _y); //dist2closestRoiBorder(D, _x, _y);
					zoneMetric = std::min(zoneMetric, dist2roi);

					// Update zone size
					zoneSize++;

					history.push_back ({_x,_y});
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
			ACluster clu = { inten, zoneMetric};
			Z.push_back(clu);
		}

	//==== Fill the zonal metric matrix

	// -- number of discrete intensity values in the image
	int Ng = (int)U.size();

	// -- max zone distance to ROI or image border
	int Nd = 0;
	for (auto& z : Z)
		Nd = std::max (Nd, z.second);

	// -- Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I (U.begin(), U.end());
	std::sort (I.begin(), I.end());	// Optional

	// -- Zone intensity -to- zone distance matrix
	SimpleMatrix<int> P;
	P.allocate (Nd, Ng);	// Ng rows, Nd columns
	P.fill(0);

	// -- fill it
	int i = 0;
	for (auto& z : Z)
	{
		// row. Gray tones are sparse so we need to find indices of tones in 'Z' and use them as rows of P-matrix
		auto iter = std::find (I.begin(), I.end(), z.first);	
		int row = (int)(iter - I.begin());
		// col (a distance). Distances are dense \in [1,Nd]
		int col = z.second - 1;	// 0-based => -1
		auto& k = P.yx (row, col);
		k++;
	}

	//==== Calculate vectors of totals by intensity (Mx) and by distance (Md)
	std::vector<double> Mx, Md;
	calc_row_and_column_sum_vectors (Mx, Md, P, Ng, Nd);

	//==== Calculate features, set variables f_GLE, f_GML, f_GLV, etc
	calc_features (Mx, Md, P, r.aux_area);
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
	// result
	int retval = std::min(std::min(std::min(dist2l, dist2r), dist2t), dist2b);
	if (retval == 0)
		retval = 1;	// Requirement of GLDZM: pixel on the border is within distance 1 from ROI border
	return retval;
}

template <class Imgmatrx> void GLDZMFeature::calc_row_and_column_sum_vectors (std::vector<double> & Mx, std::vector<double>& Md, Imgmatrx& P, const int Ng, const int Nd)
{
	// Sum distances of each grey levels
	Mx.resize (Ng);
	for (int gray_i = 0; gray_i < Ng; gray_i++)
	{
		double sumD = 0;
		for (int d = 1; d <= Nd; d++)
			sumD += P.yx (gray_i, d-1);
		Mx[gray_i] = sumD;
	}

	// Sum grey levels of each distance
	Md.resize (Nd);
	for (int d = 1; d <= Nd; d++)
	{
		double sumG = 0;
		for (int gray_i = 0; gray_i < Ng; gray_i++)
			sumG += P.yx (gray_i, d-1);
		Md[d-1] = sumG;
	}
}

template <class Imgmatrx> void GLDZMFeature::calc_features (const std::vector<double>& Mx, const std::vector<double>& Md, Imgmatrx& P, unsigned int roi_area)
{
	int Nd = Md.size();
	for (int d_ = 1; d_ <= Nd; d_++)
	{
		double d = (double) d_;
		double m = Md [d_-1];
		f_SDE += m / d / d;			// Small Distance Emphasis = \frac{1}{N_s} \sum_d \frac{m_d}{d^2}
		f_LDE += d * d * m;			// Large Distance Emphasis = \frac{1}{N_s} \sum_d d^2 m_d 
		f_ZDNU += m * m;			// Zone Distance Non-Uniformity = \frac{1}{N_s} \sum_d m_d^2
									// Zone Distance Non-Uniformity Normalized = \frac{1}{N_s^2} \sum_d m_d^2
	}

	double Ns = 0;
	for (auto levelSumOfDists : Mx)
		Ns += levelSumOfDists;

	f_SDE /= (double)Ns;
	f_LDE /= (double)Ns;
	f_ZDNU /= (double)Ns;
	f_ZDNUN = f_ZDNU / (double)Ns;

	int Ng = Mx.size();
	for (int g = 0; g < Ng; g++)
	{
		double tmp = (double)g + 1;
		double x = Mx[g];
		f_LGLE += x / (tmp * tmp);	// Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \frac{m_x}{x^2}
		f_HGLE += (tmp * tmp) * x;	// High Grey Level Emphasis = \frac{1}{N_s} \sum_x x^2 m_x
		f_GLNU += x * x;			// Grey Level Non-Uniformity = \frac{1}{N_s} \sum_x m_x^2
	}
	f_LGLE /= (double)Ns;
	f_HGLE /= (double)Ns;
	f_GLNU /= (double)Ns;
	f_GLNUN = f_GLNU / (double)Ns;	// Grey Level Non-Uniformity Normalized = \frac{1}{N_s^2} \sum_x m_x^2

	for (int g = 0; g < Ng; g++)
		for (int d = 1; d <= Nd; d++)
		{
			double g_ = (double)g + 1, d_ = d;
			double p = P.yx(g, d - 1);
			f_SDLGLE += p / g_ / g_ / d_ / d_;	// Small Distance Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{ m_{x,d}}{x^2 d^2}
			f_SDHGLE += g_ * g_ * p / d_ / d_;	// Small Distance High Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{x^2  m_{x,d}}{d^2}
			f_LDLGLE += d_ * d_ * p / g_ / g_;	// Large Distance Low Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \frac{d^2 m_{x,d}}{x^2}
			f_LDHGLE += g_ * g_ * d_ * d_ * p;		// Large Distance High Grey Level Emphasis = \frac{1}{N_s} \sum_x \sum_d \x^2 d^2 m_{x,d}
			f_GLM += g_ * p;					// Grey Level Mean = \mu_x = \sum_x \sum_d x p_{x,d}
			f_ZDM += d_ * p;					// Zone Distance Mean = \mu_d = \sum_x \sum_d d p_{x,d} 
			f_ZDE += p * log2(p + EPS);			// Zone Distance Entropy = - \sum_x \sum_d p_{x,d} \textup{log}_2 ( p_{x,d} )
		}
	f_SDLGLE /= (double)Ns;
	f_SDHGLE /= (double)Ns;
	f_LDLGLE /= (double)Ns;
	f_LDHGLE /= (double)Ns;
	f_ZP = (double)Ns / (double)roi_area;		// Zone Percentage = \frac{N_s}{N_v}
	f_GLE = f_ZDE;

	for (int g = 0; g < Ng; g++)
		for (int d = 1; d <= Nd; d++)
		{
			// Grey Level Variance = \sum_x \sum_d \left(x - \mu_x \right)^2 p_{x,d}
			double p = P.yx(g, d - 1) / (double)Ns,
				x = (double)g,
				dif = x - f_GLM;
			f_GLV += dif * dif * p;
			// Zone Distance Variance} = \sum_x \sum_d \left(d - \mu_d \right)^2 p_{x,d} 
			double d_ = (double)d;
			dif = d - f_ZDM;
			f_ZDV += dif * dif * p;
		}
}

void GLDZMFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[GLDZM_SDE][0] = f_SDE;
	fvals[GLDZM_LDE][0] = f_LDE;
	fvals[GLDZM_LGLE][0] = f_LGLE;
	fvals[GLDZM_HGLE][0] = f_HGLE;
	fvals[GLDZM_SDLGLE][0] = f_SDLGLE;
	fvals[GLDZM_SDHGLE][0] = f_SDHGLE;
	fvals[GLDZM_LDLGLE][0] = f_LDLGLE;
	fvals[GLDZM_LDHGLE][0] = f_LDHGLE;
	fvals[GLDZM_GLNU][0] = f_GLNU;
	fvals[GLDZM_GLNUN][0] = f_GLNUN;
	fvals[GLDZM_ZDNU][0] = f_ZDNU;
	fvals[GLDZM_ZDNUN][0] = f_ZDNUN;
	fvals[GLDZM_ZP][0] = f_ZP;
	fvals[GLDZM_GLM][0] = f_GLM;
	fvals[GLDZM_GLV][0] = f_GLV;
	fvals[GLDZM_ZDM][0] = f_ZDM;
	fvals[GLDZM_ZDV][0] = f_ZDV;
	fvals[GLDZM_ZDE][0] = f_ZDE;
}

void GLDZMFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	// Iterate ROIs of this batch
	for (auto i = start; i < end; i++)
	{
		// Get ahold of ROI's cached data
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Calculate feature of this ROI
		GLDZMFeature f;
		f.calculate (r);
		f.save_value (r.fvals);
	}
}

void GLDZMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void GLDZMFeature::osized_calculate (LR& r, ImageLoader&)
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
	WriteImageMatrix_nontriv D("GLDZMFeature-osized_calculate-M", r.label);
	D.allocate_from_cloud(r.raw_pixels_NT, r.aabb, false);

	// -- Squeeze pixels' intensity range for getting more prominent zones 
	PixIntens piRange = r.aux_max - 0;	// reflecting the fact that the original image's pixel intensity range is [0-r.aux_max] where 0 represents off-ROI pixels
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
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
	std::vector<PixIntens> I (U.begin(), U.end());
	std::sort (I.begin(), I.end());	// Optional

	// -- Zone intensity -to- zone distance matrix
	WriteImageMatrix_nontriv P ("GLDZMFeature-osized_calculate-P", r.label);
	P.allocate (Nd, Ng, 0);	// Ng rows, Nd columns

	// -- fill it
	for (int i=0; i<Z_int.size(); i++)
	{
		auto inten = Z_int[i].inten;
		auto dist = Z_dist[i].inten;
		// row. Gray tones are sparse so we need to find indices of tones in 'Z' and use them as rows of P-matrix
		auto iter = std::find(I.begin(), I.end(), inten);
		int row = (int)(iter - I.begin());
		// col (a distance). Distances are dense \in [1,Nd]
		int col = dist - 1;	// 0-based => -1
		auto k = P.yx (row, col);
		P.set_at (row, col, k+1);
	}

	//==== Calculate vectors of totals by intensity (Mx) and by distance (Md)
	std::vector<double> Mx, Md;
	calc_row_and_column_sum_vectors (Mx, Md, P, Ng, Nd);

	//==== Calculate features, set variables f_GLE, f_GML, f_GLV, etc
	calc_features (Mx, Md, P, r.aux_area);
}

