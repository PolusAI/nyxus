#include "fractal_dim.h"
#include "image_matrix.h"

using namespace Nyxus;

FractalDimensionFeature::FractalDimensionFeature() : FeatureMethod("FractalDimensionFeature")
{
	provide_features (FractalDimensionFeature::featureset);
	add_dependencies({ Feature2D::PERIMETER });	// FRACT_DIM_PERIMETER requires perimeter's pixels
}

void FractalDimensionFeature::calculate (LR& r, const Fsettings& s)
{
	calculate_boxcount_fdim (r);
	calculate_perimeter_fdim (r);
}

void FractalDimensionFeature::calculate_boxcount_fdim (LR & r)
{
	Power2PaddedImageMatrix pim(r.raw_pixels, r.aabb, 1, 1.0);	// image matrix of the mask

	// Debug
	// pim.print("Padded");

	// Square box coverage statistics
	std::vector<std::pair<int, int>> coverage;
	int s = pim.width;	// square tile size
	for (; s > 1; s /= 2)
	{
		int cnt = 0;
		size_t areaCov = 0;
		int n_tiles = pim.width / s;
		for (int r = 0; r < n_tiles; r++)
			for (int c = 0; c < n_tiles; c++)
			{
				// check the tile for coerage
				if (pim.tile_contains_signal(r, c, s))
					cnt++;
			}

		coverage.push_back({ s, cnt });
	}

	// Debug
	// print_curve(boxCoverage, "boxCoverage N vs R");	

	box_count_fd = -calc_lyapunov_slope(coverage);
}

void FractalDimensionFeature::calculate_boxcount_fdim_oversized (LR & r)
{
	Power2PaddedImageMatrix_NT pim ("pim", r.label, r.raw_pixels_NT, r.aabb, 1, 1.0);	// image matrix of the mask

	// Square box coverage statistics
	std::vector<std::pair<int, int>> coverage;
	int s = pim.get_width();	// square tile size
	for (; s > 1; s /= 2)
	{
		int cnt = 0;
		size_t areaCov = 0;
		int n_tiles = pim.get_width() / s;
		for (int r = 0; r < n_tiles; r++)
			for (int c = 0; c < n_tiles; c++)
			{
				// check the tile for coerage
				if (pim.tile_contains_signal(r, c, s))
					cnt++;
			}

		coverage.push_back({ s, cnt });
	}

	box_count_fd = -calc_lyapunov_slope(coverage);
}

void FractalDimensionFeature::calculate_perimeter_fdim (LR& r)
{
	std::vector<std::pair<int, int>> coverage;

	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	auto conLen = K.size();
	for (size_t s = conLen / 4; s > 0; s /= 2)
	{
		// calculate the s-approximated perimeter
		double p = 0;
		for (size_t i = s; i < conLen; i += s)
		{
			auto& px1 = K [i-s],
				px2 = K [i];
			double dist = std::sqrt(px1.sqdist(px2));
			p += dist;
		}
		// --calculate the last segment separately
		auto tail = conLen % s;
		if (tail)
		{
			auto& px1 = K [conLen - tail],
				px2 = K [0];
			double dist = px1.sqdist(px2);
			p += dist;
		}

		// save this approximation
		coverage.push_back({ s, (int)p });
	}
	// Richardson divider method: log(perimeter) vs log(ruler) has slope (1 - D), so D = 1 - slope
	perim_fd = 1.0 - calc_lyapunov_slope(coverage);
}

double FractalDimensionFeature::calc_lyapunov_slope (const std::vector<std::pair<int, int>> & coverage)
{
	// Do we have any data?
	if (coverage.size() == 0)
		return 0.;

	// Skip tiny ROIs
	if (coverage.size() < 2)
		return 0.;

	// Post-process towards local gradients (in the form of Lyapunov exponents)
	std::vector <double> X, Y;
	for (int i = 0; i < coverage.size(); i++)
	{
		X.push_back (std::log(coverage[i].first));
		Y.push_back (std::log(coverage[i].second));
	}

	// Gradients
	std::vector<double> Dn, Dr;
	for (int i = 1; i < Y.size(); i++)
	{
		auto dn = Y[i] - Y[i - 1];
		Dn.push_back(dn);
		auto dr = X[i] - X[i - 1];
		Dr.push_back(dr);
	}

	// Lyapunov exponents
	std::vector<double> Lambda;
	for (int i = 0; i < Dn.size(); i++)
	{
		auto lambda = Dn[i] / Dr[i];
		Lambda.push_back(lambda);
	}

	// The box-counting / Richardson dimension is the (average) slope of log(count) vs log(scale),
	// i.e. the MEAN of the local slopes Lambda[]. The previous code returned the least-squares
	// slope of Lambda[] *against its index* (the rate-of-change of the slope), which is ~0 for a
	// clean power law -> dimension came out ~0 (FRACT_DIM_BOXCOUNT=-0.07). Return the mean slope.
	if (Lambda.empty())
		return 0.;
	double sum = 0.;
	for (double v : Lambda)
		sum += v;
	return sum / double(Lambda.size());
}

void FractalDimensionFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void FractalDimensionFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	calculate_boxcount_fdim_oversized (r);
	calculate_perimeter_fdim (r);
}

void FractalDimensionFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::FRACT_DIM_BOXCOUNT][0] = box_count_fd;
	fvals[(int)Feature2D::FRACT_DIM_PERIMETER][0] = perim_fd;
}

void FractalDimensionFeature::extract (LR& r, const Fsettings& s)
{
	FractalDimensionFeature fd;
	fd.calculate (r, s);
	fd.save_value (r.fvals);
}

void FractalDimensionFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		extract (r, s);
	}
}

