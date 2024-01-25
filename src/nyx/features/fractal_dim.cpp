#include "fractal_dim.h"
#include "image_matrix.h"

using namespace Nyxus;

FractalDimensionFeature::FractalDimensionFeature() : FeatureMethod("FractalDimensionFeature")
{
	provide_features({ Feature2D::FRACT_DIM_BOXCOUNT, Feature2D::FRACT_DIM_PERIMETER });
	add_dependencies({ Feature2D::PERIMETER });	// FRACT_DIM_PERIMETER requires perimeter's pixels
}

void FractalDimensionFeature::calculate(LR& r)
{
	if (theFeatureSet.isEnabled(Feature2D::FRACT_DIM_BOXCOUNT))
		calculate_boxcount_fdim(r);

	if (theFeatureSet.isEnabled(Feature2D::FRACT_DIM_PERIMETER))
		calculate_perimeter_fdim(r);
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

	auto conLen = r.contour.size();
	for (size_t s = conLen / 4; s > 0; s /= 2)
	{
		// calculate the s-approximated perimeter
		double p = 0;
		for (size_t i = s; i < conLen; i += s)
		{
			auto& px1 = r.contour[i - s],
				px2 = r.contour[i];
			double dist = std::sqrt(px1.sqdist(px2));
			p += dist;
		}
		// --calculate the last segment separately
		auto tail = conLen % s;
		if (tail)
		{
			auto& px1 = r.contour[conLen - tail],
				px2 = r.contour[0];
			double dist = px1.sqdist(px2);
			p += dist;
		}

		// save this approximation
		coverage.push_back({ s, p });
	}
	perim_fd = calc_lyapunov_slope(coverage);
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

	// Estimate the slope of the series Lambda[k]
	// (given y = a + bx, the slope b = \frac {n \sum{xy} - \sum{x} \sum{y}} {n \sum{x^2} - \sum{x}^2})
	double sum_x = 0,
		sum_y = 0,
		sum_xy = 0,
		sum_x2 = 0;
	int n = Dn.size();
	for (auto i = 0; i < n; i++)
	{
		double x = double(i),
			y = Lambda[i];
		sum_x += x;
		sum_y += y;
		sum_xy += x * y;
		sum_x2 += x * x;
	}
	double slope = (sum_xy * double(n) - sum_x * sum_y) / (sum_x2 * double(n) - sum_x * sum_x);
	return slope;
}

void FractalDimensionFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void FractalDimensionFeature::osized_calculate(LR& r, ImageLoader& imloader)
{
	if (theFeatureSet.isEnabled(Feature2D::FRACT_DIM_BOXCOUNT))
		calculate_boxcount_fdim_oversized (r);

	if (theFeatureSet.isEnabled(Feature2D::FRACT_DIM_PERIMETER))
		calculate_perimeter_fdim(r);
}

void FractalDimensionFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::FRACT_DIM_BOXCOUNT][0] = box_count_fd;
	fvals[(int)Feature2D::FRACT_DIM_PERIMETER][0] = perim_fd;
}

void FractalDimensionFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		// Calculate feature
		FractalDimensionFeature fd;
		fd.calculate(r);
		fd.save_value(r.fvals);
	}
}

