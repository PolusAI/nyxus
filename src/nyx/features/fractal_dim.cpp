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

	// Square box coverage statistics: number of boxes of side s that contain signal,
	// for s = width, width/2, ... 2. With the ROI grid-aligned (origin, tight power-of-2
	// canvas) a filled shape yields the exact power law N(s) ~ s^-D.
	std::vector<std::pair<double, double>> coverage;
	int s = pim.width;	// square tile size
	for (; s > 1; s /= 2)
	{
		int cnt = 0;
		int n_tiles = pim.width / s;
		for (int r = 0; r < n_tiles; r++)
			for (int c = 0; c < n_tiles; c++)
			{
				// check the tile for coverage
				if (pim.tile_contains_signal(r, c, s))
					cnt++;
			}

		coverage.push_back({ double(s), double(cnt) });
	}

	// Box-counting dimension = -slope of log(count) vs log(box size)
	box_count_fd = -loglog_slope(coverage);
}

void FractalDimensionFeature::calculate_boxcount_fdim_oversized (LR & r)
{
	Power2PaddedImageMatrix_NT pim ("pim", r.label, r.raw_pixels_NT, r.aabb, 1, 1.0);	// image matrix of the mask

	// Square box coverage statistics (oversized/streaming path), same method as the trivial case.
	std::vector<std::pair<double, double>> coverage;
	int s = pim.get_width();	// square tile size
	for (; s > 1; s /= 2)
	{
		int cnt = 0;
		int n_tiles = pim.get_width() / s;
		for (int r = 0; r < n_tiles; r++)
			for (int c = 0; c < n_tiles; c++)
			{
				// check the tile for coverage
				if (pim.tile_contains_signal(r, c, s))
					cnt++;
			}

		coverage.push_back({ double(s), double(cnt) });
	}

	box_count_fd = -loglog_slope(coverage);
}

void FractalDimensionFeature::calculate_perimeter_fdim (LR& r)
{
	std::vector<Pixel2> K;
	r.merge_multicontour(K);

	auto conLen = K.size();
	if (conLen < 3)
	{
		perim_fd = 0.;
		return;
	}

	// Richardson "structured walk" (divider) method on the CLOSED contour K.
	// For each stride s, walk the contour in steps of s vertices, summing Euclidean chord
	// lengths and closing the loop back to the start. Record x = mean ruler (chord) length
	// and y = perimeter estimate. Since log(perimeter) ~ (1 - D) log(ruler), D = 1 - slope.
	// Validated against the analytic Koch snowflake (D = log4/log3 = 1.2619) and a circle (D = 1).
	std::vector<std::pair<double, double>> coverage;
	for (size_t s = conLen / 4; s > 0; s /= 2)
	{
		double perim = 0.;
		size_t nsteps = 0, i = 0;
		for (; i + s < conLen; i += s)
		{
			perim += std::sqrt(K[i].sqdist(K[i + s]));
			nsteps++;
		}
		// close the loop from the last visited vertex back to the start
		perim += std::sqrt(K[i].sqdist(K[0]));
		nsteps++;

		double ruler = perim / double(nsteps);	// mean chord (ruler) length at this stride
		coverage.push_back({ ruler, perim });
	}

	// Richardson divider dimension: D = 1 - slope of log(perimeter) vs log(ruler)
	perim_fd = 1.0 - loglog_slope(coverage);
}

double FractalDimensionFeature::loglog_slope (const std::vector<std::pair<double, double>> & coverage)
{
	// Least-squares slope of log(y) vs log(x) over the (scale, measure) pairs.
	// Box counting passes (box size, count) -> D = -slope; the divider method passes
	// (ruler length, perimeter) -> D = 1 - slope. A least-squares fit over all scales is
	// the standard, robust estimator (more stable than a two-endpoint slope on noisy ROIs).
	double sx = 0, sy = 0, sxy = 0, sx2 = 0;
	size_t used = 0;
	for (const auto& pr : coverage)
	{
		if (pr.first <= 0. || pr.second <= 0.)
			continue;	// log undefined; skip degenerate points (count/perimeter 0)
		double x = std::log(pr.first), y = std::log(pr.second);
		sx += x; sy += y; sxy += x * y; sx2 += x * x;
		used++;
	}
	if (used < 2)
		return 0.;
	double denom = sx2 * double(used) - sx * sx;
	if (denom == 0.)
		return 0.;
	return (sxy * double(used) - sx * sy) / denom;
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

