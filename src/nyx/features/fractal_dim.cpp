#include <algorithm>
#include "fractal_dim.h"
#include "image_matrix.h"
#include "../helpers/helpers.h"

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

namespace
{
	// Box counting is registration sensitive: the ROI is padded tight to a power of two (ceil_pow2)
	// and aligned to the ROI origin, since centring it on a larger canvas misaligns it with the coarse
	// boxes and biases the dimension low (a filled square read 1.75 instead of 2.0). Shifting grids
	// (the minimum count over grid origins, as FracLac does) remove the residual bias, which matters
	// only where few box sizes fit (small ROIs) and those are cheap to shift. So a padded side of 64
	// or more, which fits at least 6 box sizes (2, 4, ..., 64) and so a stable single-grid fit, is
	// counted on one grid, and a padded side of 32 or less over several grid origins.
	const int MAX_SHIFTED_GRID_SIDE = 32;

	int padded_side (const AABB& aabb)
	{
		return Nyxus::ceil_pow2 ((int) std::max (aabb.get_width(), aabb.get_height()));
	}

	// (box size, boxes holding a ROI pixel) per box size, on the one grid aligned with the square padded
	// image 'pim' of side 'side'. tile_contains_signal() stops at a box's first ROI pixel, so a filled
	// ROI costs about one read per box rather than per pixel.
	template <class PaddedImage>
	void count_single_grid (PaddedImage& pim, int side, std::vector<std::pair<double, double>>& coverage)
	{
		for (int s = side; s > 1; s /= 2)
		{
			int cnt = 0, nt = side / s;
			for (int tr = 0; tr < nt; tr++)
				for (int tc = 0; tc < nt; tc++)
					if (pim.tile_contains_signal (tr, tc, s))
						cnt++;
			coverage.push_back ({ double(s), double(cnt) });
		}
	}

	// (box size, boxes holding a ROI pixel) per box size, the minimum over the grid origins
	// {0, s/2} along each axis. A box index is coord >> log2(s), the box sizes being powers of two.
	template <class Cloud>
	void count_shifted_grids (const Cloud& px, const AABB& aabb, int paddedSide, std::vector<std::pair<double, double>>& coverage)
	{
		auto xmin = aabb.get_xmin();
		auto ymin = aabb.get_ymin();
		int n_off = 2;
		int shift = 0;
		for (int t = paddedSide; t > 1; t >>= 1)
			shift++;
		std::vector<char> occ;
		for (int s = paddedSide; s > 1; s >>= 1, --shift)
		{
			int span = (paddedSide >> shift) + 2;	// boxes per axis (+slack for the origin shift)
			occ.assign((size_t)span * span, 0);
			int best = -1;
			for (int oyi = 0; oyi < n_off; oyi++)
				for (int oxi = 0; oxi < n_off; oxi++)
				{
					int ox = (oxi * s) / n_off, oy = (oyi * s) / n_off;
					int cnt = 0;
					for (const Pixel2& p : px)
					{
						int col = (int)(p.x - xmin + ox) >> shift;
						int row = (int)(p.y - ymin + oy) >> shift;
						size_t idx = (size_t)row * span + col;
						if (!occ[idx]) { occ[idx] = 1; cnt++; }
					}
					if (best < 0 || cnt < best)
						best = cnt;
					if (oyi + 1 < n_off || oxi + 1 < n_off)
						std::fill(occ.begin(), occ.end(), (char)0);	// reset for next origin
				}
			coverage.push_back({ double(s), double(best) });
		}
	}
}

void FractalDimensionFeature::calculate_boxcount_fdim (LR & r)
{
	const std::vector<Pixel2>& px = r.raw_pixels;
	if (px.size() < 2)
	{
		box_count_fd = 0.;
		return;
	}

	int side = padded_side (r.aabb);
	std::vector<std::pair<double, double>> coverage;
	if (side > MAX_SHIFTED_GRID_SIDE)
	{
		Power2PaddedImageMatrix pim (px, r.aabb, 1, 1.0);
		count_single_grid (pim, (int) pim.width, coverage);
	}
	else
		count_shifted_grids (px, r.aabb, side, coverage);

	box_count_fd = boxcount_dimension (coverage);
}

void FractalDimensionFeature::calculate_boxcount_fdim_oversized (LR & r)
{
	// calculate_boxcount_fdim() over the disk-backed pixel cloud and padded image
	const OutOfRamPixelCloud& px = r.raw_pixels_NT;
	if (px.size() < 2)
	{
		box_count_fd = 0.;
		return;
	}

	int side = padded_side (r.aabb);
	std::vector<std::pair<double, double>> coverage;
	if (side > MAX_SHIFTED_GRID_SIDE)
	{
		Power2PaddedImageMatrix_NT pim ("pim", r.label, px, r.aabb, 1, 1.0);
		count_single_grid (pim, (int) pim.get_width(), coverage);
	}
	else
		count_shifted_grids (px, r.aabb, side, coverage);

	box_count_fd = boxcount_dimension (coverage);
}

double FractalDimensionFeature::boxcount_dimension (const std::vector<std::pair<double, double>>& coverage)
{
	// Box-counting dimension = -slope of log(count) vs log(box size), subtracted from 0 rather than
	// negated so that a zero slope reads +0
	return 0.0 - loglog_slope (coverage);
}

void FractalDimensionFeature::calculate_perimeter_fdim (LR& r)
{
	// The divider walks each closed border of the ROI -- the outer border of every fragment and the
	// border of every hole -- on its own, and the borders' lengths and steps are summed per stride.
	// No step joins one border to another.
	size_t longest = 0;
	for (const std::vector<Pixel2>& K : r.multicontour_)
		longest = std::max (longest, K.size());
	if (longest < 3)
	{
		perim_fd = 0.;
		return;
	}

	// Richardson "structured walk" (divider) method on the CLOSED borders.
	// For each stride s, walk every border in steps of s vertices, summing Euclidean chord
	// lengths and closing each loop back to its start. A border no longer than the stride cannot
	// have a ruler that long laid along it and is measured whole instead: it adds its closed length,
	// the length stride 1 measures it at, and no ruler, so the mean ruler is that of the borders the
	// stride walks. Record x = mean ruler (chord) length and y = perimeter estimate. Since
	// log(perimeter) ~ (1 - D) log(ruler), D = 1 - slope.
	// Validated against the analytic Koch snowflake (D = log4/log3 = 1.2619) and a circle (D = 1).
	std::vector<double> closedLen;
	for (const std::vector<Pixel2>& K : r.multicontour_)
	{
		double len = 0.;
		for (size_t i = 0, n = K.size(); n > 1 && i < n; i++)
			len += std::sqrt(K[i].sqdist(K[(i + 1) % n]));
		closedLen.push_back(len);
	}

	std::vector<std::pair<double, double>> coverage;
	for (size_t s = longest / 4; s > 0; s /= 2)
	{
		double walked = 0., whole = 0.;
		size_t nsteps = 0;
		for (size_t k = 0; k < r.multicontour_.size(); k++)
		{
			const std::vector<Pixel2>& K = r.multicontour_[k];
			const size_t conLen = K.size();
			if (conLen <= s)
			{
				whole += closedLen[k];
				continue;
			}
			size_t i = 0;
			for (; i + s < conLen; i += s)
			{
				walked += std::sqrt(K[i].sqdist(K[i + s]));
				nsteps++;
			}
			// close the loop from the last visited vertex back to the start
			walked += std::sqrt(K[i].sqdist(K[0]));
			nsteps++;
		}

		double ruler = walked / double(nsteps);	// mean chord (ruler) length of the walked borders
		coverage.push_back({ ruler, walked + whole });
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

