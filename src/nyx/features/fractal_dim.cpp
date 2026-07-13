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

void FractalDimensionFeature::calculate_boxcount_fdim (LR & r)
{
	const std::vector<Pixel2>& px = r.raw_pixels;
	if (px.size() < 2)
	{
		box_count_fd = 0.;
		return;
	}

	// Box counting is registration sensitive: the old code padded to a power of two strictly
	// larger than the ROI and centered it, misaligning the ROI with the coarse boxes and biasing
	// the dimension low (a filled square read 1.75 instead of 2.0). We pad tight (ceil_pow2) and
	// align to the ROI origin. Shifting grids (min count over grid origins, as FracLac does)
	// remove the residual bias, but that only matters when few box sizes fit (small ROIs), and
	// those are cheap to shift. So auto-switch:
	//   - large ROI  -> single origin grid via the padded mask matrix (fast early-exit tile scan)
	//   - small ROI  -> shift the grid over a few origins and take the minimum box count
	int bigSide = (int)std::max(r.aabb.get_width(), r.aabb.get_height());
	int paddedSide = Nyxus::ceil_pow2(bigSide);

	std::vector<std::pair<double, double>> coverage;

	// 32: paddedSide > 32 (i.e. >= 64) fits at least 6 box sizes (2,4,...,64), enough for a stable
	// single-grid least-squares fit; at or below it too few scales remain and grid-registration
	// bias dominates, so we shift the grid over several origins instead.
	if (paddedSide > 32)
	{
		// Large ROI: single aligned grid. tile_contains_signal early-exits on the first non-zero
		// pixel, so this is O(#tiles) for filled ROIs - much cheaper than visiting every pixel.
		Power2PaddedImageMatrix pim(px, r.aabb, 1, 1.0);
		for (int s = pim.width; s > 1; s /= 2)
		{
			int cnt = 0, nt = pim.width / s;
			for (int tr = 0; tr < nt; tr++)
				for (int tc = 0; tc < nt; tc++)
					if (pim.tile_contains_signal(tr, tc, s))
						cnt++;
			coverage.push_back({ double(s), double(cnt) });
		}
	}
	else
	{
		// Small ROI: shifting grids. n_off grid origins per axis ({0, s/2} for 2 -> 4 positions);
		// box index = coord >> log2(s) since box sizes are powers of two.
		auto xmin = r.aabb.get_xmin();
		auto ymin = r.aabb.get_ymin();
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

	// Box-counting dimension = -slope of log(count) vs log(box size)
	box_count_fd = -loglog_slope(coverage);
}

void FractalDimensionFeature::calculate_boxcount_fdim_oversized (LR & r)
{
	Power2PaddedImageMatrix_NT pim ("pim", r.label, r.raw_pixels_NT, r.aabb, 1, 1.0);	// image matrix of the mask

	// Oversized/streaming ROIs are always large, so use a single aligned grid (the same path as
	// the trivial large-ROI branch); shifting grids are only needed for small ROIs.
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
	// The divider walks ONE closed contour. merge_multicontour concatenates every contour of the
	// ROI, so a multi-contour ROI (holes / disconnected fragments) injects spurious seam chords
	// between them that bias the dimension; this path assumes a single-contour ROI.
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

