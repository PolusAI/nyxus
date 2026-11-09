#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <numeric>
#include <climits>
#include "pixel.h"
#include "../helpers/helpers.h"

bool operator == (const Pixel2& p1, const Pixel2& p2)
{
	if (p1.x != p2.x || p1.y != p2.y || p1.inten != p2.inten)
		return false;
	return true;
}

namespace
{
	// The extreme squared distance from 'p' to the ordered walk W[0..n), n >= 1: the smallest, or the
	// largest when 'farther' is set. Up to 10 points are scanned exhaustively. A longer walk is
	// sampled at a stride of about n / ln n, and the stride shrinks around the best sample until it
	// is 1, so only that neighbourhood is searched at full resolution.
	double search_sqdist (const Pixel2& p, const Pixel2* W, size_t n, bool farther, double sx, double sy)
	{
		auto better = [farther] (double dist, double extrem) { return farther ? dist > extrem : dist < extrem; };

		auto extrem_d = p.sqdist (W[0], sx, sy);
		if (n <= 10)
		{
			for (size_t i = 1; i < n; i++)
			{
				auto dist = p.sqdist (W[i], sx, sy);
				if (better (dist, extrem_d))
					extrem_d = dist;
			}
			return extrem_d;
		}

		size_t a = 0, b = n;
		auto extrem_i = a;
		int step = (b - a) / log(b - a);
		do
		{
			for (size_t i = a + step; i < b; i += step)
			{
				auto dist = p.sqdist (W[i], sx, sy);
				if (better (dist, extrem_d))
				{
					extrem_d = dist;
					extrem_i = i;
				}
			}

			// left or right ?
			auto stepL = extrem_i >= step ? step : extrem_i,
				stepR = extrem_i + step < n ? step : n - extrem_i;

			a = extrem_i - stepL;
			b = extrem_i + stepR;
			step = b - a <= 10 ? 1 : (b - a) / log(b - a);
		}
		while (b - a > 2);

		return extrem_d;
	}

	// The squared distances from 'p' to the nearest point of the box of border 'B' and to its
	// farthest corner: no pixel of the border is nearer than the first or farther than the second.
	// Taken on whole pixel offsets like Pixel2::sqdist, so they compare exactly against it.
	double box_nearest_sqdist (const Pixel2& p, const BorderedContour::Border& B, double sx, double sy)
	{
		double dx = double(p.x < B.xmin ? B.xmin - p.x : (p.x > B.xmax ? p.x - B.xmax : 0)) * sx,
			dy = double(p.y < B.ymin ? B.ymin - p.y : (p.y > B.ymax ? p.y - B.ymax : 0)) * sy;
		return dx * dx + dy * dy;
	}

	double box_farthest_sqdist (const Pixel2& p, const BorderedContour::Border& B, double sx, double sy)
	{
		double dx = double((std::max) (std::abs (p.x - B.xmin), std::abs (p.x - B.xmax))) * sx,
			dy = double((std::max) (std::abs (p.y - B.ymin), std::abs (p.y - B.ymax))) * sy;
		return dx * dx + dy * dy;
	}

	// The extremes of search_sqdist() over the borders of 'K', each border searched on its own: the
	// smallest into *mind and the largest into *maxd, either pointer null when that extreme is not
	// wanted, and only the box bounds of a wanted extreme taken. The border with the most promising
	// bound is searched first; a border whose bound cannot beat the extreme found so far is then
	// skipped, which leaves the result as if every border were searched. An empty contour gives 0.
	void search_sqdist_borders (const Pixel2& p, const BorderedContour& K, double sx, double sy, double* mind, double* maxd)
	{
		const auto& B = K.borders;
		if (B.empty())
		{
			if (mind) *mind = 0.0;
			if (maxd) *maxd = 0.0;
			return;
		}

		auto search = [&] (size_t k, bool farther) { return search_sqdist (p, K.pixels.data() + B[k].begin, B[k].end - B[k].begin, farther, sx, sy); };

		if (B.size() == 1)
		{
			if (mind) *mind = search (0, false);
			if (maxd) *maxd = search (0, true);
			return;
		}

		// per-thread scratch, so that a query allocates nothing once it has grown
		thread_local std::vector<double> nearest, farthest;
		size_t firstMin = 0, firstMax = 0;
		if (mind)
		{
			nearest.resize (B.size());
			for (size_t k = 0; k < B.size(); k++)
			{
				nearest[k] = box_nearest_sqdist (p, B[k], sx, sy);
				if (nearest[k] < nearest[firstMin])
					firstMin = k;
			}
		}
		if (maxd)
		{
			farthest.resize (B.size());
			for (size_t k = 0; k < B.size(); k++)
			{
				farthest[k] = box_farthest_sqdist (p, B[k], sx, sy);
				if (farthest[k] > farthest[firstMax])
					firstMax = k;
			}
		}

		if (mind)
		{
			double extrem_d = search (firstMin, false);
			for (size_t k = 0; k < B.size(); k++)
				if (k != firstMin && nearest[k] < extrem_d)
					extrem_d = (std::min) (extrem_d, search (k, false));
			*mind = extrem_d;
		}
		if (maxd)
		{
			double extrem_d = search (firstMax, true);
			for (size_t k = 0; k < B.size(); k++)
				if (k != firstMax && farthest[k] > extrem_d)
					extrem_d = (std::max) (extrem_d, search (k, true));
			*maxd = extrem_d;
		}
	}
}

bool border_ends_cover (size_t n_pixels, const std::vector<size_t>& ends)
{
	size_t begin = 0;
	for (size_t end : ends)
	{
		if (end <= begin)
			return false;
		begin = end;
	}
	return begin == n_pixels;
}

BorderedContour::BorderedContour (const std::vector<Pixel2>& pixels_, const std::vector<size_t>& ends)
	: pixels (pixels_)
{
	if (!border_ends_cover (pixels.size(), ends))
		throw std::invalid_argument ("BorderedContour: border ends do not cover the " + std::to_string(pixels.size()) + " contour pixels in nonempty runs");

	size_t begin = 0;
	for (size_t end : ends)
	{
		Border b { begin, end, pixels[begin].x, pixels[begin].x, pixels[begin].y, pixels[begin].y };
		for (size_t i = begin + 1; i < end; i++)
		{
			b.xmin = (std::min) (b.xmin, pixels[i].x);
			b.xmax = (std::max) (b.xmax, pixels[i].x);
			b.ymin = (std::min) (b.ymin, pixels[i].y);
			b.ymax = (std::max) (b.ymax, pixels[i].y);
		}
		borders.push_back (b);
		begin = end;
	}
}

double Pixel2::min_sqdist (const std::vector<Pixel2>& cloud, double sx, double sy) const
{
	if (cloud.empty())
		return 0.0;
	return search_sqdist (*this, cloud.data(), cloud.size(), false, sx, sy);
}

double Pixel2::min_sqdist (const BorderedContour& contour, double sx, double sy) const
{
	double mind;
	search_sqdist_borders (*this, contour, sx, sy, &mind, nullptr);
	return mind;
}

// exact minimum squared distance via a full linear scan (no ordered-contour assumption).
// Correct replacement for the approximate min_sqdist() v2 on the neighbor touch / radius-gate paths.
double Pixel2::exact_min_sqdist(const std::vector<Pixel2>& cloud) const
{
	if (cloud.empty())
		return 0.0;

	auto extrem_d = sqdist(cloud[0]);
	for (size_t n = cloud.size(), i = 1; i < n; i++)
	{
		auto dist = sqdist(cloud[i]);
		if (dist < extrem_d)
			extrem_d = dist;
	}
	return extrem_d;
}

double Pixel2::max_sqdist (const std::vector<Pixel2>& cloud, double sx, double sy) const
{
	if (cloud.empty())
		return 0.0;
	return search_sqdist (*this, cloud.data(), cloud.size(), true, sx, sy);
}

double Pixel2::max_sqdist (const BorderedContour& contour, double sx, double sy) const
{
	double maxd;
	search_sqdist_borders (*this, contour, sx, sy, nullptr, &maxd);
	return maxd;
}

/*static*/ int Pixel2::find_center(const std::vector<Pixel2>& cloud, const std::vector<Pixel2>& contour, double sx, double sy)
{
	// the whole contour as a single border
	const std::vector<size_t> ends = contour.empty() ? std::vector<size_t>() : std::vector<size_t>{ contour.size() };
	return find_center (cloud, BorderedContour (contour, ends), sx, sy);
}

/*static*/ int Pixel2::find_center(const std::vector<Pixel2>& cloud, const BorderedContour& contour, double sx, double sy)
{
	int idxMinDif = 0;
	auto minmaxDist = cloud[idxMinDif].min_max_sqdist(contour, sx, sy);
	double minDif = minmaxDist.second - minmaxDist.first;
	for (size_t n = cloud.size(), i = 1; i < n; i++)
	{
		auto minmaxDist = cloud[i].min_max_sqdist(contour, sx, sy);
		double dif = minmaxDist.second - minmaxDist.first;
		if (dif < minDif)
		{
			minDif = dif;
			idxMinDif = i;
		}
	}
	return idxMinDif;
}

std::pair<double, double> Pixel2::min_max_sqdist(const std::vector<Pixel2>& contour, double sx, double sy) const
{
	auto mind = min_sqdist(contour, sx, sy),
		maxd = max_sqdist(contour, sx, sy);
	return { mind, maxd };
}

std::pair<double, double> Pixel2::min_max_sqdist(const BorderedContour& contour, double sx, double sy) const
{
	// both extremes off one call, each from its own box bounds
	double mind, maxd;
	search_sqdist_borders (*this, contour, sx, sy, &mind, &maxd);
	return { mind, maxd };
}

double Pixel2::angle(const Pixel2& other) const
{
	double dotProd = double(this->x * other.x) + double(this->y * other.y),
		magThis = std::sqrt(this->x * this->x + this->y * this->y),
		magOther = std::sqrt(other.x * other.x + other.y * other.y),
		cosVal = dotProd / (magThis * magOther),
		ang = std::acos(cosVal);
	return ang;
}

bool Pixel2::belongs_to(const std::vector<Pixel2>& cloud) const
{
	for (auto& px : cloud)
		if (this->colocating(px) && this->inten == px.inten)
			return true;
	return false;
}

double Pixel2::sqdist_to_segment (const Pixel2& p1, const Pixel2& p2) const
{
	double x21 = p2.x - p1.x,
		y21 = p2.y - p1.y;
	double retval = (x21 * (p1.y - this->y) - (p1.x - this->x) * y21) / std::sqrt(x21 * x21 + y21 * y21);
	return std::abs(retval);
}

double Pixel2::dist_to_segment (const Pixel2 & p1, const Pixel2 & p2, double sx, double sy) const
{
	// the three points in units of length
	double x0 = this->x * sx, y0 = this->y * sy,
		x1 = p1.x * sx, y1 = p1.y * sy,
		x2 = p2.x * sx, y2 = p2.y * sy;
	double dx = x2 - x1,
		dy = y2 - y1;

	double h = dx * dx + dy * dy;
	if (h <= 0)
		return (double)INT_MAX;

	double retval = std::fabs(dy*x0 - dx*y0 + x2*y1 - y2*x1) / sqrt(h);
	return retval;
}

double Pixel2::sum_sqdist(const std::vector<Pixel2>& cloud) const
{
	double retval = 0.0;
	for (auto& px : cloud)
	{
		double sqd = this->sqdist(px);
		retval += sqd;
	}
	return retval;
}

double Pixel2::sqdist(const Pixel2& px, double sx, double sy) const
{
	double retval = sqdist(px.x, px.y, sx, sy);
	return retval;
}

double Pixel2::sqdist(int x, int y, double sx, double sy) const
{
	double dx = ((double)x - double(this->x)) * sx,
		dy = ((double)y - double(this->y)) * sy;
	double retval = dx * dx + dy * dy;
	return retval;
}

/*static*/ std::tuple<double, double, double> Pixel3::centroid(const std::vector<Pixel3>& A)
{
	double n = A.size(), 
		cx = std::accumulate (A.begin(), A.end(), 0.0, [](double sum, const Pixel3& p) {return sum + p.x;}),
		cy = std::accumulate (A.begin(), A.end(), 0.0, [](double sum, const Pixel3& p) {return sum + p.y;}), 
		cz = std::accumulate (A.begin(), A.end(), 0.0, [](double sum, const Pixel3& p) {return sum + p.z;});
	cx /= n;
	cy /= n;
	cz /= n;
	return { cx, cy, cz };
}

/*static*/ void Pixel3::calc_cov_matrix (double Sigma[3][3], const std::vector<Pixel3>& cloud)
{
	auto [ccx, ccy, ccz] = Pixel3::centroid (cloud);
	double n = cloud.size();
	std::vector<std::vector<double>> table;
	for (auto vox : cloud)
	{
		std::vector<double> voxRow = { (double)vox.x - ccx, (double)vox.y - ccy, (double)vox.z - ccz };
		table.push_back({ voxRow });
	}

	Nyxus::calc_cov_matrix (Sigma, table);
}



