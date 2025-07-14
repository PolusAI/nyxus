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

double Pixel2::min_sqdist(const std::vector<Pixel2>& cloud) const
{
	#if 0	
	//
	// v1, slower version not requiring the ordered contour
	// 

	auto extrem_d = sqdist(cloud[0]);
	int mindi = 0;

	size_t n = cloud.size();

	for (size_t i = 1; i < n; i++)
	{
		auto dist = sqdist(cloud[i]);
		if (dist < extrem_d)
		{
			extrem_d = dist;
			mindi = i;
		}
	}
	#endif

	//
	// v2
	//
	size_t n = cloud.size();
	size_t a = 0, b = n;
	auto extrem_d = sqdist(cloud[a]);
	auto extrem_i = a;
	int step = (b - a) / log(b - a);
	do
	{
		for (size_t i = a + step; i < b; i += step)
		{
			auto dist = sqdist(cloud[i]);
			if (extrem_d > dist)
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

double Pixel2::max_sqdist(const std::vector<Pixel2>& cloud) const
{
	#if 0	
	//
	// v1, slower version not requiring the ordered contour
	// 

	auto extrem_d = sqdist(cloud[0]);
	int mindi = 0;

	size_t n = cloud.size();

	for (size_t i = 1; i < n; i++)
	{
		auto dist = sqdist(cloud[i]);
		if (dist > extrem_d)
		{
			extrem_d = dist;
			mindi = i;
		}
	}
	#endif

	//
	// v2
	//
	size_t n = cloud.size();
	size_t a = 0, b = n;
	auto extrem_d = sqdist(cloud[a]);
	auto extrem_i = a;
	int step = (b - a) / log(b - a);
	do
	{
		for (size_t i = a + step; i < b; i += step)
		{
			auto dist = sqdist(cloud[i]);
			if (extrem_d < dist)
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

/*static*/ int Pixel2::find_center(const std::vector<Pixel2>& cloud, const std::vector<Pixel2>& contour)
{
	int idxMinDif = 0;
	auto minmaxDist = cloud[idxMinDif].min_max_sqdist(contour);
	double minDif = minmaxDist.second - minmaxDist.first;
	for (size_t n = cloud.size(), i = 1; i < n; i++)
	{
		auto minmaxDist = cloud[i].min_max_sqdist(contour);
		double dif = minmaxDist.second - minmaxDist.first;
		if (dif < minDif)
		{
			minDif = dif;
			idxMinDif = i;
		}
	}
	return idxMinDif;
}

std::pair<double, double> Pixel2::min_max_sqdist(const std::vector<Pixel2>& contour) const
{
	auto mind = min_sqdist(contour),
		maxd = max_sqdist(contour);
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

double Pixel2::dist_to_segment (const Pixel2 & p1, const Pixel2 & p2) const
{
	double dx = p2.x - p1.x,
		dy = p2.y - p1.y;

	double h = dx * dx + dy * dy;
	if (h <= 0)
		return (double)INT_MAX;

	double retval = std::fabs(dy*this->x - dx*this->y + p2.x*p1.y - p2.y*p1.x) / sqrt(h);
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

double Pixel2::sqdist(const Pixel2& px) const
{
	double retval = sqdist(px.x, px.y);
	return retval;
}

double Pixel2::sqdist(int x, int y) const
{
	double dx = (double)x - double(this->x),
		dy = (double)y - double(this->y);
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
		std::vector<double> voxRow = { ((double)vox.x - ccx) / n, ((double)vox.y - ccy) / n, ((double)vox.z - ccz) / n };
		table.push_back({ voxRow });
	}

	Nyxus::calc_cov_matrix (Sigma, table);
}



