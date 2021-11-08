#pragma once

#include <iostream>

// Geometry
inline double angle(const double x1, const double y1, double x2, const double y2)
{
	double dotProd = x1 * x2 + y1 * y2,
		magThis = std::sqrt(x1 * x1 + y1 * y1),
		magOther = std::sqrt(x2 * x2 + y2 * y2),
		cosVal = dotProd / (magThis * magOther),
		ang = std::acos(cosVal);
	return ang;
}


// Statistics
inline int mode(const std::vector<int>& v)
{
	int max = v.back();
	int min = v.front();
	int prev = max;
	int mode = 0;
	int maxcount = 0;
	int currcount = 0;
	for (const auto n : v)
	{
		if (n == prev)
		{
			++currcount;
			if (currcount > maxcount)
			{
				maxcount = currcount;
				mode = n;
			}
		}
		else
		{
			currcount = 1;
		}
		prev = n;
	}

	return mode;
}

// General helpers

inline int closest_pow2(const int a)
{
	int n = a;
	int cnt = 0;
	for (; n;)
	{
		n = n >> 1;
		cnt++;
	}
	int retval = 1 << cnt;
	return retval;
}

inline void print_curve (const std::vector<std::pair<int, int>>& curve, const std::string & name)
{
	std::cout << "\n\n" << name << " = [\n";
	for (auto& xy : curve)
	{
		std::cout << xy.first << ", " << xy.second << ";\n";
	}
	std::cout << "]\n";
}

