#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace Nyxus
{

	// String manipulation
	inline void parse_delimited_string(const std::string& rawString, const std::string& delim, std::vector<std::string>& result)
	{
		result.clear();

		std::vector<std::string> S;

		std::string raw = rawString;    // a safe copy
		size_t pos = 0;
		std::string token;
		while ((pos = raw.find(delim)) != std::string::npos)
		{
			token = raw.substr(0, pos);
			result.push_back(token);
			raw.erase(0, pos + delim.length());
		}
		result.push_back(raw);
	}

	// Geometry
	inline double angle(const double x1, const double y1, double x2, const double y2)
	{
		double dotProd = x1 * x2 + y1 * y2,
			magThis = std::sqrt(x1 * x1 + y1 * y1),
			magOther = std::sqrt(x2 * x2 + y2 * y2);

		if (magThis * magOther == 0.0)
			return 0;

		double cosVal = dotProd / (magThis * magOther),
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

#define INF 10E200	// Cautious infinity

	inline double round2(const double a)
	{
		double retval = std::round(a * 100.0) / 100.0;
		return retval;
	}

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

	inline void print_curve(const std::vector<std::pair<int, int>>& curve, const std::string& name)
	{
		std::cout << "\n\n" << name << " = [\n";
		for (auto& xy : curve)
		{
			std::cout << xy.first << ", " << xy.second << ";\n";
		}
		std::cout << "]\n";
	}

	std::string getTimeStr(const std::string& head = "", const std::string& tail = "");

	// Inherited from WNDCHRM, used for Feret and Martin statistics calculation
	struct Statistics
	{
		double min, max, mode;
		double mean, median, stdev;
	};

	Statistics ComputeCommonStatistics2(std::vector<double>& Data);

	// 
	//	auto operator ""_MB(unsigned long long const x) -> unsigned long long
	//	{
	//		return 1024L * 1024L * x;
	//	}
	//

}

