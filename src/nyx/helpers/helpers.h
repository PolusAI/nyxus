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

	inline bool parse_as_float(const std::string& raw, float& result)
	{
		if (sscanf(raw.c_str(), "%f", &result) != 1)
			return false;
		else
			return true;
	}

	inline bool parse_as_int(const std::string& raw, int& result)
	{
		if (sscanf(raw.c_str(), "%d", &result) != 1)
			return false;
		else
			return true;
	}

	inline bool parse_delimited_string_list_to_ints(const std::string& rawString, std::vector<int>& result, std::string& error_msg)
	{
		// It's legal to not have rotation angles specified
		if (rawString.length() == 0)
			return true;

		bool retval = true;
		std::vector<std::string> strings;
		parse_delimited_string(rawString, ",", strings);
		result.clear();
		for (auto& s : strings)
		{
			int v;
			if (!parse_as_int(s, v))
			{
				retval = false;
				error_msg = "Error: in '" + rawString + "' expecting '" + s + "' to be an integer number";
			}
			else
				result.push_back(v);
		}
		return retval;
	}

	inline std::string toupper(const std::string& s)
	{
		auto s_uppr = s;
		for (auto& c : s_uppr)
			c = ::toupper(c);
		return s_uppr;
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

	// Reserved for later
	//	auto operator ""_MB(unsigned long long const x) -> unsigned long long
	//	{
	//		return 1024L * 1024L * x;
	//	}
	//

	inline double fast_log10 (double _x)  // compute log2(x) by reducing x to [0.75, 1.5)
	{
		float x = (float)_x;

		// a*(x-1)^2 + b*(x-1) approximates log2(x) when 0.75 <= x < 1.5
		const float a = -.6296735f;
		const float b = 1.466967f;
		float signif, fexp;
		int exp;
		float lg2;
		union { float f; unsigned int i; } ux1, ux2;
		int greater;		// actually, a boolean 

		//
		// Assuming IEEE representation, which is sgn(1):exp(8):frac(23)
		// representing (1+frac)*2^(exp-127)  Call 1+frac the significand
		//

		 // get exponent
		ux1.f = x;
		exp = (ux1.i & 0x7F800000) >> 23;	// actual exponent is exp-127, will subtract 127 later

		greater = ux1.i & 0x00400000;  // true if signif > 1.5
		if (greater)
		{
			// signif >= 1.5 so need to divide by 2.  Accomplish this by 
			// stuffing exp = 126 which corresponds to an exponent of -1 
			ux2.i = (ux1.i & 0x007FFFFF) | 0x3f000000;
			signif = ux2.f;
			fexp = exp - 126.0f;    // 126 instead of 127 compensates for division by 2
			signif = signif - 1.0f;                    // <
			lg2 = fexp + a * signif * signif + b * signif;  // <
		}
		else
		{
			// get signif by stuffing exp = 127 which corresponds to an exponent of 0
			ux2.i = (ux1.i & 0x007FFFFF) | 0x3f800000;
			signif = ux2.f;
			fexp = exp - 127.0f;
			signif = signif - 1.0f;                    // <<--
			lg2 = fexp + a * signif * signif + b * signif;  // <<--
		}

		// lines marked <<-- are common code, but optimize better 
		//  when duplicated, at least when using gcc

		return lg2 * 0.30102999566;	// log2 to log10
	}

	/// @brief Converts intensity to uint8
	/// @param i Source pixel intensity
	/// @param min_i Minimum ROI's intensity
	/// @param i_range Precalculated ROI's intensity range (= max-min)
	/// @return Squeezed intensity within range [0,255]
	inline unsigned int to_grayscale (unsigned int i, unsigned int min_i, unsigned int i_range, unsigned int n_levels, bool disable_binning=false)
	{
		if (disable_binning) return i;
		
		unsigned int new_pi = (unsigned int) ((double(i-min_i) / double(i_range) * double(n_levels))) ;
		return new_pi;
	}

	bool parse_as_float(const std::string& raw, float& result);
}

