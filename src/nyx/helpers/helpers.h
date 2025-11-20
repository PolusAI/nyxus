#pragma once
#include <cfloat>	// FLT_EPSILON
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace Nyxus
{
	constexpr double INF = 10E200;	// Cautious infinity

	// String manipulation

	inline std::string toupper (const std::string& s)
	{
		auto s_uppr = s;
		for (auto& c : s_uppr)
			c = ::toupper(c);
		return s_uppr;
	}

	inline void parse_delimited_string(const std::string& rawString, const std::string& delim, std::vector<std::string>& result)
	{
		result.clear();

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

	// No tabs in 'txt' !
	inline std::string box_text(const std::string& txt)
	{
		std::vector<std::string> L;
		parse_delimited_string(txt, "\n", L);

		size_t maxlen = 0, curlen = 0;
		for (const auto& l : L)
		{
			auto len = l.size();
			maxlen = (std::max)(maxlen, len);
		}

		std::stringstream ss;

		ss << '+';
		for (auto i = 0; i < maxlen + 2; i++)
			ss << '-';
		ss << '+' << '\n';

		for (const auto& l : L)
		{
			ss << "| ";
			ss << l;
			for (auto i = l.size(); i < maxlen; i++)
				ss << ' ';
			ss << " |\n";
		}

		ss << '+';
		for (auto i = 0; i < maxlen + 2; i++)
			ss << '-';
		ss << '+' << '\n';

		return ss.str();
	}

	inline bool parse_as_float(const std::string& raw, float& result)
	{
		char* endptr;
		const char* psz = raw.c_str();
		float res = strtof (psz, &endptr);

		// Did conversion happen?
		if (endptr == psz)
			return false;

		// Was it successful?
		if (*endptr != 0)
			return false;

		// Successful conversion, return its result
		result = res;
		return true;
	}

	inline bool parse_as_int(const std::string& raw, int& result)
	{
		char* endptr;
		const char* psz = raw.c_str();
		long res = strtol(psz, &endptr, 10);

		// Did conversion happen?
		if (endptr == psz)
			return false;
		
		// Was it successful?
		if (*endptr != 0)
			return false;
		
		// Successful conversion, return its result
		result = (int)res;
		return true;
	}

	inline bool parse_as_bool (const std::string& raw, bool& result)
	{
		auto uraw = Nyxus::toupper(raw);
		if (!(uraw == "TRUE" || uraw == "FALSE" || uraw == "T" || uraw == "F"))
			return false;

		result = (uraw == "TRUE" || uraw == "T");

		return true;
	}

	inline bool parse_delimited_string_list_to_ints(const std::string& rawString, std::vector<int>& result, std::string& error_msg)
	{
		// Blank list is legal
		if (rawString.length() == 0)
			return true;

		// Parse the list
		std::vector<std::string> strings;
		parse_delimited_string(rawString, ",", strings);
		result.clear();
		for (auto& s : strings)
		{
			int v;
			if (!parse_as_int(s, v))
			{
				error_msg = "syntax error";
				return false;
			}
			else
				result.push_back(v);
		}
		return true;
	}

	inline bool parse_delimited_string_list_to_doubles (const std::string& rawString, std::vector<double>& result, std::string& error_msg)
	{
		// Blank list is legal
		if (rawString.length() == 0)
			return true;

		// Parse the list
		std::vector<std::string> strings;
		parse_delimited_string(rawString, ",", strings);
		result.clear();
		for (auto& s : strings)
		{
			float v;
			if (!parse_as_float(s, v))
			{
				error_msg = "Error: in '" + rawString + "' expecting '" + s + "' to be a real value";
				return false;
			}
			else
				result.push_back(v);
		}
		return true;
	}

	// File path manipulation
	inline std::string baseFname (const std::string & fpath)
	{
		std::string baseFN = fpath.substr (fpath.find_last_of("/\\") + 1);
		return baseFN;
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

	std::time_t getCurTime();

	// returns seconds
	double getTimeDiff(std::time_t beg, std::time_t end);
	
	std::string getTimeStr (std::time_t t = getCurTime());

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
		if (disable_binning) 
			return i;
		
		double pi = ((double(i-min_i) / double(i_range) * double(n_levels)));
		unsigned int new_pi = (unsigned int)pi;
		return new_pi;
	}

	bool parse_as_float(const std::string& raw, float& result);

	/**
	 * @brief Check is str ends with substr
	 * 
	 * @param str String to check ending of
	 * @param substr Ending to check for
	 * @return true str ends with substr
	 * @return false str does not end with substr
	 */
	inline bool ends_with_substr(const std::string& str, const std::string& substr) {

		if (str.length() >= substr.length()) {
        	return (0 == str.compare(str.length() - substr.length(), substr.length(), substr));
		} 

		return false;
	}

	inline double rad2deg (double x)
	{
		return x / 3.14159265358979323846 * 180.;
	}
	
	inline double deg2rad (double x)
	{
		return x / 180. * 3.14159265358979323846;
	}

	inline double force_finite_number (double x, double nan_substitute)
	{
		if (std::isnan(x) || std::isinf(x))
			return nan_substitute;
		else
			return x;
	}

	inline std::tuple<size_t, size_t> get_minmax_idx (const std::vector<double> & vec)
	{
		size_t n = vec.size();

		if (n == 0)
			return { 0,0 };

		const double *ptr = vec.data();

		size_t smallest = 0, largest = 0;
		double smallestVal = ptr[smallest];
		double largestVal = ptr[largest];

		for (size_t i=1; i<n; i++)
		{
			if (ptr[i] < smallestVal)
			{
				smallest = i;
				smallestVal = ptr[smallest];
			}
			if (ptr[i] > largestVal)
			{
				largest = i;
				largestVal = ptr[largest];
			}
		}

		return { smallest, largest };
	}

	inline std::string virguler_ulong (size_t x)
	{
		const char SEP = ',';

		std::string s1 = std::to_string(x);
		size_t s1_len = s1.length(),
			n_vir = s1_len / 3,
			s2_len = s1.length() + n_vir;
		std::string s2(s2_len, '_');

		size_t k = s2_len - 1;
		for (size_t i = s1_len; i >= 1; i--)
		{
			char c1 = s1[i - 1];
			s2[k--] = c1;
			if ((s1_len - i) && (s1_len - i + 1) % 3 == 0)
				s2[k--] = SEP;
			continue;
		}

		// case where s1_len % 3 != 0
		if (s2[0] == SEP)
			s2.erase(0, 1);

		return s2;
	}

	inline std::string virguler_real (double x)
	{
		// is 'x' special or weird?
		double lowest = (std::numeric_limits<double>::lowest)();
		if (x == lowest)
			return "<LOWEST>";
		double biggest = (std::numeric_limits<double>::max)();
		if (x == biggest)
			return "<BIGGEST>";

		// integer part
		size_t y = (size_t) std::abs(x);

		// fractional part ("123.4567" -> "4567")
		double f = x - (int) x;
		std::string s = std::signbit(x) ? "-" : "";
		std::string frac = f==0.0 ? "0.0" : std::to_string(f);
		if (frac.length()>=1)
			frac = frac.erase(0, 1);
		
		// all together
		s += virguler_ulong(y) + frac;
		return s;
	}

	template <typename T>
	inline std::string virguler (const std::vector<T> & x)
	{
		std::string rv;
		for (auto i = 0; i < x.size(); i++)
			rv += (i ? "," : "") + std::to_string(x[i]);
		return rv;
	}

	inline std::string remove_whitespaces (const std::string & s)
	{
		std::string s2;
		for (char c : s) 
			if (!std::isspace(c)) 
				s2 += c;
		return s2;
	}

	inline bool near_eq (double a, double b)
	{
		return std::abs(a - b) <= FLT_EPSILON;
	}

	inline double det4(
		double m00, double m01, double m02, double m03,
		double m10, double m11, double m12, double m13,
		double m20, double m21, double m22, double m23,
		double m30, double m31, double m32, double m33)
	{
		double retval = m03 * m12 * m21 * m30 - m02 * m13 * m21 * m30 -
			m03 * m11 * m22 * m30 + m01 * m13 * m22 * m30 +
			m02 * m11 * m23 * m30 - m01 * m12 * m23 * m30 -
			m03 * m12 * m20 * m31 + m02 * m13 * m20 * m31 +
			m03 * m10 * m22 * m31 - m00 * m13 * m22 * m31 -
			m02 * m10 * m23 * m31 + m00 * m12 * m23 * m31 +
			m03 * m11 * m20 * m32 - m01 * m13 * m20 * m32 -
			m03 * m10 * m21 * m32 + m00 * m13 * m21 * m32 +
			m01 * m10 * m23 * m32 - m00 * m11 * m23 * m32 -
			m02 * m11 * m20 * m33 + m01 * m12 * m20 * m33 +
			m02 * m10 * m21 * m33 - m00 * m12 * m21 * m33 -
			m01 * m10 * m22 * m33 + m00 * m11 * m22 * m33;
		return retval;
	}

	double calc_mean (const std::vector<double>& series);

	double calc_covariance (const std::vector<double>& series1, double mean1, const std::vector<double>& series2, double mean2);

	void calc_cov_matrix (double Sigma[3][3], const std::vector<std::vector<double>>& table_N_by_3);

	bool calc_eigvals (double w[3], const double A[3][3]);
}
