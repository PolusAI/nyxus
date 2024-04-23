#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <map>
#include <algorithm>

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
				error_msg = "Error: in '" + rawString + "' expecting '" + s + "' to be an integer number";
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

	inline std::string toupper(const std::string& s)
	{
		auto s_uppr = s;
		for (auto& c : s_uppr)
			c = ::toupper(c);
		return s_uppr;
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

	inline double force_finite_number (double x, double nan_substitute = 0.0)
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

	template <class T>
	inline std::vector<T> remove_padding(const std::vector<T>& img, int img_row, int img_col, int original_row_size, int original_col_size) {
    
		std::vector<T> out(original_row_size * original_col_size, 0);

		for (int i = original_row_size-1; i < 2 * original_row_size - 1; ++i) {
			for (int j = original_col_size-1; j < 2 * original_col_size - 1; ++j) {
				out[(i-original_row_size+1) * original_col_size + (j-original_col_size+1)] = img[i * img_col + j];
			}
		}

		return out;
	}

	inline  std::vector<unsigned int> add_mirror_boundary(std::vector<unsigned int>& img, int rows, int cols) {

		auto temp = img;

		int new_col_size = cols + 2 * (cols-1);
		int new_row_size = rows + 2 * (rows-1);

		auto initial_row_size = rows;
		auto initial_col_size = cols;

		size_t size;
		// Append the reversed vector to itself, excluding the first and last rows
		for (int i = 1; i < rows; ++i) {
			std::vector<unsigned int>::const_iterator first = img.begin() + (i * rows);
			std::vector<unsigned int>::const_iterator last = img.begin() + ((i+1) * rows);
			temp.insert(temp.begin(), first, last);
		}

		auto new_size = temp.size();
		
		for (int i = rows-2; i >= 0 ; --i) {
			temp.insert(temp.end(), img.begin() + i * cols, img.begin() +  ((i+1) * rows));
		}
		
		std::vector<unsigned int> padded;

		for (int i = 0; i < new_row_size; ++i) {

			std::vector<unsigned int> row = std::vector<unsigned int>(temp.begin() + i * cols, temp.begin() + (i+1) * cols);

			std::vector<unsigned int> mirrored_row = row; // Copy the row
			std::reverse(mirrored_row.begin(), mirrored_row.end()); // Reverse the copied row

			row.insert(row.begin(), mirrored_row.begin(), mirrored_row.end()-1); // Append the mirrored row to the original row
			row.insert(row.end(), mirrored_row.begin()+1, mirrored_row.end());

			padded.insert(padded.end(), row.begin(), row.end());
		}
		
		
		return padded;
	}


	inline std::vector<int> arrange(int start, int end, int step=1) {

		std::vector<int> out;

		for (int i = start; i < end; ++i) {
			out.push_back(i);
		}

		return out;
 
	}

	inline std::vector<double> nd_sum(std::vector<double> input, std::vector<int> labels, std::vector<int> index) {

		if (input.size() != labels.size()) throw std::runtime_error("input vector and labels vector must be the same size");

		std::map<int, double> sum_values;

		for (const auto& value: index) {
			sum_values[value] = 0.;
		}
		
		for (int i = 0; i < input.size(); ++i) {
			sum_values[labels[i]] += input[i];
		}

		std::vector<double> out;

		for (std::map<int, double>::iterator it = sum_values.begin(); it != sum_values.end(); ++it) {
			out.push_back(it->second);
		}

		return out;
	}


	inline std::vector<std::vector<int>> flipud (std::vector<std::vector<int>> vec) {

		std::vector<std::vector<int>> out;

		for (int i = vec.size()-1; i >=0; --i) {
			out.push_back(vec[i]);
		}

		return out;
	}

	inline std::vector<std::vector<int>> fliplr (std::vector<std::vector<int>> vec) {
		

		std::vector<std::vector<int>> out = vec;

		for (auto& row: out) {
			std::reverse(row.begin(), row.end());
		}

		return out;
	}

	inline std::vector<std::vector<int>> minimum(std::vector<std::vector<int>> vec1, std::vector<std::vector<int>> vec2) {

		std::vector<std::vector<int>> out = vec1;

		for (int i = 0; i < vec1.size(); ++i) {
			for (int j = 0; j < vec1[0].size(); ++j) {
				if (vec2[i][j] < vec1[i][j]){
					out[i][j] = vec2[i][j];
				}
			}
		} 

		return out;
	}

	template <class T>
	inline double mean(std::vector<T> vec) {

		double accum = 0;
		for (auto& element: vec ) {
			accum += element;
		}
		
		return accum / (double)vec.size();
	}

	template <class T>
	std::vector<double> normalize(const std::vector<T>& vec) {

		auto out = vec;
		double sum = 0;

		for (auto& element: out) {
			sum += element;
		}
		
		// Avoid division by zero
		if (sum == 0) return out;

		for (auto& element: out) {
			element /= sum;
		}

		return out;
	}

	template <class T>
	std::vector<T> transpose_vector(const std::vector<T>& image, int rows, int cols) {
		std::vector<T> transposed(cols * rows);
		
		// Transpose the flattened vector
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				transposed[j * cols + i] = image[i * cols + j];
			}
		}
		
		return transposed;
	}
}
