#pragma once
#include <string>
#include <vector>

class ResultsCache
{
public:

	ResultsCache() {}

	void clear()
	{
		headerBuf_.clear();
		stringColBuf_.clear();
		calcResultBuf_.clear();
		totalNumLabels_ = 0;
	}

	std::vector<std::string>& get_headerBuf() { return headerBuf_; }
	std::vector<std::string>& get_stringColBuf() { return stringColBuf_; }
	std::vector<double>& get_calcResultBuf() { return calcResultBuf_; }

	void add_to_header(std::initializer_list<std::string> cols)
	{
		for (auto c : cols)
			add_to_header(c);
	}
	void add_to_header(std::string& col)
	{
		headerBuf_.push_back(col);
	}

	void add_string (const std::string& s) { stringColBuf_.push_back(s); }
	void add_numeric(double n) { calcResultBuf_.push_back(n); }
	void inc_num_rows() { totalNumLabels_++; }
	size_t get_num_rows() { return totalNumLabels_; }

private:

	std::vector<double> calcResultBuf_;
	size_t totalNumLabels_ = 0;
	std::vector<std::string> stringColBuf_, headerBuf_;
};

// Global feature extraction results table
namespace Nyxus
{
	extern ResultsCache theResultsCache;
}
