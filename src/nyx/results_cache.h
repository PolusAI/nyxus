#pragma once
#include <string>
#include <vector>

class ResultsCache
{
public:

	ResultsCache() {}

	void clear()
	{
		headerBuf.clear();
		stringColBuf.clear();
		calcResultBuf.clear();
		totalNumLabels = 0;
	}

	std::vector<std::string>& get_headerBuf() { return headerBuf; }
	std::vector<std::string>& get_stringColBuf() { return stringColBuf; }
	std::vector<double>& get_calcResultBuf() { return calcResultBuf; }

	void add_to_header(std::initializer_list<std::string> cols)
	{
		for (auto c : cols)
			add_to_header(c);
	}
	void add_to_header(std::string& col)
	{
		headerBuf.push_back(col);
	}

	void add_string (const std::string& s) { stringColBuf.push_back(s); }
	void add_numeric(double n) { calcResultBuf.push_back(n); }
	void inc_num_rows() { totalNumLabels++; }
	size_t get_num_rows() { return totalNumLabels; }

private:

	std::vector<double> calcResultBuf;
	size_t totalNumFeatures = 0, totalNumLabels = 0;
	std::vector<std::string> stringColBuf, headerBuf;
};