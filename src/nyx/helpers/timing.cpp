#include <fstream>
#include "../environment.h"
#include "helpers.h"
#include "timing.h"

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

std::map <std::string, double> Stopwatch::totals;
bool Stopwatch::inclusive_ = true;

bool Stopwatch::exclusive()
{
	return !inclusive_;
}

bool Stopwatch::inclusive()
{
	return inclusive_;
}

void Stopwatch::set_inclusive(bool incl)
{
	inclusive_ = incl;
}

Stopwatch::Stopwatch (const std::string& header_, const std::string& tail_)
{
	header = header_;
	tail = tail_;

	if (totals.find(header) == totals.end())
		totals[header] = 0.0;

	start = std::chrono::system_clock::now();
	if (header.length() > 0)
		VERBOSLVL1(std::cout << header << "\n";)
}

Stopwatch::~Stopwatch()
{
	end = std::chrono::system_clock::now();
	std::chrono::duration<double, Unit> elap = end - start;
	VERBOSLVL1(std::cout << tail << " " << elap.count() << " us\n"; )
		totals[header] = totals[header] + elap.count();
}

void Stopwatch::reset()
{
	totals.clear();
}

void Stopwatch::print_stats()
{
	double total = 0.0;
	for (auto& t : totals)
		total += t.second;

	std::cout << "--------------------\nTotal time of all feature groups [sec] = " << total/1e6 << "\nBreak-down:\n--------------------\n";

	for (auto& t : totals)
	{
		double perc = t.second * 100.0 / total;
		std::cout << t.first << "\t" << Nyxus::round2(perc) << "%\t" << t.second << "\n";
	}

	std::cout << "--------------------\n";
}

void Stopwatch::save_stats (const std::string & fpath)
{
	// Any experiment info in the file name?
	// (Example 1 - "someimage.csv" is a regular timing file without experimental metadata.
	// Example 2 - "p0_y1_r2_c1.ome.tif.csv" is also a regular timing file without experimental metadata.
	// Example 3 - "synthetic_nrois=10_roiarea=500_nyxustiming.csv" is a timing file containing 2 variable
	// values "nrois=10" and "roiarea=500" that will be reflected in the CSV file as additional columns
	// "nrois" and "roiarea" containing values 10 and 500 respectively.)

	std::vector<std::string> vars, vals;	// experiment variables and their values
	fs::path fpa (fpath);
	std::string stm = fpa.stem().string();
	if (stm.find('=') != std::string::npos)
	{
		// Chop the stem presumably in the form part1_part2_part3_etc
		std::vector<std::string> parts;
		Nyxus::parse_delimited_string (stm, "_", parts);

		// Skip non-informative chops
		for (std::string & p : parts)
		{
			if (p.find('=') == std::string::npos)
				continue;
			// We have a variable name in 'p'
			std::vector<std::string> sides;
			Nyxus::parse_delimited_string (p, "=", sides);
			vars.push_back (sides[0]);
			vals.push_back (sides[1]);
		}
	}

	// Save timing results
	double totTime = 0.0;
	for (auto& t : totals)
		totTime += t.second;

	// report header
	std::ofstream f (fpath);

	const char _quote_ = '\"',
		_comma_ = ',';

	// -- experiment info, if any
	if (! vars.empty())
		for (std::string & lhs : vars)
			f << _quote_ << lhs << _quote_ << _comma_;

	// -- regular columns
	f << _quote_ << "h1" << _quote_ << _comma_
		<< _quote_ << "h2" << _quote_ << _comma_
		<< _quote_ << "h3" << _quote_ << _comma_
		<< _quote_ << "share%" << _quote_ << _comma_
		<< _quote_ << "color" << _quote_ << _comma_
		<< _quote_ << "codes" << _quote_ << _comma_
		<< _quote_ << "rawtime" << _quote_ << _comma_
		<< _quote_ << "totalTime" << _quote_ << _comma_
		<< _quote_ << "numReduceThreads" << _quote_ << "\n";

	// report body
	for (auto& t : totals)
	{
		// -- experiment info, if any
		if (! vals.empty())
			for (std::string & rhs : vals)
				f << _quote_ << rhs << _quote_ << _comma_;

		// Expecting the following feature caption format: category/name/acronym/color e.g. "Moments/Spatial/Ms/#ffaabb"
		// Color paltte reference: https://www.rapidtables.com/web/color/RGB_Color.html
		std::vector<std::string> nameParts;
		Nyxus::parse_delimited_string(t.first, "/", nameParts);
		std::string root = "Total",
			fcateg = nameParts.size() >= 4 ? nameParts[0] : "category",
			fname = nameParts.size() >= 4 ? nameParts[1] : "feature",
			facro = nameParts.size() >= 4 ? nameParts[2] : "acronym",
			fcolor = nameParts.size() >= 4 ? nameParts[3] : "#112233";

		double perc = t.second * 100.0 / totTime;

		// -- regular timing data
		f << _quote_ << "Total" << _quote_ << _comma_
			<< _quote_ << fcateg << _quote_ << _comma_
			<< _quote_ << fname << _quote_ << _comma_
			<< _quote_ << perc << _quote_ << _comma_
			<< _quote_ << fcolor << _quote_ << _comma_
			<< _quote_ << facro << _quote_ << _comma_ //" " << Nyxus::round2(perc) << "%" << _quote_ << _comma_
			<< _quote_ << t.second << _quote_ << _comma_
			<< _quote_ << totTime << _quote_ << _comma_
			<< _quote_ << Nyxus::theEnvironment.n_reduce_threads << _quote_
			<< "\n";
	}

	// Combined time
	// -- experiment info, if any
	if (! vals.empty())
		for (std::string & rhs : vals)
			f << _quote_ << rhs << _quote_ << _comma_;

	f << _quote_ << "Total" << _quote_ << _comma_
		<< _quote_ << "All" << _quote_ << _comma_
		<< _quote_ << "All" << _quote_ << _comma_
		<< _quote_ << "100" << _quote_ << _comma_
		<< _quote_ << "#000000" << _quote_ << _comma_
		<< _quote_ << "TOTL" << _quote_ << _comma_ //" " << Nyxus::round2(100) << "%" << _quote_ << _comma_
		<< _quote_ << totTime << _quote_ << _comma_
		<< _quote_ << totTime << _quote_ << _comma_
		<< _quote_ << Nyxus::theEnvironment.n_reduce_threads << _quote_
		<< "\n";
}

namespace Nyxus
{
	// Requires:
	//		#define _CRT_SECURE_NO_WARNINGS
	//		#include <ctime>
	//
	// Old-school equivalent:
	//		time_t my_time = time(NULL);
	//		printf("Started at %s", ctime(&my_time));
	//
	std::string getTimeStr(const std::string& head /*= ""*/, const std::string& tail /*= ""*/)
	{
		std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

		std::string s(30, '\0');
		std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
		return s;
	}
}
