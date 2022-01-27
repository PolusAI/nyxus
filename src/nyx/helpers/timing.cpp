#include <fstream>
#include "../environment.h"
#include "helpers.h"
#include "timing.h"

Stopwatch::Stopwatch (const std::string& header_, const std::string& tail_)
{
	header = header_;
	tail = tail_;

	if (totals.find(header) == totals.end())
		totals[header] = 0.0;

	start = std::chrono::system_clock::now();
	if (header.length() > 0)
		PROFUSE(std::cout << header << "\n";)
}

Stopwatch::~Stopwatch()
{
	end = std::chrono::system_clock::now();
	std::chrono::duration<double, Unit> elap = end - start;
	PROFUSE(std::cout << tail << " " << elap.count() << " us\n"; )
		totals[header] = totals[header] + elap.count();
}

void Stopwatch::print_stats()
{
	double total = 0.0;
	for (auto& t : totals)
		total += t.second;

	std::cout << "--------------------\nTotal time of all feature groups = " << total/1e6 << "\nBreak-down:\n--------------------\n";

	for (auto& t : totals)
	{
		double perc = t.second * 100.0 / total;
		std::cout << t.first << "\t" << Nyxus::round2(perc) << "%\t" << t.second << "\n";
	}

	std::cout << "--------------------\n";
}

void Stopwatch::save_stats (const std::string& fpath)
{
	double total = 0.0;
	for (auto& t : totals)
		total += t.second;

	std::ofstream f (fpath);
	
	// header
	f << "\"h1\", \"h2\", \"h3\", \"weight\", \"color\", \"codes\", \"rawtime\", \"totalTime\", \"numReduceThreads\" \n";	
	// body
	for (auto& t : totals)
	{
		// Template: 
		//	"Total", "Intensity", "Intensity", "#f58321", 9.0, "I", "123"
		//	"Total", "Moments", "Spatial", "#ffaabb", 33.3, "Ms", "456"
		//	"Total", "Moments", "Central", "#ffaabb", 57.7, "Mc", "789"

		// Expecting the following feature caption format: category/name/acronym/color e.g. "Moments/Spatial/Ms/#ffaabb"
		// Color paltte reference: https://www.rapidtables.com/web/color/RGB_Color.html
		std::vector<std::string> nameParts;
		Nyxus::parse_delimited_string(t.first, "/", nameParts);
		std::string root = "Total",
			fcateg = nameParts.size() >= 4 ? nameParts[0] : "category",
			fname = nameParts.size() >= 4 ? nameParts[1] : "feature",
			facro = nameParts.size() >= 4 ? nameParts[2] : "acronym",
			fcolor = nameParts.size() >= 4 ? nameParts[3] : "#112233";

		double perc = t.second * 100.0 / total;
		f << "\"Total\",\"" 
			<< fcateg << "\",\"" 
			<< fname << "\"," 
			<< perc << ",\"" 
			<< fcolor << "\",\"" << facro << " " << Nyxus::round2(perc) << "%\","
			<< t.second 
			<< "," << total
			<< "," << Nyxus::theEnvironment.n_reduce_threads 
			<< "\n";
	}
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
} // Nyxus