#include <fstream>
#include "environment.h"
#include "helpers.h"
#include "timing.h"


void Stopwatch::print_stats()
{
	double total = 0.0;
	for (auto& t : totals)
		total += t.second;

	std::cout << "--------------------\nTotal time of all feature groups = " << total/1e6 << "\nBreak-down:\n--------------------\n";

	for (auto& t : totals)
	{
		double perc = t.second * 100.0 / total;
		std::cout << t.first << "\t" << round2(perc) << "%\t" << t.second << "\n";
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
		parse_delimited_string(t.first, "/", nameParts);
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
			<< fcolor << "\",\"" << facro << " " << round2(perc) << "%\","
			<< t.second 
			<< "," << total
			<< "," << theEnvironment.n_reduce_threads 
			<< "\n";
	}

}
