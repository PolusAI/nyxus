#pragma once
#include <string>
#include "roi_cache.h"

/// @brief Segment data cache for finding segment hierarchies 
class HieLR: public BasicLR //: public LR
{
public:
	std::vector<int> child_segs;
	std::string get_output_csv_fname();
	std::string segFname;
};

namespace Nyxus
{
	/// @brief Tables referring ROI labels to their cache per each parent-child image pair 
	extern std::unordered_set <int> uniqueLabels1, uniqueLabels2;
	extern std::unordered_map <int, HieLR> roiData1, roiData2;
	extern std::string theParFname, theChiFname;

	void parse_csv_line(std::vector<std::string>& dst, std::istringstream& src);
	bool find_csv_record(std::string& csvLine, std::vector<std::string>& csvHeader, std::vector<std::string>& csvFields, const std::string& csvFP, int label);
}