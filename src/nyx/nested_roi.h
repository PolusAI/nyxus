#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "roi_cache.h"

/// @brief Segment data cache for finding segment hierarchies 
class HieLR: public BasicLR
{
public:
	HieLR() : BasicLR(-1) {} // using default label '-1'
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

	/// @brief Where the columns of a feature CSV are, read off its header. The non-feature
	/// columns (unit, annotation, channel and spacing columns among them) precede the features,
	/// and phys_z is the last of them.
	struct CsvLayout
	{
		int intensity = -1, mask = -1, label = -1, t_index = -1,
			first_feature = -1;
	};
	bool csv_layout (const std::vector<std::string>& header, CsvLayout& layout);

	/// @brief Find the record of ROI 'label' in feature CSV 'csvFP', locating the label column
	/// by the file's header. 'layout' receives the file's column layout.
	bool find_csv_record(std::string& csvLine, std::vector<std::string>& csvHeader, std::vector<std::string>& csvFields, CsvLayout& layout, const std::string& csvFP, int label);
	bool find_csv_record(std::string& csvLine, std::vector<std::string>& csvHeader, std::vector<std::string>& csvFields, const std::string& csvFP, int label);
}