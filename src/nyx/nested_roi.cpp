#include "globals.h"
#include "nested_roi.h"

std::string HieLR::get_output_csv_fname()
{
	std::string fullPath = "_INT_" + getPureFname(segFname) + "_SEG_" + getPureFname(segFname) + ".csv";
	return fullPath;
}

namespace Nyxus
{
	/// @brief Tables referring ROI labels to their cache per each parent-child image pair 
	std::unordered_set <int> uniqueLabels1, uniqueLabels2;
	std::unordered_map <int, HieLR> roiData1, roiData2;
	std::string theParFname, theChiFname;

	void parse_csv_line(std::vector<std::string>& dst, std::istringstream& src)
	{
		dst.clear();
		std::string field;
		while (getline(src, field, ','))
			dst.push_back(field);
	}

	bool find_csv_record(std::string& csvLine, std::vector<std::string>& csvHeader, std::vector<std::string>& csvFields, const std::string& csvFP, int label)
	{
		std::ifstream f(csvFP);
		std::string line;
		std::istringstream ssLine;

		// just store the header
		std::getline(f, line);
		ssLine.str(line);
		parse_csv_line(csvHeader, ssLine);

		while (std::getline(f, line))
		{
			std::istringstream ss(line); // ssLine.str(line);
			parse_csv_line(csvFields, ss);

			std::stringstream ssLab;
			ssLab << label;
			if (csvFields[2] == ssLab.str())
			{
				csvLine = line;
				return true;
			}
		}

		return false;
	}
}