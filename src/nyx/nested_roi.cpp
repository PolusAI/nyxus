#include <fstream>
#include <sstream>
#include "globals.h"
#include "nested_roi.h"

std::string HieLR::get_output_csv_fname()
{
	std::string fullPath = "_INT_" + Nyxus::getPureFname(segFname) + "_SEG_" + Nyxus::getPureFname(segFname) + ".csv";
	return fullPath;
}

namespace Nyxus
{
	/// @brief Tables referring ROI labels to their cache per each parent-child image pair 
	std::unordered_set <int> uniqueLabels1, uniqueLabels2;
	std::unordered_map <int, HieLR> roiData1, roiData2;
	std::string theParFname, theChiFname;
}
