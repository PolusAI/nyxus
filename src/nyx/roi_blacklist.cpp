#include <algorithm>
#include <sstream>
#include "helpers/helpers.h"
#include "roi_blacklist.h"


bool RoiBlacklist::parse_file_blacklist (const std::string & p)
{
	if (p.find(':') == std::string::npos)
	{
		ermsg = "Error: in " + p + " expecting ':'";
		return false;
	}

	std::vector<std::string> lrhs;
	Nyxus::parse_delimited_string(p, ":", lrhs);

	// Check rule <something1>:<something2>
	if (lrhs.size() != 2)
	{
		ermsg = "Error: in " + p + " expecting syntax <something1>:<something2>";
		return false;
	}

	// Use LHS
	// -- lrhs[0] is supposed to be file name so give it a light check 
	for (auto ch : lrhs[0])
		if (std::isspace(ch))
		{
			ermsg = "Error: " + lrhs[0] + " contains a space character";
			return false;
		}
	// -- file name is OK, use it
	pairType fbl;
	fbl.first = lrhs[0];

	// Use RHS
	std::vector<std::string> labels;
	Nyxus::parse_delimited_string(lrhs[1], ",", labels);

	// 1 list item or N?
	if (labels.size() == 0)
	{
		// We have just 1 item. Cast it to number
		int v;
		if (!Nyxus::parse_as_int(lrhs[1], v))
		{
			ermsg = "Error: expecting " + lrhs[1] + " to be an integer number";
			return false;
		}

		// Save it
		fbl.second.push_back(v);
	}
	else
	{
		// We have N items
		for (auto& s : labels)
		{
			int v;
			if (!Nyxus::parse_as_int(s, v))
			{
				ermsg = "Error: expecting " + s + " to be an integer number";
				return false;
			}

			// Save it
			fbl.second.push_back(v);
		}
	}

	// Save this file blacklist
	fileBlackList.push_back(fbl);
	return true;
}

bool RoiBlacklist::parse_raw_string(const std::string& raw)
{
	// Blank raw string for blacklist is not OK. The blacklist can be cleared via clear() instead of using a blank blacklist string
	if (raw.empty())
	{
		return false;
	}

	// Clear all the blacklists
	clear();

	// Syntax 1A - one or more semicolon-separated list of file-specific comma-separated labels (set of file-specific blacklists)
	// Example: 'raw' is "file1.tif:3,5,7;file2.tif:27,35,42"

	// Syntax 1B - one file-specific label list
	// Example: 'raw' is "file1.tif:3,5,7"

	// Syntax 1C - one file-specific label 
	// Example: 'raw' is "file1.tif:27"
		
	// Syntax 2A - comma-separated labels (global blacklist)
	// Example: 'raw' is "27,35,42"

	// Syntax 2B - one label (global blacklist)
	// Example: 'raw' is "27"

	if (raw.find(':') != std::string::npos)
	{
		// Syntax 1:

		if (raw.find(';') != std::string::npos)
		{
			// Syntax 1A:
			std::vector<std::string> parts;
			Nyxus::parse_delimited_string(raw, ";", parts);
			for (std::string& p : parts)
			{
				if (!parse_file_blacklist(p))
					return false;
			}
		}
		else
		{
			// Syntax 1B
			if (!parse_file_blacklist(raw))
				return false;
		}
	}
	else
	{
		// Syntax 2:
		if (!Nyxus::parse_delimited_string_list_to_ints(raw, globalBlackList, ermsg))
			return false;
	}

	// Mark the blacklist as defined
	defined_ = true;
	return true;
}

bool RoiBlacklist::defined()
{
	return defined_;
}

bool operator == (const pairType & p1, const std::string & p2)
{
	return p1.first == p2;
}

bool RoiBlacklist::check_label_blacklisted (const std::string& fname, int roilabel)
{
	if (!defined())
		return false;

	if (globalBlackList.empty())
	{
		// use per-file list
		auto foundFname = std::find (fileBlackList.begin(), fileBlackList.end(), fname);
		if (foundFname == fileBlackList.end())
			return false;
		pairType& p = *foundFname;
		auto& rois = p.second;
		bool found = std::find(rois.begin(), rois.end(), roilabel) != rois.end();
		return found;
	}
	else
	{
		// use global list
		bool found = std::find(globalBlackList.begin(), globalBlackList.end(), roilabel) != globalBlackList.end();
		return found;
	}
}

std::string RoiBlacklist::get_summary_text()
{
	std::stringstream ss;
	if (! globalBlackList.empty())
	{
		ss << "global ";
		for (auto& i : globalBlackList)
			ss << i << " ";
	}
	else
	{
		for (auto& fbl : fileBlackList)
		{
			ss << fbl.first << " : ";
			for (auto& i : fbl.second)
				ss << i << " ";
		}
	}

	return ss.str();
}

void RoiBlacklist::clear()
{
	if (globalBlackList.size())
		globalBlackList.clear();
	if (fileBlackList.size())
		fileBlackList.clear();
	defined_ = false;
	ermsg = "";
}

std::string RoiBlacklist::get_last_er_msg()
{
	return ermsg;
}