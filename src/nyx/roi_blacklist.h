#pragma once

#include <string>
#include <vector>

class pairType : public std::pair<std::string, std::vector<int>> {};

class RoiBlacklist
{
public:
	bool parse_raw_string(const std::string &);
	bool defined();
	bool check_label_blacklisted(const std::string& fname, int roilabel);
	std::string get_summary_text();
	void clear();
	std::string get_last_er_msg();

private:
	bool defined_ = false;
	std::string ermsg;
	std::vector<int> globalBlackList; 
	std::vector<pairType> fileBlackList;
	bool parse_file_blacklist (const std::string &);
};