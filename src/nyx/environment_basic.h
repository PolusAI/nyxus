#pragma once
#include <string>

class BasicEnvironment
{
public:
	BasicEnvironment();

	std::string get_file_pattern();

	void set_file_pattern(const std::string & pat);
	bool check_file_pattern(const std::string& pat);

	int get_verbosity_level();
	void set_verbosity_level(int vl);

	std::string get_temp_dir_path() const;
protected:
	std::string file_pattern = ".*";
	int verbosity_level = 1; // 0 = silent
	std::string temp_dir_path = "";
};
