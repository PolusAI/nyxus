#pragma once
#include <string>
#include "strpat.h"

class BasicEnvironment
{
public:
	BasicEnvironment();
	
	std::string get_file_pattern();
	
	void set_file_pattern(const std::string & pat);

	// Checks file pattern correctness
	bool check_2d_file_pattern (const std::string & pat);
	bool check_3d_file_pattern (const std::string & pat);

	int get_verbosity_level();
	void set_verbosity_level(int vl);

	std::string get_temp_dir_path() const;

	StringPattern file_pattern_3D;

protected:
	std::string rawFilePattern = ".*";
	int verbosity_level = 1; // 0 = silent
	std::string temp_dir_path = "";
};