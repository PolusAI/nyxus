#include <iostream>
#include <regex>
#include "environment_basic.h"
#include "helpers/fsystem.h"

BasicEnvironment::BasicEnvironment()
{
	temp_dir_path = fs::temp_directory_path().string();

	// Add slash to path if needed
	if (!temp_dir_path.empty() && temp_dir_path.back() != fs::path::preferred_separator) {
        temp_dir_path += fs::path::preferred_separator;
    }
}

bool BasicEnvironment::check_2d_file_pattern(const std::string& pat)
{
	try
	{
		std::regex re(pat);
	}
	catch (...)
	{
		std::cerr << "Exception while checking file pattern " << pat << "\n";
		return false;
	}

	return true;
}

bool BasicEnvironment::check_3d_file_pattern(const std::string& pat)
{
	return file_pattern_3D.set_filepattern (pat);
}

std::string BasicEnvironment::get_file_pattern()
{
	return rawFilePattern;
}

void BasicEnvironment::set_file_pattern (const std::string& pat)
{
	rawFilePattern = pat;
}

int BasicEnvironment::get_verbosity_level()
{
	return verbosity_level;
}

void BasicEnvironment::set_verbosity_level (int vl)
{
	verbosity_level = vl;
}

std::string BasicEnvironment::get_temp_dir_path() const
{
	return temp_dir_path;
}