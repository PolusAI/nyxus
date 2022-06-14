#include <iostream>
#include <regex>
#include "environment_basic.h"

BasicEnvironment::BasicEnvironment()
{}

bool BasicEnvironment::check_file_pattern(const std::string& pat)
{
	try
	{
		std::regex re(pat);
	}
	catch (...)
	{
		std::cerr << "Exception checking file pattern " << pat << "\n";
		return false;
	}

	return true;
}

std::string BasicEnvironment::get_file_pattern()
{
	return file_pattern;
}

void BasicEnvironment::set_file_pattern (const std::string& pat)
{
	file_pattern = pat;
}

int BasicEnvironment::get_verbosity_level()
{
	return verbosity_level;
}

void BasicEnvironment::set_verbosity_level (int vl)
{
	verbosity_level = vl;
}