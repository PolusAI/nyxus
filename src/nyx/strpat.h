#pragma once
#include <map>
#include <string>
#include <vector>

class StringPattern
{
public:
	StringPattern();

	// initialize the file pattern object with a string
	bool set_pattern(const std::string& s);

	// returns whether the file pattern is initialized and usable
	bool good() const;

	// returns the last error message
	std::string get_ermsg() const;

	// returns true if a string matches the pattern
	bool match (const std::string& s, std::map<std::string, std::vector<std::string>>& imgDirs, std::string& external_ermsg) const;

	std::string get_cached_pattern_string() const;

protected:
	std::string cached_pattern_string;
	bool good_ = false;
	std::string ermsg_;
	std::vector<std::string> grammar_;

	// if successful, sets tokCodes and tokVals
	bool tokenize(
		const std::string & s, 
		std::vector<std::string> & tokCodes,
		std::vector<std::string> & tokVals) const;

	bool filepatt_to_grammar (const std::string& filePatt, std::vector<std::string>& grammar, std::string& errMsg);

};
