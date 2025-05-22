#pragma once
#include <map>
#include <string>
#include <vector>

class StringPattern
{
public:
	StringPattern();

	// Returns true if 'p' is a file pattern of a 3D image in 
	// layout-A (example: BRATS_{d+}_z{set d+}_t{d+}.ome.tif).
	// The alternative file pattern is regex pattern (example: *\.nii\.gz)
	bool is_layoutA_fpattern (const std::string& p) const;

	inline bool is_25D() const { return is_layoutA_fpattern(cached_pattern_string); }

	// Initialize the instance using a Polus-stype filepattern (example: BRATS_{d+}_z{set d+}_t{d+}.ome.tif)
	// Error details are available via get_ermsg()
	bool set_filepattern(const std::string & pat);

	// Initialize the instance using anexplicit definition (example: "TEXT=BRATS SEP=_ NUM SEP=_ TEXT=z NUM=* SEP=_ TEXT=t NUM SEP=. TEXT=ome SEP=. TEXT=tif")
	// Error details are available via get_ermsg()
	bool set_raw_pattern(const std::string & pat);

	// Returns whether the file pattern is initialized and usable
	bool good() const;

	// Returns the last error message
	std::string get_ermsg() const;

	// Returns true if a string matches the pattern
	bool match (const std::string& s, std::map<std::string, std::vector<std::string>>& imgDirs, std::string& external_ermsg) const;

	std::string get_cached_pattern_string() const;

protected:
	std::string cached_pattern_string;
	bool good_ = false;
	std::string ermsg_;
	std::vector<std::string> grammar_;

	// If successful, sets tokCodes and tokVals
	bool tokenize(
		const std::string & s, 
		std::vector<std::string> & tokCodes,
		std::vector<std::string> & tokVals) const;

	bool filepatt_to_grammar (const std::string& filePatt, std::vector<std::string>& grammar, std::string& errMsg);

	std::string get_term_context (const std::string& term);

private:
	// Terminals
	const char *t_TEXT = "TEXT", *t_NUM = "NUM", *t_SEP = "SEP", *t_STAR = "STAR";

};
