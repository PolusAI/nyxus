#include <regex>
#include "strpat.h"

StringPattern::StringPattern()
{
}

// Example of a valid pat: BRATS_{d+}_z{set d+}_t{d+}.ome.tif for BRATS_001_z004_t002.ome.tif
bool StringPattern::set_filepattern (const std::string & pat)
{
	// parse a Polus-style filepattern 
	const std::string magicAnyStr = "mzmzmzmzmzmzmzmzmzmzmzm",	// a string highly unlikely to happen to be a part of file name
		magicAnyNum = "18446744073709551615" "000",	// int value that will never occur (max 64-bit int \times 10^3)
		magicStarNum = "18446744073709551615" "111";	// int value that will never occur (max 64-bit int \times 10^3 + 111)

	// replace all {d+} with NUM
	std::string repl1 = std::regex_replace (pat, std::regex("\\{d\\+\\}"), magicAnyNum);

	// replace all {c+} with TEXT
	std::string repl2 = std::regex_replace (repl1, std::regex("\\{c\\+\\}"), magicAnyStr);

	// replace all {set d+} or its variant {set,d+} with =*
	std::string repl3 = std::regex_replace (repl2, std::regex("\\{set d\\+\\}"), magicStarNum);
	std::string repl4 = std::regex_replace (repl3, std::regex("\\{set,d\\+\\}"), magicStarNum);

	// validate all the expressions in curly brackets 
	if (repl4.find("{") != std::string::npos || repl4.find("}") != std::string::npos)
	{
		ermsg_ = "illegal {Expression}. Only {d+}, {c+}, and {set d+} or {set,d+} are permitted";
		return false;
	}

	// now lexify this file pattern into a raw pattern to produce a grammar
	std::vector<std::string> tokCodes;
	std::vector<std::string> tokVals;

	bool ok = tokenize(
		repl4,
		tokCodes,
		tokVals);

	if (!ok)
		return false;

	std::string join;
	for (int i = 0; i < tokCodes.size(); i++)
	{
		if (tokVals[i] == magicAnyStr || tokVals[i] == magicAnyNum)
			join += tokCodes[i];
		else
			if(tokVals[i] == magicStarNum)
				join += tokCodes[i] + "=*";
			else
				join += tokCodes[i] + "=" + tokVals[i];
		join += " ";
	}

	ok = set_raw_pattern(join);
	
	return ok;
}

// initialize the file pattern object with a string
bool StringPattern::set_raw_pattern (const std::string& s)
{
	// Cache the pattern string no matter if it's correct or not
	cached_pattern_string = s;

	// Check if the pattern is meaningful
	if (!filepatt_to_grammar(s, grammar_, ermsg_))
	{
		good_ = false;
		return false;
	}

	// Success
	good_ = true;
	ermsg_.clear();
	return true;
}

// returns whether the file pattern is initialized and usable
bool StringPattern::good() const
{
	return good_;
}

// returns the last error message
std::string StringPattern::get_ermsg() const
{
	return ermsg_;
}

std::string StringPattern::get_term_context (const std::string & term)
{
	size_t idxFound = term.find("=");
	if (idxFound != std::string::npos)
	{
		std::string tc = term.substr(idxFound + 1, term.size() - 1);
		return tc;
	}
	else
		return "";
}

// returns true if a string matches the pattern
bool StringPattern::match (const std::string& s, std::map<std::string, std::vector<std::string>> & imgDirs, std::string & external_ermsg) const
{
	if (!good())
		return false;

	// parse the file name lexically
	std::vector<std::string> tokCodes;
	std::vector<std::string> tokVals;

	if (tokenize(s, tokCodes, tokVals) == false)
	{
		external_ermsg = "Error: cannot tokenize " + s;
		return false;
	}

	// check if 's' matches the grammar in the number of tokens
	if (tokCodes.size() != grammar_.size())
		return false;

	// check the file name string versus a grammar of 3D layout A
	std::string aggrValue;
	std::string mapKey;

	// check grammar

	for (int i = 0; i < grammar_.size(); i++)
	{
		auto term = grammar_[i];

		// do we have a grammer term with constant? 
		std::string pureTerm = term,
			termContext;

		// if we have a token with a qualifying constant, tear the token apart into a pure term and its context
		size_t idxFound = term.find("=");

		bool haveEq = false;
		if (idxFound != std::string::npos)
		{
			haveEq = true;
			pureTerm = term.substr(0, idxFound);
			termContext = term.substr(idxFound + 1, term.size() - 1);
		}

		// grammar check
		if (tokCodes[i] != pureTerm)
		{
			external_ermsg = "skipping " + mapKey + tokCodes[i] + " not ending " + pureTerm;
			return false;
		}

		// grammar tern coming with a qualifier?
		if (!haveEq)
		{
			mapKey += tokVals[i];
		}
		else
		{
			// token not matching the qualifier ?
			if (termContext != "*" && termContext != tokVals[i])
			{
				external_ermsg = "skipping " + mapKey + termContext + " not ending " + tokVals[i];
				return false;
			}

			// ok, matching

			if (pureTerm == t_TEXT)
			{
				mapKey += tokVals[i];
				//---state.push_back(termContext);	// for example "z"
				continue;
			}
			if (pureTerm == t_SEP)
			{
				mapKey += tokVals[i];
				continue;
			}
			if (pureTerm == t_NUM)
			{
				mapKey += (termContext == "*" ? termContext : tokVals[i]);
				aggrValue = tokVals[i]; // for example "0457" in "z0457"

				continue;
			}

		}
	} //- grammar walk

	// if we are at this point, syntax is OK. Now update filename's association with a value mined from a set term
	// make an aggregation action using an aggregator name-value(s) tuple cached in state
	auto imdir = imgDirs.find(mapKey);
	if (imdir == imgDirs.end())
	{
		std::vector zValues { aggrValue };
		imgDirs[mapKey] = zValues;
	}
	else
	{
		std::vector<std::string>& zValues = imdir->second;
		zValues.push_back (aggrValue);
	}

	return true;
}

bool StringPattern::tokenize (
	const std::string& s,
	std::vector<std::string>& tokCodes,
	std::vector<std::string>& tokVals) const
{
	tokCodes.clear();
	tokVals.clear();

	// use std::vector instead, we need to have it in this order
	std::vector<std::pair<std::string, std::string>> v
	{
		{ "[0-9]+" , t_NUM } ,
		{ "[a-z]+|[A-Z]+" , t_TEXT },
		{ "~|`|!|@|#|\\$|%|\\^|&|\\(|\\)|_|-|\\+|=|\\{|\\}|\\[|]|'|;|,|\\.", t_SEP },
		{ "\\*", t_STAR }
	};

	std::string reg;

	for (auto const& x : v)
		reg += "(" + x.first + ")|"; // parenthesize the submatches

	reg.pop_back();

	std::regex re(reg, std::regex::extended); // std::regex::extended for longest match

	auto words_begin = std::sregex_iterator(s.begin(), s.end(), re);
	auto words_end = std::sregex_iterator();

	for (auto it = words_begin; it != words_end; ++it)
	{
		size_t index = 0;

		for (; index < it->size(); ++index)
			if (!it->str(index + 1).empty()) // determine which submatch was matched
				break;

		tokCodes.push_back(v[index].second);
		tokVals.push_back(it->str());
	}

	// count tokens' characters to check if all the input string characters were digested
	size_t totLen = 0;
	for (const auto& v : tokVals)
		totLen += v.length();
	if (totLen == s.length())
		return true;
	else
		return false;
}

bool StringPattern::filepatt_to_grammar(const std::string& filePatt, std::vector<std::string>& grammar, std::string& errMsg)
{
	grammar.clear();
	const char* delimiters = " ";
	char* dupFP = strdup(filePatt.c_str());
	char* token = std::strtok(dupFP, delimiters);
	int n_aggrs = 0;
	while (token)
	{
		std::string strToken = token;

		// check 1: illegal terms
		if (strToken.find(t_TEXT) != 0 && strToken.find(t_NUM) != 0 && strToken.find(t_SEP) != 0)
		{
			errMsg = "error: " + strToken + " needs to be TEXT, NUM, or SEP";
			return false;
		}

		// check 2: unique aggregator
		std::string tc = get_term_context(strToken);
		if (tc == "*")
			n_aggrs++;

		// save
		grammar.push_back(strToken);
		token = std::strtok(nullptr, delimiters);
	}
	free(dupFP);

	// check 2: unique aggregator
	if (n_aggrs != 1)
	{
		errMsg = "error: aggregator needs to be unique (actual count is " + std::to_string(n_aggrs) + ")";
		return false;
	}

	return true;
}

std::string StringPattern::get_cached_pattern_string() const
{
	return cached_pattern_string;
}