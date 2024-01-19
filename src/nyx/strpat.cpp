#include <regex>
#include "strpat.h"

StringPattern::StringPattern()
{
}

// initialize the file pattern object with a string
bool StringPattern::set_pattern(const std::string& s)
{
	// Cache the pattern string no matter if it's correct or not
	cached_pattern_string = s;

	// Check if the pattern is meaningful
	if (!filepatt_to_grammar(s, grammar_, ermsg_))
	{
		good_ = false;
		ermsg_ = "tokenize error";
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

	// check the file name string versus a grammar of 3D layout A
	std::vector<std::string> state;
	std::string mapKey;

	// check grammar

	for (int i = 0; i < grammar_.size(); i++)
	{
		auto term = grammar_[i];

		// do we have a grammer term with constant? 
		std::string pureTerm = term,
			termContext;

		// if we have a token with a qualifying constant, tear the token off 
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
			external_ermsg = "after " + mapKey + " expecting " + pureTerm + " while actual is " + tokCodes[i] + " (" + tokVals[i] + "), so skipping file " + s;
			return false;
		}

		// grammar tern coming with a qualifier?
		if (!haveEq)
		{
			mapKey += tokVals[i];
		}
		else
		{
			if (pureTerm == "TEXT")
			{
				mapKey += tokVals[i];
				state.push_back(termContext);	// for example "z"
				continue;
			}
			if (pureTerm == "NUM")
			{
				mapKey += (termContext == "*" ? termContext : tokVals[i]);
				state.push_back(tokVals[i]);	// for example "457"
				continue;
			}
			if (pureTerm == "#")
			{
				// merge the rest of the input string with the mapping key and quit traversing the grammar
				for (int j = i; j < tokCodes.size(); j++)
					mapKey += tokCodes[i];
				break;	// quit the grammar check
			}
		}
	} //- grammar walk

	// if we are at this point, syntax is OK. Now update filename's association with a value mined from a set term
	// no match with an aggregator?
	if (state.size() == 0)
	{
		external_ermsg = "expecting " + s + " to contain an aggregator";
		return false; 
	}
	// incomplete aggregator match?
	if (state.size() == 1)
	{
		external_ermsg = "incomplete aggregator in " + s;
		return false;
	}
	// make an aggregation action using an aggregator name-value(s) tuple cached in state
	if (state[0] == "z")
	{
		auto imdir = imgDirs.find(mapKey);
		if (imdir == imgDirs.end())
		{
			std::vector zValues{ state[1] };
			imgDirs[mapKey] = zValues;
		}
		else
		{
			std::vector<std::string>& zValues = imdir->second;
			zValues.push_back(state[1]);
		}
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
		{ "[0-9]+" , "NUM" } ,
		{ "[a-z]+|[A-Z]+" , "TEXT" },
		{ "~|`|!|@|#|\\$|%|\\^|&|\\(|\\)|_|-|\\+|=|\\{|\\}|\\[|]|'|;|,|\\.", "SEP" }
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
	const char* delimiters = "_ -";
	char* dupFP = strdup(filePatt.c_str());
	char* token = std::strtok(dupFP, delimiters);
	while (token)
	{
		std::string strToken = token;

		// check
		if (strToken.find("TEXT") != 0 && strToken.find("NUM") != 0 && strToken.find("SEP") != 0)
		{
			errMsg = "error: " + strToken + " needs to be TEXT, NUM, or SEP";
			return false;
		}

		// save
		grammar.push_back(strToken);
		token = std::strtok(nullptr, delimiters);
	}
	free(dupFP);

	return true;
}

std::string StringPattern::get_cached_pattern_string() const
{
	return cached_pattern_string;
}