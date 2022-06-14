#pragma once
#include <string>

namespace Nyxus
{
	enum AvailableChildFeatureAggregations { aNONE = 0, aSUM, aMEAN, aMIN, aMAX, aWMA };
}

class ChildFeatureAggregation
{
public:
	ChildFeatureAggregation()
	{
		method = Nyxus::AvailableChildFeatureAggregations::aNONE;
	}

	ChildFeatureAggregation(const char* _option_name)
	{
		option_name = _option_name;
	}

	bool parse(const char* psz_raw_aggr_option)
	{
		std::string raw_aggregate_option(psz_raw_aggr_option);
		if (raw_aggregate_option.length() == 0)
			return false;

		std::string sMethod;
		bool ok = find_string_argument(raw_aggregate_option, option_name.c_str(), sMethod);
		if (!ok)
			return false;
		else
		{
			if (sMethod == "SUM")
				method = Nyxus::AvailableChildFeatureAggregations::aSUM;
			else
				if (sMethod == "MEAN")
					method = Nyxus::AvailableChildFeatureAggregations::aMEAN;
				else
					if (sMethod == "MIN")
						method = Nyxus::AvailableChildFeatureAggregations::aMIN;
					else
						if (sMethod == "MAX")
							method = Nyxus::AvailableChildFeatureAggregations::aMAX;
						else
							if (sMethod == "WMA")
								method = Nyxus::AvailableChildFeatureAggregations::aWMA;
							else
							{
								std::cerr << "\nUnrecognized aggregation method " << sMethod << " ! Defaulting to no aggregation\n";
								method = Nyxus::AvailableChildFeatureAggregations::aNONE;
								return false;
							}
			return true;
		}
	}

	static std::string get_valid_options()
	{
		return "SUM, MEAN, MIN, MAX, or WMA";
	}

	Nyxus::AvailableChildFeatureAggregations get_method() const
	{
		return method;
	}

	std::string get_method_string() const
	{
		switch (method)
		{
		case Nyxus::AvailableChildFeatureAggregations::aNONE:	return "NONE";
		case Nyxus::AvailableChildFeatureAggregations::aSUM:		return "SUM";
		case Nyxus::AvailableChildFeatureAggregations::aMEAN:	return "MEAN";
		case Nyxus::AvailableChildFeatureAggregations::aMIN:		return "MIN";
		case Nyxus::AvailableChildFeatureAggregations::aMAX:		return "MAX";
		case Nyxus::AvailableChildFeatureAggregations::aWMA:		return "WMA";
		}
		return "UNKNOWN";
	}

private:
	Nyxus::AvailableChildFeatureAggregations method = Nyxus::AvailableChildFeatureAggregations::aNONE;
	std::string option_name;	// initialized via constructor

	bool find_string_argument(std::string& raw, const char* arg, std::string& arg_value)
	{
		std::string a = arg;
		// Syntax #2 <arg>=<value>
		a.append("=");
		auto pos = raw.find(a);
		if (pos != std::string::npos)
		{
			arg_value = raw.substr(a.length());
			return true;
		}

		// Argument was not recognized
		return false;
	}
};

