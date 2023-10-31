#include "cli_fpimage_options.h"
#include "environment.h"
#include "helpers/helpers.h"

bool FpImageOptions::parse_input()
{
	if (!raw_min_intensity.empty())
	{
		// string -> real
		float x;
		bool ok = Nyxus::parse_as_float (raw_min_intensity, x);
		if (!ok)
		{
			ermsg = "Error in " + raw_min_intensity + ": expecting a real value";
			return false;
		}

		// set feature class parameter
		min_intensity_ = x;
	}

	if (!raw_max_intensity.empty())
	{
		// string -> real
		float x;
		bool ok = Nyxus::parse_as_float(raw_max_intensity, x);
		if (!ok)
		{
			ermsg = "Error in " + raw_max_intensity + ": expecting a real value";
			return false;
		}

		// set feature class parameter
		max_intensity_ = x;
	}

	if (min_intensity_ >= max_intensity_)
	{
		ermsg = "Error: the minimum " + std::to_string(min_intensity_) + " must be smaller than maximum " + std::to_string(max_intensity_);
		return false;
	}

	if (!raw_target_dyn_range.empty())
	{
		// string -> integer
		int x;
		bool ok = Nyxus::parse_as_int (raw_target_dyn_range, x);
		if (!ok || x <= 0)
		{
			ermsg = "Error in " + raw_target_dyn_range + ": expecting a positive integer value";
			return false;
		}

		// set feature class parameter
		target_dyn_range_ = x;
	}

	return true;
}

bool FpImageOptions::empty()
{
	return raw_max_intensity.empty() 
		&& raw_min_intensity.empty() 
		&& raw_target_dyn_range.empty();
}

std::string FpImageOptions::get_summary_text()
{
	std::string s, eq = "=", sep = "\n";

	if (!raw_min_intensity.empty())
		s += FPIMAGE_MIN + eq + raw_min_intensity + sep;

	if (!raw_max_intensity.empty())
		s += FPIMAGE_MAX + eq + raw_max_intensity + sep;

	if (!raw_target_dyn_range.empty())
		s += FPIMAGE_TARGET_DYNRANGE + eq + raw_target_dyn_range + sep;

	return s;
}

std::string FpImageOptions::get_last_er_msg()
{
	return ermsg;
}
