#pragma once

#include <string>
#include <vector>

class FpImageOptions
{
public:
	// Parses 'raw*', set 'defined_' and 'ermsg'
	bool parse_input();

	// Returns true if all 'raw*' are empty
	bool empty();

	std::string get_summary_text();
	std::string get_last_er_msg();

	// intentionally public to be accessed by Environment
	std::string raw_min_intensity = "",	// matches FPIMAGE_MIN
		raw_max_intensity = "",			// matches FPIMAGE_MAX
		raw_target_dyn_range = "";		// matches FPIMAGE_TARGET_DYNRANGE

	// value-oriented getters
	float min_intensity() const 
	{ 
		return min_intensity_;  
	}
	float max_intensity() const 
	{ 
		return max_intensity_;  
	}
	float target_dyn_range() const 
	{ 
		return target_dyn_range_; 
	}

private:
	bool defined_ = false;
	std::string ermsg = "";

	float min_intensity_ = 0.0,
		max_intensity_ = 1.0,
		target_dyn_range_ = 1e4;
};