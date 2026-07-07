#pragma once

#include <string>
#include <vector>

class FpImageOptions
{
public:
	// Parses 'raw*', set 'defined_' and 'ermsg'
	bool parse_input();

	// Returns true if all 'raw*' are empty
	bool empty() const;

	std::string get_summary_text();
	std::string get_last_er_msg();

	// intentionally public to be accessed by Environment
	std::string raw_min_intensity = "",	// matches FPIMAGE_MIN
		raw_max_intensity = "",			// matches FPIMAGE_MAX
		raw_target_dyn_range = "",		// matches FPIMAGE_TARGET_DYNRANGE
		raw_preserve_hu = "";			// matches clo_PRESERVE_HU (TRUE/FALSE/T/F)

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

	void set_min_intensity(float min_intensity) {
		min_intensity_ = min_intensity;
	}

	void set_max_intensity(float max_intensity) {
		max_intensity_ = max_intensity;
	}

	void set_target_dyn_range(float target_dyn_range) {
		target_dyn_range_ = target_dyn_range;
	}

	// CT/HU preservation mode: keep 1 grey level == 1 intensity unit (offset by the
	// floored global min) instead of min-max rescaling; see SlideProps::uint_friendly_inten.
	bool preserve_hu() const {
		return preserve_hu_;
	}

	void set_preserve_hu(bool v) {
		preserve_hu_ = v;
	}

private:
	bool defined_ = false;
	std::string ermsg = "";

	float min_intensity_ = 0.0,
		max_intensity_ = 1.0,
		target_dyn_range_ = 1e4;
	bool preserve_hu_ = false;			// HU mode off by default
};