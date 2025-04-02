#pragma once

#include <optional>
#include <tuple>
#include <string>
#include <vector>

class AnisotropyOptions
{
public:
	// parses "raw_use_gpu" and "raw_requested_device_ids"
	std::tuple<bool, std::optional<std::string>> parse_input();

	// true if the parameters have never been specified via "raw_*"
	bool empty() { return ! (raw_aniso_x.empty() || raw_aniso_y.empty() || raw_aniso_z.empty()); }

	std::string get_summary_text() { return std::to_string(aniso_x) + "," + std::to_string(aniso_y) + "," + std::to_string(aniso_z); }

	// accessors
	bool specified() { return aniso_specified; }
	double get_aniso_x() { return aniso_x; }
	double get_aniso_y() { return aniso_y; }
	double get_aniso_z() { return aniso_z; }

	// accessor of the "using" status 
	//void set_using_gpu(bool use);
	//bool get_using_gpu();

	// accessor of active device ID
	//bool set_single_device_id(int id);
	//int get_single_device_id();

	// exposed to command line processor
	std::string raw_aniso_x, raw_aniso_y, raw_aniso_z;

private:
	bool aniso_specified = false;
	double aniso_x = 1, aniso_y = 1, aniso_z = 1;
};