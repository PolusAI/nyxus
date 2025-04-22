#pragma once

#include <optional>
#include <tuple>
#include <string>
#include <vector>
#include "helpers/helpers.h"

class AnisotropyOptions
{
public:

	// true if the parameters have never been specified via raw_<whatever>
	bool nothing2parse() { return raw_aniso_x.empty() && raw_aniso_y.empty() && raw_aniso_z.empty(); }

	// true if x-, y, or z-anisotropy is non-default
	bool customized() { return customized_; }

	// getters
	double get_aniso_x() { return aniso_x; }
	double get_aniso_y() { return aniso_y; }
	double get_aniso_z() { return aniso_z; }

	// setters (Python API scenario)
	void set_aniso_x(double a) { if (!Nyxus::near_eq(a, 1.0)) { aniso_x = a; customized_ = true; } }
	void set_aniso_y(double a) { if (!Nyxus::near_eq(a, 1.0)) { aniso_y = a; customized_ = true; } }
	void set_aniso_z(double a) { if (!Nyxus::near_eq(a, 1.0)) { aniso_z = a; customized_ = true; } }

	// helper for CLI scenarios
	std::string get_summary_text() { return std::to_string(aniso_x) + "," + std::to_string(aniso_y) + "," + std::to_string(aniso_z); }

	// parses raw_<whatever> into corresponding aniso_<whatever>. In case of an error returns false and an error details string
	std::tuple<bool, std::optional<std::string>> parse_input();

	// exposed to command line processor (CLI scenario), supposed to be followed by parse_input()
	std::string raw_aniso_x, raw_aniso_y, raw_aniso_z;

private:
	bool customized_ = false;
	double aniso_x = 1, aniso_y = 1, aniso_z = 1;
};