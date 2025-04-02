#include "cli_anisotropy_options.h"
#include "helpers/helpers.h"

std::tuple<bool, std::optional<std::string>> AnisotropyOptions::parse_input()
{ 
	float val;

	if (!raw_aniso_x.empty())
	{
		if (!Nyxus::parse_as_float(raw_aniso_x, val))
			return { false, "Error in " + raw_aniso_x + ": expecting a real value" };
		aniso_x = val;
	}

	if (!raw_aniso_y.empty())
	{
		if (!Nyxus::parse_as_float(raw_aniso_y, val))
			return { false, "Error in " + raw_aniso_y + ": expecting a real value" };
		aniso_y = val;
	}

	if (!raw_aniso_z.empty())
	{
		if (!Nyxus::parse_as_float(raw_aniso_z, val))
			return { false, "Error in " + raw_aniso_z + ": expecting a real value" };
		aniso_z = val;
	}

	return { true, std::nullopt };
}
