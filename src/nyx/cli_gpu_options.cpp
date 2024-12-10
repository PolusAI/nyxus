#include "cli_gpu_options.h"
#include "helpers/helpers.h"

namespace NyxusGpu
{
	bool get_best_device(
		// in
		const std::vector<int>& devIds,
		// out
		int& best_id,
		std::string& lastCuErmsg);
}

bool GpuOptions::empty()
{
	return raw_use_gpu.empty();
}

void GpuOptions::set_using_gpu(bool use)
{
	using_gpu_ = use;
}

bool GpuOptions::get_using_gpu()
{
	return using_gpu_;
}

bool GpuOptions::set_single_device_id(int id)
{
	if (get_using_gpu())
	{
		this->best_device_id_ = id;
		return true;
	}
	else
		return false;
}

int GpuOptions::get_single_device_id()
{
	return best_device_id_;
}

bool GpuOptions::parse_input (std::string & ermsg)
{
	auto u = Nyxus::toupper (this->raw_use_gpu);
	if (u.length() == 0)
	{
		set_using_gpu (false);
	}
	else
	{
		auto t = Nyxus::toupper("true"),
			f = Nyxus::toupper("false");
		if (u != t && u != f)
		{
			ermsg = "valid values are " + t + " or " + f;
			return false;
		}
		set_using_gpu (u == t);

		// process user's GPU device choice
		if (get_using_gpu())
		{
			std::vector<int> devIds;

			if (! this->raw_requested_device_ids.empty())
			{
				// user input -> vector of IDs
				std::vector<std::string> S;
				Nyxus::parse_delimited_string (this->raw_requested_device_ids, ",", S);
				// examine those IDs
				for (const auto& s : S)
				{
					if (!s.empty())
					{
						// string -> int
						int id;
						if (sscanf (s.c_str(), "%d", &id) != 1 || id < 0)
						{
							ermsg = s + ": expecting a non-negative integer";
							return false;
						}
						devIds.push_back (id);
					}
				}
			}
			else
				devIds.push_back (0); // user did not requested a specific ID, so default it to 0

			// given a set of suggested devices, choose the least memory-busy one
			int best_id = -1;
			std::string lastCuErmsg;
			if (! NyxusGpu::get_best_device(devIds, best_id, lastCuErmsg))
			{
				ermsg = "cannot use any of GPU devices in " + Nyxus::virguler<int> (devIds) + " due to " + lastCuErmsg;
				return false;
			}

			// we found at least one workable
			this->best_device_id_ = best_id;
		}
	}

	return true;
}