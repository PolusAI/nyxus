#pragma once

#include <string>
#include <vector>

class GpuOptions
{
public:
	// parses "raw_use_gpu" and "raw_requested_device_ids"
	bool parse_input (std::string & ermsg);

	// true if the parameters have never been specified via "raw_*"
	bool empty();

	// accessor of the "using" status 
	void set_using_gpu(bool use);
	bool get_using_gpu();

	// accessor of active device ID
	bool set_single_device_id(int id);
	int get_single_device_id();

	// exposed to command line processor
	std::string raw_use_gpu;	
	std::string raw_requested_device_ids;

private:
	bool using_gpu_ = false;
	int best_device_id_ = -1;
};