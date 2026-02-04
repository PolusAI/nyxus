#pragma once

#include <string>
#include <vector>

class GLCMoptions
{
public:
	// Parses 'raw*', set 'defined_' and 'ermsg'
	bool parse_input();

	std::string get_last_er_msg();

	bool empty();

	// intentionally public to be accessed by Environment
	std::string rawOffs = "",	// matches GLCMOFFSET
		rawAngles = "";			// matches GLCMANGLES

	int offset_ = 1;
	std::vector<int> glcmAngles = { 0, 45, 90, 135 };

private:
	bool defined_ = false;
	std::string ermsg = "";
};