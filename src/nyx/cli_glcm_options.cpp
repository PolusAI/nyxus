#include "cli_glcm_options.h"
#include "features/glcm.h"

using namespace Nyxus;

bool GLCMoptions::parse_input()
{
	if (!rawAngles.empty())
	{
		//==== Parse rotations
		if (!rawAngles.empty())
		{
			std::string ermsg;
			if (!Nyxus::parse_delimited_string_list_to_ints(rawAngles, glcmAngles, ermsg))
			{
				std::cerr << "Error parsing list of integers " << rawAngles << ": " << ermsg << "\n";
				return false;
			}

			// The angle list parsed well, let's tell it to GLCMFeature 
			GLCMFeature::angles = glcmAngles;
		}
	}

	if (!rawOffs.empty())
	{
		// string -> real
		int x;
		bool ok = Nyxus::parse_as_int (rawOffs, x);
		if (!ok || x <= 0)
		{
			ermsg = "Error in " + rawOffs + ": expecting a positive integer value";
			return false;
		}

		// set feature class parameter
		offset_ = x;
	}

	return true;
}

bool GLCMoptions::empty()
{
	return rawAngles.empty() && rawOffs.empty();
}

std::string GLCMoptions::get_last_er_msg()
{
	return ermsg;
}

