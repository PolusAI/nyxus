#include <algorithm>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "environment.h"
#include "featureset.h"

#include "features/basic_morphology.h"
#include "features/chords.h"
#include "features/convex_hull.h"
#include "features/erosion.h"
#include "features/caliper.h"
#include "features/circle.h"
#include "features/ellipse_fitting.h"
#include "features/euler_number.h"
#include "features/extrema.h"
#include "features/fractal_dim.h"
#include "features/geodetic_len_thickness.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/gldm.h"
#include "features/gldzm.h"
#include "features/glrlm.h"
#include "features/glszm.h"
#include "features/hexagonality_polygonality.h"
#include "features/2d_geomoments.h"
#include "features/intensity.h"
#include "features/neighbors.h"
#include "features/ngldm.h"
#include "features/ngtdm.h"
#include "features/radial_distribution.h"
#include "features/roi_radius.h"
#include "features/zernike.h"

#include "features/3d_surface.h"
#include "features/3d_glcm.h"
#include "features/3d_gldm.h"
#include "features/3d_ngldm.h"
#include "features/3d_ngtdm.h"
#include "features/3d_gldzm.h"
#include "features/3d_glrlm.h"
#include "features/3d_glszm.h"
#include "features/3d_intensity.h"
#include "features/3d_surface.h"

#include "features/focus_score.h"
#include "features/power_spectrum.h"
#include "features/saturation.h"
#include "features/sharpness.h"

#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "version.h"

// Parses the value of a <family>/greydepth metaparameter with the same rule as --coarseGrayDepth:
// returns the error message, or nothing once 'n_greys' holds the depth
static std::optional<std::string> parse_greydepth_metaparam (const std::string& raw, const std::string& name, int& n_greys)
{
	switch (Nyxus::parse_grey_depth (raw, n_greys))
	{
	case Nyxus::GreyDepthParse::not_an_integer:
		return "error: cannot parse value \"" + raw + "\" of " + name + ": expecting an integer";
	case Nyxus::GreyDepthParse::out_of_range:
		return "error: value \"" + raw + "\" of " + name + " is out of range: the magnitude must be at most " + std::to_string (INT_MAX);
	case Nyxus::GreyDepthParse::ok:
		break;
	}
	return std::nullopt;
}

std::optional<std::string> Environment::set_metaparam (const std::string & p_val)
{
	// parse LHS=RHS
	std::vector<std::string> eq_sides;
	Nyxus::parse_delimited_string (p_val, "=", eq_sides);
	if (eq_sides.size() != 2)
		return "syntax error in \"" + p_val + "\": expecting <paramName>=<paramVal>";

	// parse LHS
	std::vector<std::string> ppath;
	Nyxus::parse_delimited_string (eq_sides[0], "/", ppath);
	if (!(ppath.size() == 1 || ppath.size() == 2))
		return "syntax error in <paramName>=<paramVal> of \"" + p_val + "\": expecting <paramName> to be <feature name>/<parameter name> or <common parameter name>";

	// feature parameters
	if (ppath.size() == 2)
	{
		if (ppath[0] == "glcm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as a grey depth
				int n_greys;
				if (auto err = parse_greydepth_metaparam (eq_sides[1], "glcm/greydepth", n_greys))
					return err;
				STNGS_GLCM_GREYDEPTH(fsett_GLCM) = n_greys;
			}
			else if (ppath[1] == "offset")
			{
				// interpret eq_sides[1] as int
				int offs;
				if (Nyxus::parse_as_int(eq_sides[1], offs) == false)
				{
					return "error: cannot parse value \"" + eq_sides[1] + "\" of glcm/offset: expecting an integer";
				}
				STNGS_GLCM_OFFSET(fsett_GLCM) = offs;
			}
			else
			{
				return "error: unrecognized feature parameter of feature glcm: \"" + ppath[1] + "\"";
			}
		}
		else
		if (ppath[0] == "3glcm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as a grey depth
				int n_greys;
				if (auto err = parse_greydepth_metaparam (eq_sides[1], "3glcm/greydepth", n_greys))
					return err;
				STNGS_GLCM_GREYDEPTH (fsett_D3_GLCM) = n_greys;
			}
			else if (ppath[1] == "offset")
			{
				// interpret eq_sides[1] as int
				int offs;
				if (Nyxus::parse_as_int(eq_sides[1], offs) == false)
				{
					return "error: cannot parse value \"" + eq_sides[1] + "\" of 3glcm/offset: expecting an integer";
				}
				STNGS_GLCM_OFFSET (fsett_D3_GLCM) = offs;
			}
			else if (ppath[1] == "numang")
			{
				// interpret eq_sides[1] as int
				int n_angs;
				if (Nyxus::parse_as_int(eq_sides[1], n_angs) == false)
				{
					return "error: cannot parse value \"" + eq_sides[1] + "\" of 3glcm/numang: expecting an integer";
				}
				STNGS_GLCM_NUMANG(fsett_D3_GLCM) = n_angs;
			}
			else if (ppath[1] == "sparseintensities")
			{
				// interpret eq_sides[1] as boolean
				bool bval;
				if (Nyxus::parse_as_bool(eq_sides[1], bval) == false)
				{
					return "error: cannot parse value \"" + eq_sides[1] + "\" of 3glcm/sparseintensities: expecting a boolean (\"true\" or \"false\")";
				}
				STNGS_GLCM_SPARSEINTENS(fsett_D3_GLCM) = bval;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3glcm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3gldm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as a grey depth
				int n_greys;
				if (auto err = parse_greydepth_metaparam (eq_sides[1], "3gldm/greydepth", n_greys))
					return err;
				STNGS_GLDM_GREYDEPTH (fsett_D3_GLDM) = n_greys;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3gldm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3ngtdm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as a grey depth
				int n_greys;
				if (auto err = parse_greydepth_metaparam (eq_sides[1], "3ngtdm/greydepth", n_greys))
					return err;
				STNGS_NGTDM_GREYDEPTH(fsett_D3_NGTDM) = n_greys;
			}
			else
			if (ppath[1] == "radius")
			{
				// interpret eq_sides[1] as int
				int radius = -1;
				if (Nyxus::parse_as_int(eq_sides[1], radius) == false || radius <= 0)
				{
					return "error: cannot parse value \"" + eq_sides[1] + "\" of 3ngtdm/radius: expecting a positive integer";
				}
				STNGS_NGTDM_RADIUS(fsett_D3_NGTDM) = radius;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3gldm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3glrlm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as a grey depth
				int n_greys;
				if (auto err = parse_greydepth_metaparam (eq_sides[1], "3glrlm/greydepth", n_greys))
					return err;
				STNGS_GLRLM_GREYDEPTH (fsett_D3_GLRLM) = n_greys;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3glrlm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3glszm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as a grey depth
				int n_greys;
				if (auto err = parse_greydepth_metaparam (eq_sides[1], "3glszm/greydepth", n_greys))
					return err;
				STNGS_GLSZM_GREYDEPTH(fsett_D3_GLSZM) = n_greys;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3glrlm: \"" + ppath[1] + "\"";
			}
		}
		else
		{
			return "error: unrecognized feature \"" + ppath[0] + "\"";
		}
	}

	return std::nullopt;
}

std::optional<std::string> Environment::get_metaparam (double & p_val, const std::string& p_name)
{
	// parse LHS
	std::vector<std::string> ppath;
	Nyxus::parse_delimited_string(p_name, "/", ppath);
	if (!(ppath.size() == 1 || ppath.size() == 2))
		return "syntax error in <paramName>=<paramVal> of \"" + p_name + "\": expecting <paramName> to be <feature name>/<parameter name> or <common parameter name>";

	// feature parameters
	if (ppath.size() == 2)
	{
		if (ppath[0] == "glcm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLCM_GREYDEPTH(fsett_GLCM);
				p_val = (double)ival;
			}
			else if (ppath[1] == "offset")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLCM_OFFSET(fsett_GLCM);
				p_val = (double)ival;
			}
			else
			{
				return "error: unrecognized feature parameter of feature glcm: \"" + ppath[1] + "\"";
			}
		}
		else
		if (ppath[0] == "3glcm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLCM_GREYDEPTH(fsett_GLCM);
				p_val = (double) ival;
			}
			else if (ppath[1] == "offset")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLCM_OFFSET(fsett_GLCM);
				p_val = (double)ival;
			}
			else if (ppath[1] == "numang")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLCM_NUMANG(fsett_GLCM);
				p_val = (double)ival;
			}
			else if (ppath[1] == "sparseintensities")
			{
				// interpret eq_sides[1] as boolean
				bool bval = STNGS_GLCM_SPARSEINTENS(fsett_GLCM);
				p_val = (double)bval;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3glcm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3gldm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLDM_GREYDEPTH (fsett_D3_GLDM);
				p_val = (double)ival;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3gldm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3ngtdm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_NGTDM_GREYDEPTH (fsett_D3_NGTDM);
				p_val = (double)ival;
			}
			else
			if (ppath[1] == "radius")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_NGTDM_RADIUS (fsett_D3_NGTDM);
				p_val = (double)ival;
			}			
			else
			{
				return "error: unrecognized feature parameter of feature 3gldm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3glrlm")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLRLM_GREYDEPTH (fsett_D3_GLRLM);
				p_val = (double)ival;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3glrlm: \"" + ppath[1] + "\"";
			}
		}
		else if (ppath[0] == "3glsz")
		{
			// check feature-specific parameter name
			if (ppath[1] == "greydepth")
			{
				// interpret eq_sides[1] as int
				int ival = STNGS_GLSZM_GREYDEPTH (fsett_D3_GLSZM);
				p_val = (double)ival;
			}
			else
			{
				return "error: unrecognized feature parameter of feature 3glszm: \"" + ppath[1] + "\"";
			}
		}
		else
		{
			return "error: unrecognized feature \"" + ppath[0] + "\"";
		}
	}

	return std::nullopt;
}