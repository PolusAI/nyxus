#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "cli_gabor_options.h"
#include "features/gabor.h"

bool GaborOptions::parse_input()
{
	if (!rawFreqs.empty())
	{
		// string -> vector of doubles
		std::vector<double> x;
		if (!Nyxus::parse_delimited_string_list_to_doubles(rawFreqs, x, ermsg))
			return false;

		// set feature class parameter
		GaborFeature::f0 = x;	
	}

	if (!rawGamma.empty())
	{
		// string -> real
		float x;
		bool ok = parse_as_float(rawGamma, x);
		if (!ok || x <= 0)
		{
			ermsg = "Error in " + rawGamma + ": expecting a positive real value";
			return false;
		}

		// set feature class parameter
		GaborFeature::gamma = x;
	}

	if (!rawSig2lam.empty())
	{
		// string -> real
		float x;
		bool ok = parse_as_float(rawSig2lam, x);
		if (!ok || x <= 0)
		{
			ermsg = "Error in " + rawSig2lam + ": expecting a positive real value";
			return false;
		}

		// set feature class parameter
		GaborFeature::sig2lam = x;
	}

	if (!rawKerSize.empty())
	{
		// string -> integer
		int x;
		bool ok = parse_as_int (rawKerSize, x);
		if (!ok || x <= 0)
		{
			ermsg = "Error in " + rawKerSize + ": expecting a positive integer value";
			return false;
		}

		// set feature class parameter
		GaborFeature::n = x;
	}

	if (!rawF0.empty())
	{
		// string -> real
		float x;
		bool ok = parse_as_float(rawF0, x);
		if (!ok || x <= 0)
		{
			ermsg = "Error in " + rawF0 + ": expecting a positive real value";
			return false;
		}

		// set feature class parameter
		GaborFeature::f0LP = x;
	}

	if (!rawTheta.empty())
	{
		// string -> real
		float x;
		bool ok = parse_as_float(rawTheta, x);
		if (!ok || x < 0 || x > 180)
		{
			ermsg = "Error in " + rawTheta + ": expecting a real value in [0,180]";
			return false;
		}

		// set feature class parameter (in radians)
		GaborFeature::theta = x/180.0 * M_PI;
	}

	if (!rawGrayThreshold.empty())
	{
		// string -> real
		float x;
		bool ok = parse_as_float(rawGrayThreshold, x);
		if (!ok)
		{
			ermsg = "Error in " + rawGrayThreshold + ": expecting a real value";
			return false;
		}

		// set feature class parameter
		GaborFeature::GRAYthr = x;
	}

	return true;
}

bool GaborOptions::empty()
{
	return rawFreqs.empty() &&
		rawGamma.empty() &&
		rawSig2lam.empty() &&
		rawKerSize.empty() &&
		rawF0.empty() &&
		rawTheta.empty() &&
		rawGrayThreshold.empty();
}

std::string GaborOptions::get_summary_text()
{
	std::string s, eq = "=", sep = "\n";

	if (!rawFreqs.empty())
		s += GABOR_FREQS + eq + rawFreqs + sep;

	if (!rawGamma.empty())
		s += GABOR_GAMMA + eq + rawGamma + sep;

	if (!rawSig2lam.empty())
		s += GABOR_SIG2LAM + eq + rawSig2lam + sep;

	if (!rawKerSize.empty())
		s += GABOR_KERSIZE + eq + rawKerSize + sep;

	if (!rawF0.empty())
		s += GABOR_F0 + eq + rawF0 + sep;

	if (!rawTheta.empty())
		s += GABOR_THETA + eq + rawTheta + sep;

	if (!rawGrayThreshold.empty())
		s += GABOR_THRESHOLD + eq + rawGrayThreshold + sep;

	return s;
}

std::string GaborOptions::get_last_er_msg()
{
	return ermsg;
}

