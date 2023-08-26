#include "cli_gabor_options.h"
#include "features/gabor.h"

bool GaborOptions::parse_input()
{
	// OK to have both angles and frequencies specified but not one of them
	if (!rawTheta.empty() && !rawFreqs.empty())
	{
		std::vector<double> t;
		if (!Nyxus::parse_delimited_string_list_to_doubles(rawTheta, t, ermsg))
		{
			ermsg = "Error in " + rawTheta + ": expecting a list of real values";
			return false;
		}

		std::vector<double> f;
		if (!Nyxus::parse_delimited_string_list_to_doubles(rawFreqs, f, ermsg))
		{
			ermsg = "Error in " + rawFreqs + ": expecting a list of real values";
			return false;
		}

		// Check lengths
		if (t.size() != f.size())
		{
			ermsg = "Error: frequency and angle lists must me of same size. Received thetas=" + rawTheta + " freqs=" + rawFreqs;
			return false;
		}

		// All the checks passed, now cache the f0/theta pairs in the feature class
		GaborFeature::f0_theta_pairs.clear();	// clear the previously cached pairs
		int n = t.size();
		for (int i = 0; i < n; i++)
		{
			std::pair p (f[i], deg2rad(t[i]));	// angle in radians
			GaborFeature::f0_theta_pairs.push_back(p);
		}
	}
	else
		if (rawTheta.empty() != rawFreqs.empty())
		{
			ermsg = "Error: frequency and angle lists are allowed to be both empty or non-empty";
			return false;
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

